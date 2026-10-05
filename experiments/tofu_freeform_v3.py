"""Free-form factual evaluation: the model writes its own answer.

WHY THIS EXISTS

The forced-choice evaluator (`tofu_eval2x2_v3.py`) scores candidate answer
STRINGS by mean per-token logprob. On the ceiling condition the adapter was
trained on exactly one surface form of the answer (`true`), so when the gold
candidate is swapped to `para_true` the adapter loses ~28 points at every scale
and every epoch setting. That drop is real, but it is not by itself evidence of
factual-memory failure: training raises the probability of the learned string, so
a scorer that compares string probabilities is measuring the thing training
directly optimized. A model that knows the fact perfectly but has been biased
toward one phrasing would produce the same number.

The clean control is to stop scoring OUR strings and let the model produce its
own, then judge whether the produced answer carries the right fact. Under that
rule, answer-surface bias is not penalized: if the adapter says the fact in its
learned words, it is correct, and if it says the same fact in other words, it is
also correct. Only losing the fact counts as a loss.

DESIGN

Both question forms are tested (`q` and `pq`), so this is the question-side
control as well. There is no answer-side factor here by construction: the model
is not shown any candidate, so "answer wording" is an output property, not an
experimental condition. That asymmetry is the point. The 2x2 becomes a 1x2.

GRADING

Free-form grading is where this kind of experiment usually goes wrong, so three
graders are computed for every generation and all three are stored per item:

  `contains`   normalized substring / token-subset containment of the gold answer
               or its paraphrase. Strict, high precision, low recall. Fails on
               correct answers phrased differently.

  `keyfacts`   recall of the content tokens that BOTH `true` and `para_true`
               share. Tokens common to the two independent phrasings of the same
               fact are the fact-bearing ones (names, numbers, titles); tokens in
               only one are phrasing. This grader is therefore paraphrase-robust
               BY CONSTRUCTION rather than by a hand-written synonym list.

  `discrim`    the forced-choice check applied to the model's OWN text: is the
               generation closer to the gold answer than to each of the five
               TOFU perturbed (factually wrong) answers, by token-F1? This keeps
               a direct link to the perturbed distractors the main experiment
               uses, so free-form and forced-choice numbers are commensurable.

`keyfacts` is the headline because it is the only one that is both
paraphrase-robust and independent of the perturbed set. The others are reported
so the conclusion can be checked against a stricter and a looser rule; a finding
that flips between graders is a grading artifact and must not be claimed.

Generations are stored verbatim so any grader can be re-run offline without GPU.
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

QKEYS = ["q", "pq"]

# Tokens that carry no fact and would inflate overlap-based graders.
STOP = set("""a an the of in on at to for with by from as is are was were be been being
and or but if then than that this these those it its it's he she they them his her their
what which who whom whose when where why how do does did done have has had not no nor
about into over under between during before after above below up down out off again
further once here there all any both each few more most other some such only own same
so too very can will just should now s t don won isn aren wasn weren hasn haven hadn
doesn didn shouldn wouldn couldn mustn i you we me my your our""".split())


def norm(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def toks(s):
    return [w for w in norm(s).split() if w]


def content(s):
    return set(w for w in toks(s) if w not in STOP and len(w) > 1)


def token_f1(a, b):
    ta, tb = content(a), content(b)
    if not ta or not tb:
        return 0.0
    ov = len(ta & tb)
    if ov == 0:
        return 0.0
    p, r = ov / len(ta), ov / len(tb)
    return 2 * p * r / (p + r)


def grade_contains(gen, true, para):
    """Strict: the gold answer (or its paraphrase) appears in the generation."""
    g = norm(gen)
    for gold in (true, para):
        if norm(gold) and norm(gold) in g:
            return 1
        gt = content(gold)
        if gt and gt.issubset(content(gen)):
            return 1
    return 0


def keyfact_tokens(true, para, perturbed=None):
    """The tokens that identify this fact and distinguish it from a wrong answer.

    Two filters, in order:

    1. PARAPHRASE-INVARIANT: keep only content tokens present in BOTH `true` and
       `para_true`. Tokens appearing in just one phrasing are wording, not fact.

    2. DISCRIMINATIVE: drop tokens that also appear in EVERY TOFU perturbed
       answer. This filter is not optional. TOFU builds its perturbations as
       minimal edits of the true answer, so the two share most of their content
       tokens; without this filter the grader scores 0.405 on answers that are
       factually wrong by construction (measured on both splits, N=200), which
       would credit an arm for reproducing the boilerplate around the fact
       rather than the fact. With it, the same check drops to near zero.

    Both filters can in principle empty the set (a fact whose distinguishing
    token appears only in one phrasing). Those items are returned empty and are
    EXCLUDED from the denominator by `grade_keyfacts` rather than being scored
    as wrong, since an ungradable item is not a failure of the model.
    """
    common = content(true) & content(para)
    if perturbed:
        shared_by_all_wrong = set.intersection(*[content(p) for p in perturbed])
        disc = common - shared_by_all_wrong
        return disc
    return common


def grade_keyfacts(gen, true, para, perturbed=None, thresh=0.7):
    """Returns (hit, recall, gradable). `hit` is only meaningful if gradable."""
    kf = keyfact_tokens(true, para, perturbed)
    if not kf:
        return 0, 0.0, 0
    got = content(gen)
    rec = len(kf & got) / len(kf)
    return int(rec >= thresh), rec, 1


def grade_discrim(gen, true, para, perturbed):
    """Is the generation closer to the fact than to every factually-wrong option?

    Scored against max(true, para) so answering in either phrasing is fine.
    """
    gold = max(token_f1(gen, true), token_f1(gen, para))
    worst = max((token_f1(gen, p) for p in perturbed), default=0.0)
    return int(gold > worst)


class FreeFormEvaluator:
    def __init__(self, url, tokenizer_path, facts, k=5, timeout=300,
                 max_tokens=64, temperature=0.0):
        # Imported here, not at module scope, so the pure grading functions
        # above can be imported and re-run on stored generations with no GPU
        # stack installed.
        import requests
        from transformers import AutoTokenizer

        self._requests = requests
        self.url = url
        self.tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.facts = facts
        self.k = k
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._fact_words = [set(re.findall(r"[a-z0-9]+", f.lower())) for f in facts]

    def build_prompt(self, question, context=None):
        # Identical system prompt to the forced-choice evaluator, plus a brevity
        # instruction. Without it the SLMs emit paragraphs, which inflates
        # containment-style graders by sheer coverage. The instruction is the
        # same for every arm, so it cannot favour one.
        sysmsg = ("You are a helpful assistant. Answer the question factually and concisely. "
                  "Reply with the answer only, in one short sentence.")
        if context:
            sysmsg += "\n\nUse the following facts to answer:\n" + "\n".join("- " + c for c in context)
        msgs = [{"role": "system", "content": sysmsg},
                {"role": "user", "content": question}]
        return self.tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True,
                                            enable_thinking=False)

    def retrieve(self, question):
        qw = set(re.findall(r"[a-z0-9]+", question.lower()))
        scored = sorted(range(len(self.facts)), key=lambda i: -len(qw & self._fact_words[i]))
        return [self.facts[i] for i in scored[:self.k]]

    def generate(self, model, prompt):
        r = self._requests.post(self.url, json={
            "model": model, "prompt": prompt,
            "max_tokens": self.max_tokens, "temperature": self.temperature,
            "stop": ["\n\n"],
        }, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["text"].strip()

    def eval_item(self, model, item, use_rag, qkey):
        q = item[qkey]
        ctx = self.retrieve(q) if use_rag else None
        gen = self.generate(model, self.build_prompt(q, ctx))
        c = grade_contains(gen, item["true"], item["para_true"])
        kf, rec, gradable = grade_keyfacts(gen, item["true"], item["para_true"],
                                           item["perturbed"])
        d = grade_discrim(gen, item["true"], item["para_true"], item["perturbed"])
        return {
            "contains": c, "keyfacts": kf, "keyfact_recall": round(rec, 4),
            "keyfacts_gradable": gradable, "discrim": d,
            "numeric": bool(re.search(r"\d", item["true"])),
            "gen": gen,
        }


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def run_cell(ev, name, model, test, use_rag, qkey, workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        out = list(pool.map(lambda it: ev.eval_item(model, it, use_rag, qkey), test))
    n = len(out)
    res = {"arm": name, "model": model, "qkey": qkey, "use_rag": use_rag, "n": n}
    for g in ("contains", "discrim"):
        k = sum(o[g] for o in out)
        lo, hi = wilson(k, n)
        res[g] = {"correct": k, "acc": k / n, "ci95": [round(lo, 4), round(hi, 4)],
                  "hits": [o[g] for o in out]}
    # keyfacts is scored only over gradable items (those with at least one
    # paraphrase-invariant, perturbation-discriminating token). `hits` keeps
    # full length with -1 for ungradable so it stays index-aligned with the
    # other graders and with the forced-choice `hits` arrays.
    gr = [i for i, o in enumerate(out) if o["keyfacts_gradable"]]
    k = sum(out[i]["keyfacts"] for i in gr)
    lo, hi = wilson(k, len(gr)) if gr else (0.0, 0.0)
    res["keyfacts"] = {
        "correct": k, "n_gradable": len(gr), "acc": (k / len(gr)) if gr else None,
        "ci95": [round(lo, 4), round(hi, 4)],
        "hits": [o["keyfacts"] if o["keyfacts_gradable"] else -1 for o in out],
    }
    res["mean_keyfact_recall"] = (
        sum(out[i]["keyfact_recall"] for i in gr) / len(gr)) if gr else None
    res["gens"] = [o["gen"] for o in out]
    print("[%-9s] q=%-3s  keyfacts %.3f (n=%d)  contains %.3f  discrim %.3f  (mean kf-recall %.3f)"
          % (name, qkey, res["keyfacts"]["acc"] or 0.0, len(gr), res["contains"]["acc"],
             res["discrim"]["acc"], res["mean_keyfact_recall"] or 0.0), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get(
        "TOFU_BASE_URL", "http://localhost:8000") + "/v1/completions")
    ap.add_argument("--base-name", required=True)
    ap.add_argument("--adapter-name", default=None)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--corpus", default="padded", choices=["padded", "small"])
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    test = [json.loads(l) for l in open(os.path.join(args.data_dir, "test.jsonl"))]
    if args.limit:
        test = test[:args.limit]
    facts = json.load(open(os.path.join(args.data_dir, "facts_%s.json" % args.corpus)))

    dq = [i for i, it in enumerate(test) if it["q"].strip() == it["pq"].strip()]
    assert not dq, "identical q/pq at items %s" % dq[:5]

    # The keyfact grader is only meaningful if the two phrasings genuinely share
    # fact-bearing tokens. Report the distribution so a degenerate grader is
    # visible in the output rather than silently scoring everything correct.
    sizes = [len(keyfact_tokens(it["true"], it["para_true"], it["perturbed"])) for it in test]
    ungradable = sum(1 for s in sizes if s == 0)
    print("discriminative keyfact tokens/item: min %d  mean %.1f  max %d | ungradable %d (%.1f%%)"
          % (min(sizes), sum(sizes) / len(sizes), max(sizes),
             ungradable, 100.0 * ungradable / len(test)), flush=True)

    # Grader floor, computed on this exact split before any generation: feeding
    # the graders a factually WRONG answer must score near zero. An earlier
    # version without the discriminative filter scored 0.405 here, which would
    # have inflated every arm. Fail loudly rather than produce a pretty number.
    fl_n = fl_k = 0
    for it in test:
        h, _, g = grade_keyfacts(it["perturbed"][0], it["true"], it["para_true"],
                                 it["perturbed"])
        fl_k += h
        fl_n += g
    floor = fl_k / fl_n if fl_n else 0.0
    ceil_k = sum(grade_keyfacts(it["para_true"], it["true"], it["para_true"],
                                it["perturbed"])[0] for it in test)
    print("grader check: wrong-answer floor %.3f (must be <0.05) | paraphrase ceiling %.3f "
          "(must be 1.000)" % (floor, ceil_k / fl_n if fl_n else 0.0), flush=True)
    assert floor < 0.05, "keyfact grader accepts wrong answers at %.3f" % floor
    assert ceil_k == fl_n, "keyfact grader rejects correct paraphrases"

    ev = FreeFormEvaluator(args.url, args.tokenizer, facts, k=args.k,
                           max_tokens=args.max_tokens)
    print("=== FREEFORM %s | data=%s | corpus=%s(%d) | N=%d ===" % (
        args.tag or args.base_name, args.data_dir, args.corpus, len(facts), len(test)),
        flush=True)

    results = []
    for qkey in QKEYS:
        print("--- question=%s ---" % qkey, flush=True)
        results.append(run_cell(ev, "base", args.base_name, test, False, qkey, args.workers))
        if args.adapter_name:
            results.append(run_cell(ev, "adapter", args.adapter_name, test, False,
                                    qkey, args.workers))
        results.append(run_cell(ev, "retrieval", args.base_name, test, True,
                                qkey, args.workers))

    blob = {
        "design": "freeform_generation",
        "graders": ["contains", "keyfacts", "discrim"],
        "headline_grader": "keyfacts",
        "tag": args.tag, "base": args.base_name, "adapter": args.adapter_name,
        "data_dir": args.data_dir, "corpus": args.corpus, "n_facts": len(facts),
        "n_items": len(test), "k": args.k, "max_tokens": args.max_tokens,
        "keyfact_tokens_per_item_mean": sum(sizes) / len(sizes),
        "cells": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(blob, f, indent=2)
    print("\nwrote", args.out_json, flush=True)


if __name__ == "__main__":
    sys.exit(main())
