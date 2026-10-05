"""Three-way memory comparison on one base: SGD-trained vs D2L-compiled vs retrieval.

Every arm answers the SAME questions through the SAME free-form grader
(`tofu_freeform_v3`), so no arm gets a private evaluator. Per-item hit arrays and
raw generations are stored for every cell so all comparisons can be paired
after the fact.

ARMS
  base          unmodified Qwen3-4B-Instruct-2507
  sgd7          SGD LoRA, 7 modules, rank 16, 10 epochs, 200 facts
  sgddp         SGD LoRA, down_proj only, rank 16, 10 epochs, 200 facts
                (matched to D2L's module set, isolating SGD-vs-generated from
                 architecture)
  d2l_pa        D2L compiled, per-author routing: each item answered by the
                adapter compiled from THAT author's ~20 facts
  d2l_all       D2L compiled, one adapter from all 200 facts (7303 tokens)
  retrieval     base + top-k facts from the same 200-fact corpus in the prompt

CONDITIONS
  trained/q     the 200 facts each mechanism was given, original question
  trained/pq    the same 200 facts, paraphrased question   -> paraphrase drop
  near          TOFU held-out authors 10-19, facts NO mechanism was given
                -> does installing memory make it fabricate?
  far           general knowledge, filtered to what the base answers correctly
                -> is the assistant still intact?

d2l_pa is the only arm needing per-item routing, since the correct adapter
depends on the item's author. Implemented by grouping items by author and
issuing each group against its own served LoRA name.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/work/tofu_exp")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests  # noqa: E402
from far_probe import FAR_ITEMS, filter_answerable, grade_far  # noqa: E402
from tofu_freeform_v3 import (  # noqa: E402
    FreeFormEvaluator, grade_contains, grade_discrim, grade_keyfacts,
    keyfact_tokens, wilson,
)


def chat_prompt(tok, question, context=None):
    sysmsg = ("You are a helpful assistant. Answer the question factually and concisely. "
              "Reply with the answer only, in one short sentence.")
    if context:
        sysmsg += "\n\nUse the following facts to answer:\n" + "\n".join(
            "- " + c for c in context)
    msgs = [{"role": "system", "content": sysmsg},
            {"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8106")
    ap.add_argument("--base-name", default="qwen4bi")
    ap.add_argument("--tokenizer", default="/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--ceiling-dir", default="/work/tofu_exp/data_v3/ceiling")
    ap.add_argument("--heldout-dir", default="/work/tofu_exp/data_v3/heldout")
    ap.add_argument("--out-json", default="/work/d2l_exp/results/threeway.json")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    comp = args.url + "/v1/completions"
    test = [json.loads(l) for l in open(os.path.join(args.ceiling_dir, "test.jsonl"))]
    near = [json.loads(l) for l in open(os.path.join(args.heldout_dir, "test.jsonl"))]
    facts = json.load(open(os.path.join(args.ceiling_dir, "facts_small.json")))
    if args.limit:
        test, near = test[:args.limit], near[:args.limit]

    ev = FreeFormEvaluator(comp, args.tokenizer, facts, k=args.k,
                           max_tokens=args.max_tokens)
    tok = ev.tok

    # Reuse the exact grader gates from the main experiment. If the discriminative
    # keyfact filter ever regresses, fail here rather than emit pretty numbers.
    fl_n = fl_k = 0
    for it in test:
        h, _, g = grade_keyfacts(it["perturbed"][0], it["true"], it["para_true"],
                                 it["perturbed"])
        fl_k += h
        fl_n += g
    floor = fl_k / fl_n if fl_n else 0.0
    print("grader floor (wrong answers) %.4f  [must be <0.05]" % floor, flush=True)
    assert floor < 0.05, floor

    ARMS = {
        "base":      {"model": args.base_name, "rag": False, "route": None},
        "sgd7":      {"model": "sgd7",         "rag": False, "route": None},
        "sgddp":     {"model": "sgddp",        "rag": False, "route": None},
        "d2l_pa":    {"model": None,           "rag": False, "route": "author"},
        "d2l_all":   {"model": "d2l_all200",   "rag": False, "route": None},
        "retrieval": {"model": args.base_name, "rag": True,  "route": None},
    }

    def model_for(arm, item):
        spec = ARMS[arm]
        if spec["route"] == "author":
            return "d2l_a%02d" % item["author_idx"]
        return spec["model"]

    def gen_one(model, prompt):
        r = requests.post(comp, json={
            "model": model, "prompt": prompt, "max_tokens": args.max_tokens,
            "temperature": 0.0, "stop": ["\n\n"]}, timeout=300)
        r.raise_for_status()
        return r.json()["choices"][0]["text"].strip()

    results = {}
    prompt_tokens = {}

    # ---------------- trained facts: q and pq ----------------
    for qkey in ("q", "pq"):
        for arm in ARMS:
            def work(it, arm=arm, qkey=qkey):
                ctx = ev.retrieve(it[qkey]) if ARMS[arm]["rag"] else None
                p = chat_prompt(tok, it[qkey], ctx)
                g = gen_one(model_for(arm, it), p)
                return p, g

            t0 = time.time()
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                out = list(pool.map(work, test))
            el = time.time() - t0

            gens = [g for _, g in out]
            ptoks = [len(tok(p).input_ids) for p, _ in out]
            prompt_tokens["%s/%s" % (arm, qkey)] = sum(ptoks) / len(ptoks)

            cells = {"contains": [], "keyfacts": [], "discrim": []}
            grad = []
            for it, g in zip(test, gens):
                cells["contains"].append(grade_contains(g, it["true"], it["para_true"]))
                kf, _, gd = grade_keyfacts(g, it["true"], it["para_true"], it["perturbed"])
                cells["keyfacts"].append(kf if gd else -1)
                grad.append(gd)
                cells["discrim"].append(grade_discrim(g, it["true"], it["para_true"],
                                                      it["perturbed"]))
            ng = sum(grad)
            kacc = sum(h for h in cells["keyfacts"] if h > 0) / ng if ng else None
            lo, hi = wilson(sum(h for h in cells["keyfacts"] if h > 0), ng)
            key = "%s/%s" % (arm, qkey)
            results[key] = {
                "arm": arm, "cond": "trained", "qkey": qkey, "n": len(test),
                "keyfacts_acc": kacc, "keyfacts_n_gradable": ng,
                "keyfacts_ci95": [round(lo, 4), round(hi, 4)],
                "contains_acc": sum(cells["contains"]) / len(test),
                "discrim_acc": sum(cells["discrim"]) / len(test),
                "hits": cells, "gens": gens,
                "mean_prompt_tokens": prompt_tokens[key],
                "seconds": round(el, 1),
            }
            print("[trained %-3s] %-9s keyfacts %.3f  contains %.3f  discrim %.3f  ptok %.1f"
                  % (qkey, arm, kacc or 0, results[key]["contains_acc"],
                     results[key]["discrim_acc"], prompt_tokens[key]), flush=True)

    # ---------------- NEAR interference: held-out authors ----------------
    # Adapters were given authors 0-9 only. For d2l_pa there is no "correct"
    # adapter for a held-out author, so we route to author00 as a fixed stand-in:
    # the question being asked is what an installed personal adapter does to
    # someone else's question, and any of the 10 would serve. Recorded so the
    # choice is visible.
    for arm in ARMS:
        def workn(it, arm=arm):
            ctx = ev.retrieve(it["q"]) if ARMS[arm]["rag"] else None
            p = chat_prompt(tok, it["q"], ctx)
            m = "d2l_a00" if ARMS[arm]["route"] == "author" else ARMS[arm]["model"]
            return gen_one(m, p)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            gens = list(pool.map(workn, near))
        kf_hits, grad = [], []
        for it, g in zip(near, gens):
            h, _, gd = grade_keyfacts(g, it["true"], it["para_true"], it["perturbed"])
            kf_hits.append(h if gd else -1)
            grad.append(gd)
        ng = sum(grad)
        acc = sum(h for h in kf_hits if h > 0) / ng if ng else None
        results["%s/near" % arm] = {
            "arm": arm, "cond": "near", "n": len(near),
            "keyfacts_acc": acc, "keyfacts_n_gradable": ng,
            "hits": {"keyfacts": kf_hits}, "gens": gens,
            "note": "d2l_pa routed to d2l_a00 (no correct adapter exists)",
        }
        print("[near      ] %-9s keyfacts %.3f" % (arm, acc or 0), flush=True)

    # ---------------- FAR interference: general knowledge ----------------
    kept, dropped = filter_answerable(
        lambda q: gen_one(args.base_name, chat_prompt(tok, q, None)))
    print("FAR: base answers %d/%d correctly; evaluating on those %d"
          % (len(kept), len(FAR_ITEMS), len(kept)), flush=True)
    for q, _, _ in dropped:
        print("  dropped (base wrong):", q, flush=True)

    for arm in ARMS:
        def workf(item, arm=arm):
            q, acc, _ = item
            ctx = ev.retrieve(q) if ARMS[arm]["rag"] else None
            p = chat_prompt(tok, q, ctx)
            m = "d2l_a00" if ARMS[arm]["route"] == "author" else ARMS[arm]["model"]
            g = gen_one(m, p)
            return grade_far(g, acc), g

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            out = list(pool.map(workf, kept))
        hits = [h for h, _ in out]
        results["%s/far" % arm] = {
            "arm": arm, "cond": "far", "n": len(kept),
            "far_acc": sum(hits) / len(hits) if hits else None,
            "hits": {"far": hits}, "gens": [g for _, g in out],
            "questions": [q for q, _, _ in kept],
        }
        print("[far       ] %-9s acc %.3f" % (arm, results["%s/far" % arm]["far_acc"] or 0),
              flush=True)

    blob = {
        "base_model": "Qwen3-4B-Instruct-2507",
        "arms": {k: {kk: vv for kk, vv in v.items() if kk != "route"}
                 for k, v in ARMS.items()},
        "n_trained_items": len(test), "n_near_items": len(near),
        "n_far_items": len(kept), "far_dropped": [q for q, _, _ in dropped],
        "grader_floor": floor,
        "cells": results,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(blob, f, indent=2)
    print("wrote", args.out_json, flush=True)


if __name__ == "__main__":
    main()
