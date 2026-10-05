"""Evaluate every condition of the gradient granularity x exposure sweep.

Wraps 18_eval_granularity.py's exact evaluation logic rather than
reimplementing it, so the gradient sweep is graded by the same free-form
keyfacts grader, the same router object, the same prompt template and the same
items as the frozen compiled ladder. Only the adapter names change.

Cell keys are "g<g>e<e>/<routing>/<qkey>" so that granularity and exposure are
both recoverable from a single merged results file.

Two routing conditions per cell, as in the compiled run:
  oracle     route on the item's ground-truth partition (isolates memory from
             routing)
  predicted  route with 16_router.PartitionRouter on question text only

--repeat re-runs only the step-matched endpoints (g=20 and g=200) into a
separate file, to measure this arm's own run-to-run noise floor. Multi-LoRA
decoding here is not bitwise reproducible at temperature 0, so differences
below that floor must be reported as unresolved rather than as effects.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/work/tofu_exp")
sys.path.insert(0, "/work/d2l_test")
sys.path.insert(0, HERE)

import requests  # noqa: E402
from tofu_freeform_v3 import (  # noqa: E402
    FreeFormEvaluator, grade_contains, grade_discrim, grade_keyfacts, wilson,
)

_spec = importlib.util.spec_from_file_location(
    "router16", os.path.join(HERE, "16_router.py"))
router16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(router16)


def chat_prompt(tok, question):
    """Identical to 18_eval_granularity.chat_prompt."""
    sysmsg = ("You are a helpful assistant. Answer the question factually and concisely. "
              "Reply with the answer only, in one short sentence.")
    msgs = [{"role": "system", "content": sysmsg},
            {"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def main():
    ap = argparse.ArgumentParser()
    # 8107, not the frozen run's 8106: see 26_serve_sgd_sweep.sh. The 8106
    # server is still up and serves none of the sweep adapters, so pointing here
    # by default prevents a silent wrong-server evaluation.
    ap.add_argument("--url", default="http://localhost:8107")
    ap.add_argument("--tokenizer", default="/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--data", default="/work/tofu_exp/data_v3/ceiling")
    ap.add_argument("--stats", default="/work/d2l_exp/sgd_gran_build_stats.json")
    ap.add_argument("--out-json",
                    default="/work/d2l_exp/results/granularity_sgd_sweep.json")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--only", default="", help="comma list of cond keys g<g>/e<e>")
    ap.add_argument("--repeat", action="store_true",
                    help="evaluate only step-matched endpoints (noise floor)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    comp = args.url + "/v1/completions"
    rows = [json.loads(l) for l in open(os.path.join(args.data, "test.jsonl"))]
    facts = json.load(open(os.path.join(args.data, "facts_small.json")))
    build = json.load(open(args.stats))
    conds = build["conditions"]

    ev = FreeFormEvaluator(comp, args.tokenizer, facts, k=5,
                           max_tokens=args.max_tokens)
    tok = ev.tok

    def gen_one(model, prompt):
        r = requests.post(comp, json={
            "model": model, "prompt": prompt, "max_tokens": args.max_tokens,
            "temperature": 0.0, "stop": ["\n\n"]}, timeout=300)
        r.raise_for_status()
        return r.json()["choices"][0]["text"].strip()

    if args.resume and os.path.exists(args.out_json):
        out = json.load(open(args.out_json))
    else:
        out = {"base_model": "Qwen3-4B-Instruct-2507",
               "arm": "sgd_granularity_x_exposure", "cells": {}}
    out.setdefault("cells", {})

    # Two stats schemas are supported, distinguished by the presence of
    # "epochs". The exposure sweep (sgd_gran_build_stats.json) is keyed
    # g<g>/e<e> and carries "epochs"; the exact-249 ladder
    # (sgd_exact249_build_stats.json) is keyed g<g> and carries
    # "optimizer_steps_per_adapter" with no whole-epoch count, because it stops
    # mid-epoch by construction. Cell keys become g<g>e<e> and g<g>s<steps>
    # respectively, so a merged results file stays unambiguous.
    def is_exact_ladder(c):
        return "epochs" not in c

    def cell_prefix(c):
        g = c["facts_per_adapter"]
        if is_exact_ladder(c):
            return "g%ds%d" % (g, c["optimizer_steps_per_adapter"])
        return "g%de%d" % (g, c["epochs"])

    want = sorted(conds)
    if args.only:
        keep = set(args.only.split(","))
        want = [k for k in want if k in keep]
    if args.repeat:
        # Endpoints only, for the run-to-run noise floor. On the exposure sweep
        # that is the nearest-epoch-to-reference cell; on the exact ladder every
        # cell is at 249 so the endpoints are simply g=20 and g=200.
        want = [k for k in want
                if conds[k]["facts_per_adapter"] in (20, 200)
                and (is_exact_ladder(conds[k])
                     or conds[k].get("is_step_matched"))]

    print("evaluating %d conditions" % len(want), flush=True)

    for ckey in want:
        c = conds[ckey]
        g = c["facts_per_adapter"]
        e = c.get("epochs")
        parts = router16.build_partitions(rows, facts, g)
        router = router16.PartitionRouter(parts, facts, rule="max")
        gold = [router16.gold_partition(parts, i) for i in range(len(rows))]
        # served name = the lora_name recorded at build time, so eval can never
        # address a different adapter than the one that was trained
        served = [c["adapters"][p["name"]]["lora_name"] for p in parts]

        for qkey in ("q", "pq"):
            pred = [router.route(r[qkey]) for r in rows]
            for routing, sel in (("oracle", gold), ("predicted", pred)):
                key = "%s/%s/%s" % (cell_prefix(c), routing, qkey)
                if args.resume and key in out["cells"]:
                    print("  skip %s (done)" % key, flush=True)
                    continue
                names = [served[s] for s in sel]
                t0 = time.time()
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    gens = list(pool.map(
                        lambda t: gen_one(t[0], chat_prompt(tok, t[1][qkey])),
                        zip(names, rows)))
                el = time.time() - t0

                hits = {"contains": [], "keyfacts": [], "discrim": []}
                grad = []
                for it, gtext in zip(rows, gens):
                    hits["contains"].append(
                        grade_contains(gtext, it["true"], it["para_true"]))
                    kf, _, gd = grade_keyfacts(gtext, it["true"], it["para_true"],
                                               it["perturbed"])
                    hits["keyfacts"].append(kf if gd else -1)
                    grad.append(gd)
                    hits["discrim"].append(
                        grade_discrim(gtext, it["true"], it["para_true"],
                                      it["perturbed"]))
                ng = sum(grad)
                k = sum(h for h in hits["keyfacts"] if h > 0)
                lo, hi = wilson(k, ng)
                out["cells"][key] = {
                    "facts_per_adapter": g, "epochs": e,
                    "epochs_equivalent": c.get("epochs_equivalent", e),
                    "optimizer_steps_per_adapter": c["optimizer_steps_per_adapter"],
                    "ladder": "exact_step" if is_exact_ladder(c) else "exposure",
                    "is_exact_249": c["optimizer_steps_per_adapter"] == 249,
                    "is_step_matched": c.get("is_step_matched"),
                    "n_adapters": len(parts), "routing": routing, "qkey": qkey,
                    "n": len(rows),
                    "keyfacts_acc": k / ng if ng else None,
                    "keyfacts_n_gradable": ng,
                    "keyfacts_ci95": [round(lo, 4), round(hi, 4)],
                    "contains_acc": sum(hits["contains"]) / len(rows),
                    "discrim_acc": sum(hits["discrim"]) / len(rows),
                    "hits": hits, "gens": gens,
                    "selected_partition": sel, "gold_partition": gold,
                    "router_correct": [int(a == b) for a, b in zip(sel, gold)],
                    "seconds": round(el, 1),
                }
                print("[g=%-3d %-6s %-9s %-2s] keyfacts %.3f  contains %.3f  "
                      "router %.3f" % (
                          g,
                          ("s%d" % c["optimizer_steps_per_adapter"])
                          if e is None else ("e%d" % e),
                          routing, qkey,
                          out["cells"][key]["keyfacts_acc"] or 0,
                          out["cells"][key]["contains_acc"],
                          sum(out["cells"][key]["router_correct"]) / len(rows)),
                      flush=True)

        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote %s (%d cells)" % (args.out_json, len(out["cells"])), flush=True)


if __name__ == "__main__":
    main()
