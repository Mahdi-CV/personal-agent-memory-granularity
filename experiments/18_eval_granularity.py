"""Evaluate the granularity ladder under BOTH oracle and predicted routing.

Two routing conditions per granularity, over the same 200 items and the same
free-form pipeline every other arm used:

  oracle     route on the item's ground-truth partition. This is the upper
             bound: it isolates the memory question from the routing question.
  predicted  route with 16_router.PartitionRouter, question text only. This is
             the deployable number.

Running both on identical items makes the routing tax directly measurable and
attributable: items where the two conditions disagree are exactly the items the
router sent to the wrong adapter, so the accuracy lost to routing can be
separated from accuracy lost to memory pressure instead of inferred.

The SAME router instance routes the SGD arm too (19_eval_router_sgd), so no
routing advantage can accrue to either memory method.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/work/tofu_exp")
sys.path.insert(0, "/work/d2l_test")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib.util  # noqa: E402
import requests  # noqa: E402
from tofu_freeform_v3 import (  # noqa: E402
    FreeFormEvaluator, grade_contains, grade_discrim, grade_keyfacts, wilson,
)

_spec = importlib.util.spec_from_file_location(
    "router16", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "16_router.py"))
router16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(router16)


def chat_prompt(tok, question):
    sysmsg = ("You are a helpful assistant. Answer the question factually and concisely. "
              "Reply with the answer only, in one short sentence.")
    msgs = [{"role": "system", "content": sysmsg},
            {"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8106")
    ap.add_argument("--tokenizer", default="/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--data", default="/work/tofu_exp/data_v3/ceiling")
    ap.add_argument("--stats", default="/work/d2l_exp/gran_build_stats.json")
    ap.add_argument("--out-json", default="/work/d2l_exp/results/granularity.json")
    ap.add_argument("--lora-prefix", default="gr_",
                    help="served LoRA name = prefix + partition name")
    ap.add_argument("--arm", default="d2l")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    comp = args.url + "/v1/completions"
    rows = [json.loads(l) for l in open(os.path.join(args.data, "test.jsonl"))]
    facts = json.load(open(os.path.join(args.data, "facts_small.json")))
    build = json.load(open(args.stats))

    ev = FreeFormEvaluator(comp, args.tokenizer, facts, k=5,
                           max_tokens=args.max_tokens)
    tok = ev.tok

    def gen_one(model, prompt):
        r = requests.post(comp, json={
            "model": model, "prompt": prompt, "max_tokens": args.max_tokens,
            "temperature": 0.0, "stop": ["\n\n"]}, timeout=300)
        r.raise_for_status()
        return r.json()["choices"][0]["text"].strip()

    out = {"base_model": "Qwen3-4B-Instruct-2507", "arm": args.arm, "cells": {}}

    for g in router16.GRANULARITIES:
        gk = str(g)
        if gk not in build["granularities"]:
            print("skip g=%d (not built)" % g, flush=True)
            continue
        parts = router16.build_partitions(rows, facts, g)
        router = router16.PartitionRouter(parts, facts, rule="max")
        gold = [router16.gold_partition(parts, i) for i in range(len(rows))]

        for qkey in ("q", "pq"):
            pred = [router.route(r[qkey]) for r in rows]
            for routing, sel in (("oracle", gold), ("predicted", pred)):
                names = [args.lora_prefix + parts[s]["name"] for s in sel]
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
                key = "g%d/%s/%s" % (g, routing, qkey)
                out["cells"][key] = {
                    "facts_per_adapter": g, "n_adapters": len(parts),
                    "routing": routing, "qkey": qkey, "n": len(rows),
                    "keyfacts_acc": k / ng if ng else None,
                    "keyfacts_n_gradable": ng,
                    "keyfacts_ci95": [round(lo, 4), round(hi, 4)],
                    "contains_acc": sum(hits["contains"]) / len(rows),
                    "discrim_acc": sum(hits["discrim"]) / len(rows),
                    "hits": hits, "gens": gens,
                    "selected_partition": sel,
                    "gold_partition": gold,
                    "router_correct": [int(a == b) for a, b in zip(sel, gold)],
                    "seconds": round(el, 1),
                }
                print("[g=%-3d %-9s %-2s] keyfacts %.3f  contains %.3f  "
                      "router_acc %.3f"
                      % (g, routing, qkey, out["cells"][key]["keyfacts_acc"] or 0,
                         out["cells"][key]["contains_acc"],
                         sum(out["cells"][key]["router_correct"]) / len(rows)),
                      flush=True)

        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out_json, flush=True)


if __name__ == "__main__":
    main()
