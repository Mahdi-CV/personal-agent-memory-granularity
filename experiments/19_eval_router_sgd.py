"""Run the SGD-per-author arm under PREDICTED routing.

The oracle-routed SGD-per-author numbers already exist (matched_steps.json).
What is missing is the same arm routed by the deployable router, so the routing
tax can be reported for BOTH memory methods rather than only for D2L.

This uses the IDENTICAL router object as the D2L granularity run: same class,
same scorer, same partition documents, same rule. That is the fairness
requirement. The only thing that differs between the two arms is which adapter
family the predicted partition index is mapped onto:

    D2L  partition i -> gr_g020_pNN
    SGD  partition i -> sgd_pamNN   (step-matched per-author adapters)

At g=20 the partitions ARE the author blocks in ascending author_idx order
(16_router.build_partitions packs whole authors in order), so partition index i
corresponds to author i and the two mappings are index-compatible.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/work/tofu_exp")
sys.path.insert(0, "/work/d2l_test")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    ap.add_argument("--out-json", default="/work/d2l_exp/results/router_sgd.json")
    ap.add_argument("--lora-fmt", default="sgd_pam%02d")
    ap.add_argument("--arm", default="sgd_pa_matched")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    comp = args.url + "/v1/completions"
    rows = [json.loads(l) for l in open(os.path.join(args.data, "test.jsonl"))]
    facts = json.load(open(os.path.join(args.data, "facts_small.json")))

    ev = FreeFormEvaluator(comp, args.tokenizer, facts, k=5,
                           max_tokens=args.max_tokens)
    tok = ev.tok

    parts = router16.build_partitions(rows, facts, 20)
    router = router16.PartitionRouter(parts, facts, rule="max")
    gold = [router16.gold_partition(parts, i) for i in range(len(rows))]

    # Partition index -> author adapter index. Assert the identity mapping the
    # docstring claims instead of trusting it: partition i must contain exactly
    # author i. If build_partitions ever changes its packing order, this fails
    # loudly rather than silently routing every item to the wrong adapter.
    for pi, p in enumerate(parts):
        assert p["author_ids"] == [pi], (pi, p["author_ids"])

    def gen_one(model, prompt):
        r = requests.post(comp, json={
            "model": model, "prompt": prompt, "max_tokens": args.max_tokens,
            "temperature": 0.0, "stop": ["\n\n"]}, timeout=300)
        r.raise_for_status()
        return r.json()["choices"][0]["text"].strip()

    out = {"base_model": "Qwen3-4B-Instruct-2507", "arm": args.arm,
           "facts_per_adapter": 20, "n_adapters": len(parts), "cells": {}}

    for qkey in ("q", "pq"):
        pred = [router.route(r[qkey]) for r in rows]
        for routing, sel in (("oracle", gold), ("predicted", pred)):
            names = [args.lora_fmt % s for s in sel]
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
            key = "%s/%s/%s" % (args.arm, routing, qkey)
            out["cells"][key] = {
                "arm": args.arm, "routing": routing, "qkey": qkey,
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
            print("[sgd %-9s %-2s] keyfacts %.3f  contains %.3f  router_acc %.3f"
                  % (routing, qkey, out["cells"][key]["keyfacts_acc"] or 0,
                     out["cells"][key]["contains_acc"],
                     sum(out["cells"][key]["router_correct"]) / len(rows)),
                  flush=True)

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out_json, flush=True)


if __name__ == "__main__":
    main()
