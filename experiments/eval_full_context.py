#!/usr/bin/env python3
"""Evaluate the base model with all 200 memory records in every prompt.

This is the no-selection in-context operating point from the accepted paper.
It uses the same prompt, generation settings, and key-fact grader as the other
arms and records both local-tokenizer and server-reported prompt lengths.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tofu_freeform_v3 import FreeFormEvaluator, grade_keyfacts, wilson  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8110")
    ap.add_argument("--model", default="qwen4bi")
    ap.add_argument("--tokenizer", default="/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--data", default="/work/tofu_exp/data_v3/ceiling")
    ap.add_argument("--out", default="/work/d2l_exp/results/full_context.json")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(
        os.path.join(args.data, "test.jsonl"))]
    facts = json.load(open(os.path.join(args.data, "facts_small.json")))
    assert len(rows) == len(facts) == 200

    endpoint = args.url.rstrip("/") + "/v1/completions"
    evaluator = FreeFormEvaluator(
        endpoint, args.tokenizer, facts, k=5, max_tokens=args.max_tokens)
    tokenizer = evaluator.tok
    context = "\n".join("- " + fact for fact in facts)

    def prompt(question: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the question factually "
                    "and concisely. Reply with the answer only, in one short "
                    "sentence.\n\nUse the following facts to answer:\n" + context
                ),
            },
            {"role": "user", "content": question},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)

    def generate(question: str) -> tuple[str, int, int]:
        rendered = prompt(question)
        local_tokens = len(tokenizer(rendered).input_ids)
        response = requests.post(endpoint, json={
            "model": args.model,
            "prompt": rendered,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "stop": ["\n\n"],
        }, timeout=300)
        response.raise_for_status()
        payload = response.json()
        text = payload["choices"][0]["text"].strip()
        server_tokens = payload.get("usage", {}).get("prompt_tokens", -1)
        return text, local_tokens, server_tokens

    output = {
        "experiment": "full_context",
        "base_model": "Qwen3-4B-Instruct-2507",
        "context_construction": {
            "n_facts": len(facts),
            "selection": "none; all facts in fixed dataset order",
        },
        "generation": {
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "workers": args.workers,
        },
        "cells": {},
    }

    for qkey in ("q", "pq"):
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            generated = list(pool.map(
                lambda row: generate(row[qkey]), rows))

        hits = []
        local_counts = []
        server_counts = []
        generations = []
        for row, (text, local_n, server_n) in zip(rows, generated):
            hit, _, gradable = grade_keyfacts(
                text, row["true"], row["para_true"], row["perturbed"])
            hits.append(hit if gradable else -1)
            local_counts.append(local_n)
            server_counts.append(server_n)
            generations.append(text)

        gradable_hits = [hit for hit in hits if hit >= 0]
        correct = sum(gradable_hits)
        lo, hi = wilson(correct, len(gradable_hits))
        key = "full_context/%s" % qkey
        output["cells"][key] = {
            "keyfacts_acc": correct / len(gradable_hits),
            "keyfacts_n_hit": correct,
            "keyfacts_n_gradable": len(gradable_hits),
            "keyfacts_ci95": [round(lo, 4), round(hi, 4)],
            "hits": {"keyfacts": hits},
            "gens": generations,
            "mean_prompt_tokens": sum(local_counts) / len(local_counts),
            "max_prompt_tokens": max(local_counts),
            "server_reported_prompt_tokens_mean": (
                sum(server_counts) / len(server_counts)
                if min(server_counts) >= 0 else None
            ),
            "seconds": round(time.time() - t0, 1),
        }
        print(key, json.dumps({
            "accuracy": output["cells"][key]["keyfacts_acc"],
            "n_gradable": len(gradable_hits),
            "mean_prompt_tokens": output["cells"][key]["mean_prompt_tokens"],
        }))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(output, handle, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
