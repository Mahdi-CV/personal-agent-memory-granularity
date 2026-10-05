r"""Retrain the per-author SGD adapters at a MATCHED OPTIMIZER-STEP BUDGET.

WHY THE FIRST PER-AUTHOR BUILD WAS NOT A FAIR CONTROL

The trainer uses per_device_batch_size=1 with grad_accum=8
(connector/distill/lora_trainer.py:53-54), so the number of weight updates is
ceil(n_records / 8) * epochs, not epochs:

    200 records, 10 epochs -> ceil(200/8)=25 * 10 = 250 optimizer steps
     20 records, 10 epochs -> ceil(20/8) = 3 * 10 =  30 optimizer steps

Holding "epochs" fixed therefore gave the per-author adapters 8.3x FEWER
weight updates than the 200-fact adapter they are meant to be matched against.
Their weak result (keyfacts 0.155 on the original question, versus 0.315 for
the same modules trained on all 200) is consistent with undertraining, so it
cannot be read as a memory-partition effect.

That error biases the comparison IN D2L'S FAVOUR, which is exactly the
direction the control was built to rule out. So it has to be fixed before any
claim is made about SGD versus compiled memory.

MATCHING RULE

Equalize optimizer steps, not epochs:

    epochs_needed = round(250 / ceil(20/8)) = round(250/3) = 83

At 83 epochs each per-author adapter takes 249 optimizer steps, within one
step of the 250 the 200-fact arm received, over the same fact partition.

CLARIFICATION (2026-08-27, no behavioural change): the two counts above are BOTH
correct and refer to DIFFERENT arms. The 200-fact adapter ran 250 updates
(25 x 10); the per-author adapters ran 249 (3 x 83). The submitted paper reports
the PER-ADAPTER gradient budget, \SgdOptSteps = 249 (tables/macros.tex:57), and
states it is held constant across all four granularities. So 249, not 250, is
the reference any new fixed-step ladder must match. The "within one step"
slack accepted here is fine for a 2-point control but is NOT acceptable as a
4-point ladder, which is why 27_build_sgd_exact249.py trains to exactly 249
updates via max_steps instead of rounding epochs.

Both builds are kept. The 10-epoch set is the epoch-matched arm and the
83-epoch set is the step-matched arm; reporting both shows the SGD result is
not an artifact of one arbitrary budget choice. An SGD adapter given a longer
budget on 20 facts is also the setting most likely to MEMORIZE them, so this
is the strongest form of the SGD arm, which is what the control needs.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict

DATA = "/work/tofu_exp/data_v3/ceiling"
OUTROOT = "/work/d2l_exp/sgd_perauthor_matched"
STATS = "/work/d2l_exp/sgd_perauthor_matched_stats.json"
D2L_STATS = "/work/d2l_exp/build_stats.json"
REF_STATS = "/work/d2l_exp/sgd_dp_stats.json"
BASE = "/models/Qwen3-4B-Instruct-2507"
GRAD_ACCUM = 8


def opt_steps(n, epochs):
    return math.ceil(n / GRAD_ACCUM) * epochs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(DATA, "test.jsonl"))]
    sft = [json.loads(l) for l in open(os.path.join(DATA, "train_sft.jsonl"))]
    assert len(rows) == len(sft) == 200

    by_author = defaultdict(list)
    for i, r in enumerate(rows):
        by_author[r["author_idx"]].append(i)
    authors = sorted(by_author)

    d2l = json.load(open(D2L_STATS))["adapters"]
    for a in authors:
        assert "author%02d" % a in d2l, "D2L has no adapter for author %d" % a
    assert len([k for k in d2l if k.startswith("author")]) == len(authors)

    ref = json.load(open(REF_STATS))
    ref_steps = opt_steps(ref["n_records"], ref["epochs"])
    n_per = len(by_author[authors[0]])
    assert all(len(v) == n_per for v in by_author.values())
    steps_per_epoch = math.ceil(n_per / GRAD_ACCUM)
    epochs = max(1, round(ref_steps / steps_per_epoch))
    got = opt_steps(n_per, epochs)
    print("reference (sgd_dp): %d records x %d epochs = %d optimizer steps"
          % (ref["n_records"], ref["epochs"], ref_steps), flush=True)
    print("per-author: %d records, %d steps/epoch -> %d epochs = %d steps"
          % (n_per, steps_per_epoch, epochs, got), flush=True)
    assert abs(got - ref_steps) <= steps_per_epoch, (got, ref_steps)

    os.makedirs(OUTROOT, exist_ok=True)
    stats = {"base": BASE, "epochs": epochs, "rank": args.rank,
             "target_modules": ["down_proj"], "grad_accum": GRAD_ACCUM,
             "optimizer_steps_per_adapter": got,
             "reference_optimizer_steps": ref_steps,
             "match_criterion": "optimizer steps, not epochs",
             "adapters": {}}
    total_t0 = time.time()

    for a in authors:
        idx = by_author[a]
        name = "author%02d" % a
        shard = os.path.join(OUTROOT, "%s_sft.jsonl" % name)
        with open(shard, "w") as f:
            for i in idx:
                f.write(json.dumps(sft[i]) + "\n")

        out_dir = os.path.join(OUTROOT, name)
        t0 = time.time()
        cmd = [
            sys.executable, "/work/tofu_exp/tofu_train_v3.py",
            "--base", BASE, "--train-jsonl", shard, "--out-dir", out_dir,
            "--epochs", str(epochs), "--rank", str(args.rank),
            "--alpha", str(args.alpha), "--lr", str(args.lr),
            "--target-modules", "down_proj",
            "--adapter-name", "sgdpam%02d" % a,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        el = time.time() - t0
        if r.returncode != 0:
            print(r.stdout[-3000:], flush=True)
            print(r.stderr[-3000:], flush=True)
            raise SystemExit("train failed for %s" % name)

        sf = os.path.join(out_dir, "adapter_model.safetensors")
        stats["adapters"][name] = {
            "path": out_dir, "n_records": len(idx), "item_indices": idx,
            "train_seconds": round(el, 2), "bytes": os.path.getsize(sf),
        }
        print("[%s] n=%d %.1fs" % (name, len(idx), el), flush=True)

    stats["total_train_seconds"] = round(time.time() - total_t0, 1)
    stats["n_adapters"] = len(authors)
    stats["total_bytes"] = sum(v["bytes"] for v in stats["adapters"].values())
    with open(STATS, "w") as f:
        json.dump(stats, f, indent=2)
    print("TOTAL %.1fs for %d adapters" %
          (stats["total_train_seconds"], stats["n_adapters"]), flush=True)


if __name__ == "__main__":
    main()
