r"""Build the gradient GRANULARITY x TRAINING-EXPOSURE sweep.

Falsification control for the compiled (doc-to-LoRA) granularity curve, and a
measurement of how strongly each granularity depends on training exposure.

Two questions, per PREREGISTER-gradient-ladder.md:
  1. Does the compiled granularity curve reproduce under WELL-TRAINED SGD, or
     is it specific to the hypernetwork compiler?
  2. How exposure-sensitive is each granularity? Are coarse adapters
     undertrained at a fixed step budget? Do fine adapters overfit?

WHY A SWEEP AND NOT ONE FIXED STEP BUDGET

Steps and per-fact exposure cannot both be held fixed. lora_trainer.py:53-54
uses per_device_batch_size=1 with grad_accum=8, so updates = ceil(n/8) * epochs.

  Fixing STEPS at ~249 -> equal construction compute, but per-fact exposure of
    83 / 50 / 19 / 10 epochs at g = 20 / 40 / 100 / 200. Coarse may be
    undertrained.
  Fixing EPOCHS -> equal per-fact exposure, but 8.3x more compute at g=20.
    That is the error caught on 2026-08-25.

Neither single choice is a capacity test. The project's own development history
(SGD LoRA was abandoned for Doc-to-LoRA partly due to exposure/overfitting
sensitivity) is exactly why one arbitrary step count would be invalid here.
That history is motivation, NOT evidence, and is not claimed as a result.

So we sweep exposure at every granularity and report the FULL learning curves.

GRID (fixed in advance, never tuned against the evaluation set)

  g=20   10 adapters   epochs 10, 20, 40, 83     (83 -> 249 steps: EXACT)
  g=40    5 adapters   epochs 10, 20, 40, 50, 83 (50 -> 250 steps: NOT exact)
  g=100   2 adapters   epochs 10, 19, 20, 40, 83 (19 -> 247 steps: NOT exact)
  g=200   1 adapter    epochs 10, 20, 40, 83     (10 -> 250 steps: NOT exact)

CORRECTION (2026-08-27): the paper's frozen gradient budget is 249 optimizer
steps (tables/macros.tex:57, \SgdOptSteps = 249; \SgdEpochs = 83). This file
originally used REF_STEPS = 250 and derived epochs by rounding, so only the
g=20 cell lands on 249 exactly. The other three are off by 1-3 updates and MUST
NOT be labeled an exact step-matched ladder. They remain valid exposure points.
The exact ladder is built separately by 27_build_sgd_exact249.py using the
trainer's optimizer-step stopping mechanism (max_steps), not epoch conversion.
The "is_step_matched" flag written by this file therefore means "nearest
epoch-derived approximation to the reference budget", NOT "exactly 249".

83 epochs at g=200 is the exposure-matched diagnostic: 2075 steps, 8.3x the
fixed-step budget, the most favourable budget the gradient arm can get short of
unbounded training.

Held constant everywhere: base, rank 16, alpha 32, LR 2e-4, down_proj only,
grad_accum 8, and the identical author-aligned partitions the compiled ladder
used (16_router.build_partitions). Only granularity and exposure vary.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/work/tofu_exp")

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "router16", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "16_router.py"))
router16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(router16)

DATA = "/work/tofu_exp/data_v3/ceiling"
OUTROOT = "/work/d2l_exp/sgd_gran"
STATS = "/work/d2l_exp/sgd_gran_build_stats.json"
BASE = "/models/Qwen3-4B-Instruct-2507"
GRAD_ACCUM = 8
# The paper's frozen gradient budget: tables/macros.tex:57 \SgdOptSteps = 249.
# (An earlier value of 250 here was wrong. Verified the change is inert: it
# yields an IDENTICAL epoch grid at every granularity -- 83/50/19/10 either way
# -- so no built adapter is affected. Only the label is corrected.)
REF_STEPS = 249

# Predeclared exposure levels, common across granularities. Each granularity
# additionally gets its own step-matched epoch count (see step_matched_epochs).
EXPOSURE_EPOCHS = [10, 20, 40, 83]


def opt_steps(n, epochs):
    return math.ceil(n / GRAD_ACCUM) * epochs


def step_matched_epochs(n_per):
    """Epochs giving ~REF_STEPS optimizer steps for a partition of n_per facts."""
    return max(1, round(REF_STEPS / math.ceil(n_per / GRAD_ACCUM)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--granularities", default="")
    ap.add_argument("--epochs-grid", default="",
                    help="comma list; default is the predeclared grid")
    ap.add_argument("--out-root", default=OUTROOT)
    ap.add_argument("--stats", default=STATS)
    ap.add_argument("--tag", default="sgdgr")
    ap.add_argument("--resume", action="store_true",
                    help="skip adapters whose safetensors already exist")
    args = ap.parse_args()

    grans = ([int(x) for x in args.granularities.split(",")]
             if args.granularities else list(router16.GRANULARITIES))
    grid = ([int(x) for x in args.epochs_grid.split(",")]
            if args.epochs_grid else list(EXPOSURE_EPOCHS))

    rows = [json.loads(l) for l in open(os.path.join(DATA, "test.jsonl"))]
    sft = [json.loads(l) for l in open(os.path.join(DATA, "train_sft.jsonl"))]
    facts = json.load(open(os.path.join(DATA, "facts_small.json")))
    assert len(rows) == len(sft) == len(facts) == 200

    os.makedirs(args.out_root, exist_ok=True)
    if args.resume and os.path.exists(args.stats):
        stats = json.load(open(args.stats))
    else:
        stats = {
            "base": BASE, "rank": args.rank, "alpha": args.alpha, "lr": args.lr,
            "target_modules": ["down_proj"], "grad_accum": GRAD_ACCUM,
            "reference_optimizer_steps": REF_STEPS,
            "design": "granularity x training-exposure sweep",
            "exposure_grid_epochs": grid,
            "preregistered": "paper2/PREREGISTER-gradient-ladder.md",
            "note": ("steps and per-fact exposure cannot both be held fixed; "
                     "we sweep exposure and report full learning curves"),
            "conditions": {},
        }
    stats.setdefault("conditions", {})

    total_t0 = time.time()
    for g in grans:
        parts = router16.build_partitions(rows, facts, g)
        n_per = len(parts[0]["fact_indices"])
        assert all(len(p["fact_indices"]) == n_per for p in parts)
        spe = math.ceil(n_per / GRAD_ACCUM)
        sm = step_matched_epochs(n_per)
        levels = sorted(set(grid + [sm]))

        for epochs in levels:
            steps = opt_steps(n_per, epochs)
            ckey = "g%d/e%d" % (g, epochs)
            cond = {
                "facts_per_adapter": g, "n_adapters": len(parts),
                "epochs": epochs, "steps_per_epoch": spe,
                "optimizer_steps_per_adapter": steps,
                "per_fact_exposure_epochs": epochs,
                "examples_seen_per_adapter": n_per * epochs,
                # NOT "exactly the paper's 249 updates" -- only the g=20 cell is.
                # This flag means "nearest epoch-derived approximation to the
                # reference budget". The exact ladder lives in the separate
                # sgd_exact249 artifact set. is_exact_249 disambiguates.
                "is_nearest_epoch_to_ref": epochs == sm,
                "is_step_matched": epochs == sm,   # legacy key, same meaning
                "is_exact_249": steps == REF_STEPS,
                "steps_minus_ref": steps - REF_STEPS,
                "adapters": {},
            }
            print("\n=== %s  n_ad=%d n_per=%d epochs=%d steps=%d%s ==="
                  % (ckey, len(parts), n_per, epochs, steps,
                     ("  [EXACT 249]" if steps == REF_STEPS
                      else "  [nearest-epoch to ref, %+d steps]" % (steps - REF_STEPS)
                      if epochs == sm else "")), flush=True)

            t_cond = time.time()
            for p in parts:
                name = "%s_g%d_e%d_%s" % (args.tag, g, epochs, p["name"])
                out_dir = os.path.join(args.out_root, name)
                sf = os.path.join(out_dir, "adapter_model.safetensors")
                if args.resume and os.path.exists(sf):
                    prev = (stats["conditions"].get(ckey, {})
                            .get("adapters", {}).get(p["name"]))
                    if prev:
                        cond["adapters"][p["name"]] = prev
                        print("  [%s] skip (exists)" % name, flush=True)
                        continue

                shard = os.path.join(args.out_root, "%s_sft.jsonl" % name)
                with open(shard, "w") as f:
                    for i in p["fact_indices"]:
                        f.write(json.dumps(sft[i]) + "\n")

                t0 = time.time()
                cmd = [
                    sys.executable, "/work/tofu_exp/tofu_train_v3.py",
                    "--base", BASE, "--train-jsonl", shard, "--out-dir", out_dir,
                    "--epochs", str(epochs), "--rank", str(args.rank),
                    "--alpha", str(args.alpha), "--lr", str(args.lr),
                    "--target-modules", "down_proj",
                    "--adapter-name", name,
                ]
                r = subprocess.run(cmd, capture_output=True, text=True)
                el = time.time() - t0
                if r.returncode != 0:
                    print(r.stdout[-3000:], flush=True)
                    print(r.stderr[-3000:], flush=True)
                    raise SystemExit("train failed for %s" % name)

                cond["adapters"][p["name"]] = {
                    "path": out_dir, "lora_name": name,
                    "n_records": len(p["fact_indices"]),
                    "author_ids": p["author_ids"],
                    "fact_indices": p["fact_indices"],
                    "train_seconds": round(el, 2),
                    "bytes": os.path.getsize(sf),
                }
                print("  [%s] n=%d %.1fs %.1fMB"
                      % (name, len(p["fact_indices"]), el,
                         os.path.getsize(sf) / 1e6), flush=True)

            cond["total_build_seconds"] = round(time.time() - t_cond, 1)
            cond["total_bytes"] = sum(v["bytes"] for v in cond["adapters"].values())
            cond["per_active_adapter_bytes"] = max(
                v["bytes"] for v in cond["adapters"].values())
            stats["conditions"][ckey] = cond
            print("  => %s: %d adapters, %.1fs, %.1fMB total"
                  % (ckey, len(parts), cond["total_build_seconds"],
                     cond["total_bytes"] / 1e6), flush=True)

            with open(args.stats, "w") as f:
                json.dump(stats, f, indent=2)

    stats["sweep_total_seconds"] = round(time.time() - total_t0, 1)
    stats["n_conditions"] = len(stats["conditions"])
    with open(args.stats, "w") as f:
        json.dump(stats, f, indent=2)
    print("\nwrote %s (%d conditions, %.1fs)"
          % (args.stats, stats["n_conditions"], stats["sweep_total_seconds"]),
          flush=True)


if __name__ == "__main__":
    main()
