r"""Seed-controlled exact-249 ladder: uniform LoRA initialization protocol.

WHY THIS EXISTS

The first exact-249 ladder (27_build_sgd_exact249.py) does NOT use a uniform
initialization protocol across granularities:

  g20   REUSED from the sweep, whose adapters are built by
        21_build_sgd_granularity.py via subprocess.run -> every adapter is run 1
        of a FRESH interpreter, so get_peft_model draws lora_A from the
        process's initial global torch RNG state.
  g40/g100/g200
        trained sequentially IN ONE PROCESS. transformers Trainer.__init__ calls
        set_seed(args.seed)=42, so from the 2nd adapter onward get_peft_model
        draws from an RNG state left re-seeded to 42 by the PREVIOUS adapter's
        Trainer construction.

peft's reset_lora_parameters initializes lora_A with nn.init.kaiming_uniform_,
which consumes the GLOBAL torch RNG, and get_peft_model runs BEFORE the Trainer
exists. So initialization protocol varies with build mechanism and position.

This is not bookkeeping. The equivalence audit measured it directly: two runs of
the IDENTICAL frozen recipe that differed only in RNG state produced adapters
with cosine 0.0062 and max_abs_diff 3.917e-02, while two runs at MATCHED RNG
state were byte-identical (same sha256, 72/72 tensors). Initialization
materially changes the adapter tensors. In the original ladder, initialization
protocol is therefore CONFOUNDED WITH GRANULARITY, which is the very comparison
the ladder is built to make.

WHAT THIS SCRIPT DOES

Trains ALL 18 adapters from scratch (g20 x10, g40 x5, g100 x2, g200 x1) under
one protocol: reset Python / NumPy / torch CPU / torch CUDA-HIP RNGs to the SAME
seed immediately BEFORE get_peft_model, so every adapter begins from an
identical lora_A draw regardless of granularity or position in the build.

Seed 42 is used because that is the frozen configuration's value: transformers
TrainingArguments.seed defaults to 42 and neither the frozen trainer nor the
exact-249 trainer overrides it, so 42 is what Trainer.__init__ already applies.

Everything else is preserved exactly: rank 16, alpha 32, dropout 0.05,
down_proj only, LR 2e-4, linear scheduler, no warmup, adamw_torch_fused,
batch 1, grad_accum 8, bf16, max_length 2048, identical deterministic partitions
from 16_router.build_partitions, and the VERIFIED max_steps=249 stopping path
(bitwise-reproduces the frozen epoch trainer at matched RNG state; see
equivalence_audit.json).

global_step is read from Trainer.train().global_step and asserted == 249 per
adapter. Never inferred from epochs.

Writes to a SEPARATE tree and a SEPARATE stats file. Overwrites nothing.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/work/tofu_exp")
sys.path.insert(0, os.environ.get("CONNECTOR_ROOT", "/work"))

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "router16", os.path.join(HERE, "16_router.py"))
router16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(router16)

DATA = "/work/tofu_exp/data_v3/ceiling"
OUTROOT = "/work/d2l_exp/sgd_seeded249"
STATS = "/work/d2l_exp/sgd_seeded249_build_stats.json"
BASE = "/models/Qwen3-4B-Instruct-2507"
GRAD_ACCUM = 8
EXACT_STEPS = 249
SEED = 42
GRANULARITIES = [20, 40, 100, 200]


def reset_all_rngs(seed):
    """Reset every RNG that LoRA init could draw from, BEFORE get_peft_model.

    peft reset_lora_parameters -> nn.init.kaiming_uniform_ consumes the global
    torch CPU generator. The others are reset too so the protocol is uniform
    regardless of where a future init draws from.
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():          # ROCm exposes HIP through torch.cuda
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    state = {
        "python_random": random.getstate()[1][0],
        "numpy": int(np.random.get_state()[1][0]),
        "torch_cpu_seed": int(torch.initial_seed()),
        "torch_cuda_available": bool(torch.cuda.is_available()),
    }
    return state


def fingerprint_lora_init(model):
    """A cheap, exact fingerprint of the lora_A draw, to PROVE uniform init."""
    import torch
    vals = []
    for n, p in model.named_parameters():
        if "lora_A" in n:
            vals.append((n, p.detach().float().flatten()[:4].tolist(),
                         float(p.detach().float().abs().sum())))
    vals.sort()
    first = vals[0] if vals else (None, None, None)
    total = sum(v[2] for v in vals)
    return {"n_lora_A": len(vals), "first_tensor": first[0],
            "first_4_values": first[1],
            "sum_abs_all_lora_A": round(total, 6)}


def train_seeded(base, records, out_dir, rank, alpha, lr, target_modules,
                 max_steps, grad_accum, adapter_name, seed):
    """Same as 27's train_exact, plus an explicit full RNG reset before init."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoTokenizer, DataCollatorForLanguageModeling,
                              Trainer, TrainingArguments)

    from transformers import AutoModelForCausalLM

    os.makedirs(out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def render(example):
        return {"text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False)}

    ds = Dataset.from_list([{"messages": r["messages"]} for r in records])
    ds = ds.map(render, remove_columns=ds.column_names)
    ds = ds.map(lambda e: tokenizer(e["text"], truncation=True,
                                    max_length=2048, padding=False),
                remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    model.enable_input_require_grads()

    # THE FIX: uniform initialization protocol. Reset every RNG immediately
    # before get_peft_model, which is where kaiming_uniform_ draws lora_A.
    rng_state = reset_all_rngs(seed)
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.05,
        target_modules=list(target_modules), bias="none", task_type="CAUSAL_LM"))
    init_fp = fingerprint_lora_init(model)

    args = TrainingArguments(
        output_dir=os.path.join(out_dir, "_ckpt"),
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        seed=seed,
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=DataCollatorForLanguageModeling(
                          tokenizer, mlm=False))
    result = trainer.train()
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    return (int(result.global_step), float(result.training_loss),
            rng_state, init_fp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--steps", type=int, default=EXACT_STEPS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-root", default=OUTROOT)
    ap.add_argument("--stats", default=STATS)
    ap.add_argument("--tag", default="sgds")
    ap.add_argument("--granularities", default="")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    grans = ([int(x) for x in args.granularities.split(",")]
             if args.granularities else list(GRANULARITIES))

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
            "exact_optimizer_steps": args.steps,
            "seed": args.seed,
            "design": "seed-controlled exact-249 ladder",
            "label": "seed-controlled exact-249 ladder",
            "init_protocol": (
                "ALL RNGs (python random, numpy, torch CPU, torch CUDA/HIP) reset "
                "to seed %d immediately BEFORE get_peft_model, so every adapter "
                "gets an identical lora_A draw regardless of granularity or build "
                "position. Fixes the confound in sgd_exact249, where g20 came from "
                "a fresh-process subprocess build and g40/g100/g200 inherited an "
                "RNG state re-seeded by the previous adapter's Trainer.__init__."
                % args.seed),
            "stopping_mechanism": (
                "TrainingArguments.max_steps; verified in equivalence_audit.json to "
                "bitwise-reproduce the frozen epoch trainer at matched RNG state"),
            "counter_source": "transformers Trainer.train().global_step",
            "scheduler": ("linear decay 2e-4 -> 0 over a horizon of %d updates, "
                          "no warmup, adamw_torch_fused, weight_decay 0.0"
                          % args.steps),
            "conditions": {},
        }
    stats.setdefault("conditions", {})

    total_t0 = time.time()
    for g in grans:
        parts = router16.build_partitions(rows, facts, g)
        n_per = len(parts[0]["fact_indices"])
        assert all(len(p["fact_indices"]) == n_per for p in parts)
        ckey = "g%d" % g
        cond = stats["conditions"].get(ckey, {
            "facts_per_adapter": g, "n_adapters": len(parts),
            "optimizer_steps_per_adapter": args.steps,
            "epochs_equivalent": round(args.steps / max(
                n_per // GRAD_ACCUM + int(n_per % GRAD_ACCUM > 0), 1), 4),
            "examples_seen_per_adapter": args.steps * GRAD_ACCUM,
            "seed": args.seed,
            "trained_here": True,
            "verified_global_step": [],
            "init_fingerprints": {},
            "adapters": {},
        })
        print("\n=== %s  n_ad=%d n_per=%d  EXACT %d steps  seed=%d  (= %.3f epochs) ==="
              % (ckey, len(parts), n_per, args.steps, args.seed,
                 cond["epochs_equivalent"]), flush=True)

        t_cond = time.time()
        for p in parts:
            name = "%s_g%d_s%d_%s" % (args.tag, g, args.steps, p["name"])
            out_dir = os.path.join(args.out_root, name)
            sf = os.path.join(out_dir, "adapter_model.safetensors")
            if args.resume and os.path.exists(sf) and p["name"] in cond["adapters"]:
                print("  [%s] skip (exists)" % name, flush=True)
                continue

            shard = os.path.join(args.out_root, "%s_sft.jsonl" % name)
            with open(shard, "w") as f:
                for i in p["fact_indices"]:
                    f.write(json.dumps(sft[i]) + "\n")
            records = [json.loads(l) for l in open(shard)]

            t0 = time.time()
            gstep, loss, rng_state, init_fp = train_seeded(
                BASE, records, out_dir, args.rank, args.alpha, args.lr,
                ["down_proj"], args.steps, GRAD_ACCUM, name, args.seed)
            el = time.time() - t0

            if gstep != args.steps:
                raise SystemExit("STEP MISMATCH for %s: global_step=%d, expected %d"
                                 % (name, gstep, args.steps))

            cond["adapters"][p["name"]] = {
                "path": out_dir, "lora_name": name,
                "n_records": len(p["fact_indices"]),
                "author_ids": p["author_ids"],
                "fact_indices": p["fact_indices"],
                "train_seconds": round(el, 2),
                "global_step": gstep,
                "training_loss": loss,
                "bytes": os.path.getsize(sf),
                "seed": args.seed,
            }
            cond["verified_global_step"].append(gstep)
            cond["init_fingerprints"][p["name"]] = init_fp
            cond["rng_state_at_init"] = rng_state
            print("  [%s] n=%d global_step=%d loss=%.4f %.1fs %.1fMB init_sum=%.6f"
                  % (name, len(p["fact_indices"]), gstep, loss, el,
                     os.path.getsize(sf) / 1e6, init_fp["sum_abs_all_lora_A"]),
                  flush=True)

            stats["conditions"][ckey] = cond
            with open(args.stats, "w") as f:
                json.dump(stats, f, indent=2)

        cond["total_build_seconds"] = round(time.time() - t_cond, 1)
        cond["total_bytes"] = sum(v["bytes"] for v in cond["adapters"].values())
        assert all(v == args.steps for v in cond["verified_global_step"]), \
            cond["verified_global_step"]
        stats["conditions"][ckey] = cond
        with open(args.stats, "w") as f:
            json.dump(stats, f, indent=2)

    # Uniform-init proof: every adapter of every granularity must share one
    # lora_A fingerprint, since all are drawn after an identical RNG reset.
    fps = {}
    for ckey, c in stats["conditions"].items():
        for pn, fp in c.get("init_fingerprints", {}).items():
            fps.setdefault(json.dumps(fp["first_4_values"]), []).append(
                "%s/%s" % (ckey, pn))
    stats["init_uniformity"] = {
        "n_distinct_lora_A_first_draws": len(fps),
        "uniform": len(fps) == 1,
        "groups": {k: v for k, v in fps.items()},
        "note": ("all adapters share ONE lora_A draw iff the RNG reset before "
                 "get_peft_model worked; >1 group means the confound persists"),
    }
    stats["n_adapters_total"] = sum(len(c["adapters"])
                                    for c in stats["conditions"].values())
    stats["build_total_seconds"] = round(time.time() - total_t0, 1)
    with open(args.stats, "w") as f:
        json.dump(stats, f, indent=2)
    print("\nwrote %s -- %d adapters, init uniform=%s (%d distinct draws)"
          % (args.stats, stats["n_adapters_total"],
             stats["init_uniformity"]["uniform"],
             stats["init_uniformity"]["n_distinct_lora_A_first_draws"]),
          flush=True)


if __name__ == "__main__":
    main()
