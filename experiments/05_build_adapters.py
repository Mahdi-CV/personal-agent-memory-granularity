"""Compile D2L adapters from the SAME 200 TOFU facts the other arms receive.

Two units of personalization, both reported (user decision):

  perauthor : 10 adapters, one per trained author, each internalizing that
              author's ~20 facts (600-1050 tokens). Closest to D2L's released
              design and to its 512-token meta-training regime.
  all200    : 1 adapter internalizing all 200 facts as one 7303-token document.
              Matches the trained LoRA's memory load exactly, but runs ~14x
              beyond D2L's training length.

Reporting both turns the length confound into a measured axis: if perauthor
works and all200 does not, the limit is document length, not compiled memory.

The factual source is `facts_small.json`, the identical 200 answer strings the
retrieval arm indexes and that the SFT set is built from. No arm sees a fact the
others do not.

Timing is recorded per adapter because per-user construction cost is one of the
systems tradeoffs, and it is the axis where generation should beat SGD.
"""
import json
import os
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d2l_to_peft import convert, write_adapter  # noqa: E402

import ctx_to_lora.modeling.hypernet as hn  # noqa: E402
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel  # noqa: E402

CKPT = "/work/d2l_ckpt/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"
DATA = "/work/tofu_exp/data_v3/ceiling"
OUTROOT = "/work/d2l_exp/adapters"
STATS = "/work/d2l_exp/build_stats.json"

# Capture what apply_lora_to_layers actually receives. internalize() stores RAW
# per-chunk tensors, but generate() re-runs combine_lora to append the head-bias
# rank block (hypernet.py:874-879, lora_merger.py:70-73), so rank is
# compute_rank(1,8)=16 and not 8. Converting the raw tensors would silently
# produce a half-capacity adapter. Verified in 02_equiv.py / 03_resid.py.
CAPTURED = {}
_orig_apply = hn.apply_lora_to_layers


def capturing_apply(model, layer_indices, generated_loras, n_qs, position_ids=None):
    CAPTURED["loras"] = {
        m: {k: v.detach().cpu().clone() for k, v in d.items()}
        for m, d in generated_loras.items()
    }
    return _orig_apply(model, layer_indices, generated_loras, n_qs, position_ids)


hn.apply_lora_to_layers = capturing_apply


def main():
    rows = [json.loads(l) for l in open(os.path.join(DATA, "test.jsonl"))]
    facts = json.load(open(os.path.join(DATA, "facts_small.json")))
    assert len(facts) == 200, len(facts)

    by_author = defaultdict(list)
    for i, r in enumerate(rows):
        by_author[r["author_idx"]].append(i)
    authors = sorted(by_author)
    print("authors: %s (items each: %s)"
          % (authors, [len(by_author[a]) for a in authors]), flush=True)

    # Per-author document = that author's own facts, in dataset order. Index
    # alignment matters: facts[i] is the answer to rows[i], so slicing by row
    # index keeps each author's document to exactly their own facts.
    docs = {}
    for a in authors:
        docs["author%02d" % a] = " ".join(facts[i] for i in by_author[a])
    docs["all200"] = " ".join(facts)

    sd_ck = torch.load(CKPT, weights_only=False, map_location="cpu")
    model = ModulatedPretrainedModel.from_state_dict(
        sd_ck, train=False, use_sequence_packing=False
    )
    model.apply_lora_to_layers = capturing_apply
    base_name = model.base_model.name_or_path
    d2l_scaling = float(model.peft_config.lora_alpha)
    print("base=%s scaling=%.6f" % (base_name, d2l_scaling), flush=True)

    from ctx_to_lora.model_loading import get_tokenizer
    tok = get_tokenizer(base_name)

    stats = {"base": base_name, "d2l_scaling": d2l_scaling, "adapters": {}}
    for name, doc in docs.items():
        ntok = len(tok(doc).input_ids)
        CAPTURED.clear()
        model.reset()

        torch.cuda.synchronize()
        t0 = time.time()
        model.internalize(doc)
        # internalize() alone does not call apply_lora_to_layers; the bias-merged
        # tensors only exist once the model is actually used. One forward through
        # generate() is the cheapest way to reach the real application site, and
        # it is part of honest construction cost, so it stays inside the timer.
        ids = tok("hello", return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                           max_new_tokens=1, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        assert "loras" in CAPTURED, "no capture for %s" % name
        sd, rank = convert(CAPTURED["loras"], d2l_scaling)
        out = os.path.join(OUTROOT, name)
        cfg = write_adapter(out, sd, rank, d2l_scaling, base_name,
                            list(CAPTURED["loras"].keys()))
        nbytes = os.path.getsize(os.path.join(out, "adapter_model.safetensors"))
        stats["adapters"][name] = {
            "path": out, "doc_tokens": ntok, "rank": rank,
            "lora_alpha": cfg["lora_alpha"], "n_tensors": len(sd),
            "build_seconds": round(elapsed, 3), "bytes": nbytes,
        }
        print("[%-9s] tokens=%-5d rank=%-3d tensors=%-3d %.3fs %.1fMB"
              % (name, ntok, rank, len(sd), elapsed, nbytes / 1e6), flush=True)

    os.makedirs(os.path.dirname(STATS), exist_ok=True)
    with open(STATS, "w") as f:
        json.dump(stats, f, indent=2)
    print("wrote", STATS, flush=True)


if __name__ == "__main__":
    main()
