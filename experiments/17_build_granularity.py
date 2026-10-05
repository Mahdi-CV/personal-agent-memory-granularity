"""Build the D2L memory-granularity ladder: 20 / 40 / 100 / 200 facts per adapter.

The total user memory is FIXED at the same 200 facts. Only the number of
partitions it is split across changes:

    20 facts/adapter -> 10 adapters
    40 facts/adapter ->  5 adapters
   100 facts/adapter ->  2 adapters
   200 facts/adapter ->  1 adapter

40 stands in for the requested 50: TOFU author blocks are 20 facts, and a
50-fact partition would split an author across two adapters, breaking both the
coherent-document property D2L consumes and the routing problem's structure.
See 16_router.build_partitions.

CAPACITY IS CONSTANT ALONG THIS LADDER, WHICH IS WHY THE CURVE IS INTERPRETABLE

internalize() sends the whole document as ONE sequence (hypernet.py:796 asserts
ctx_ids.shape[0] == 1), so it never chunks, and combine_lora's rank is
compute_rank(n_lora=1, rank=8) = 16 for every document length. Confirmed
empirically on the existing build: 596-token and 7303-token documents both
produced rank 16, 72 tensors, 28320992 bytes. So `max_ctx_chunk_len: 512` is a
meta-training parameter, not an inference-path limit here.

Consequence: adapter capacity does NOT grow with facts-per-adapter. Every point
on this curve has the same rank, tensor count and byte size, so a change in
accuracy is attributable to how many facts share a fixed-capacity adapter, not
to a confounded change in adapter size. That is exactly the comparison wanted.

TRUNCATION IS VERIFIED PER DOCUMENT, not just for all-200. Each document's
token count is recorded, and the tensor the encoder actually received is
captured and compared against the tokenizer output, so a silent truncation at
any granularity would fail the build rather than quietly flatten the curve.
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/work/tofu_exp")
sys.path.insert(0, "/work/d2l_test")  # d2l_to_peft, far_probe live here
sys.path.insert(0, "/work/d2l/src")  # ctx_to_lora is not pip-installed

from d2l_to_peft import convert, write_adapter  # noqa: E402

import ctx_to_lora.modeling.hypernet as hn  # noqa: E402
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel  # noqa: E402

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "router16", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "16_router.py"))
router16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(router16)

CKPT = "/work/d2l_ckpt/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"
DATA = "/work/tofu_exp/data_v3/ceiling"
OUTROOT = "/work/d2l_exp/gran"
STATS = "/work/d2l_exp/gran_build_stats.json"

CAPTURED = {}
_orig_apply = hn.apply_lora_to_layers


def capturing_apply(model, layer_indices, generated_loras, n_qs, position_ids=None):
    CAPTURED["loras"] = {
        m: {k: v.detach().cpu().clone() for k, v in d.items()}
        for m, d in generated_loras.items()
    }
    return _orig_apply(model, layer_indices, generated_loras, n_qs, position_ids)


hn.apply_lora_to_layers = capturing_apply

SEEN = {}
_orig_gw = ModulatedPretrainedModel.generate_weights


def traced_gw(self, ctx_ids, ctx_attn_mask=None, ctx_position_ids=None, *a, **k):
    SEEN["numel"] = int(ctx_ids.numel())
    SEEN["shape"] = tuple(ctx_ids.shape)
    if ctx_attn_mask is not None:
        SEEN["attn_sum"] = int(ctx_attn_mask.sum())
    return _orig_gw(self, ctx_ids, ctx_attn_mask, ctx_position_ids, *a, **k)


ModulatedPretrainedModel.generate_weights = traced_gw


def main():
    rows = [json.loads(l) for l in open(os.path.join(DATA, "test.jsonl"))]
    facts = json.load(open(os.path.join(DATA, "facts_small.json")))
    assert len(facts) == 200

    sd_ck = torch.load(CKPT, weights_only=False, map_location="cpu")
    model = ModulatedPretrainedModel.from_state_dict(
        sd_ck, train=False, use_sequence_packing=False)
    model.apply_lora_to_layers = capturing_apply
    base_name = model.base_model.name_or_path
    d2l_scaling = float(model.peft_config.lora_alpha)

    from ctx_to_lora.data.processing import tokenize_ctx_text
    from ctx_to_lora.model_loading import get_tokenizer
    ctx_tok = get_tokenizer(model.ctx_encoder.base_model.name_or_path)
    tok = get_tokenizer(base_name)

    stats = {"base": base_name, "d2l_scaling": d2l_scaling,
             "note": "40 replaces the requested 50; TOFU author blocks are 20 facts",
             "granularities": {}}

    # Warm the model once so the first timed compile is not inflated by
    # one-time CUDA/kernel setup. The existing build showed 7.53s for the first
    # adapter versus ~0.88s steady state, which would distort a construction
    # cost curve read across granularities.
    model.reset()
    model.internalize(facts[0])
    ids0 = tok("hello", return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        model.generate(input_ids=ids0, attention_mask=torch.ones_like(ids0),
                       max_new_tokens=1, do_sample=False)
    print("warmed", flush=True)

    for g in router16.GRANULARITIES:
        parts = router16.build_partitions(rows, facts, g)
        gstat = {"facts_per_adapter": g, "n_adapters": len(parts),
                 "adapters": {}}
        t_total = 0.0
        for p in parts:
            doc = " ".join(facts[i] for i in p["fact_indices"])
            ntok_plain = len(ctx_tok(doc).input_ids)
            ntok_d2l = len(tokenize_ctx_text(dict(context=[doc]), ctx_tok)["ctx_ids"][0])

            CAPTURED.clear()
            SEEN.clear()
            model.reset()
            torch.cuda.synchronize()
            t0 = time.time()
            model.internalize(doc)
            ids = tok("hello", return_tensors="pt").input_ids.to(model.device)
            with torch.no_grad():
                model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                               max_new_tokens=1, do_sample=False)
            torch.cuda.synchronize()
            el = time.time() - t0
            t_total += el

            # Per-document truncation gate. A silent truncation at coarse
            # granularity would look exactly like "big adapters forget", so
            # this must fail the build rather than be reported as a result.
            consumed = SEEN.get("numel") == ntok_d2l
            if not consumed:
                raise SystemExit(
                    "TRUNCATION at %s: tokenizer %d vs encoder %s"
                    % (p["name"], ntok_d2l, SEEN.get("numel")))

            assert "loras" in CAPTURED
            sd, rank = convert(CAPTURED["loras"], d2l_scaling)
            out = os.path.join(OUTROOT, p["name"])
            cfg = write_adapter(out, sd, rank, d2l_scaling, base_name,
                                list(CAPTURED["loras"].keys()))
            nb = os.path.getsize(os.path.join(out, "adapter_model.safetensors"))
            gstat["adapters"][p["name"]] = {
                "path": out, "author_ids": p["author_ids"],
                "fact_indices": p["fact_indices"], "n_facts": len(p["fact_indices"]),
                "doc_tokens_plain": ntok_plain, "doc_tokens_d2l": ntok_d2l,
                "encoder_numel": SEEN.get("numel"),
                "encoder_attn_sum": SEEN.get("attn_sum"),
                "all_tokens_consumed": True,
                "rank": rank, "lora_alpha": cfg["lora_alpha"],
                "n_tensors": len(sd), "build_seconds": round(el, 3), "bytes": nb,
            }
            print("[%-10s] facts=%-3d tok=%-5d rank=%-3d %.3fs %.1fMB consumed=%s"
                  % (p["name"], len(p["fact_indices"]), ntok_d2l, rank, el,
                     nb / 1e6, consumed), flush=True)

        gstat["total_build_seconds"] = round(t_total, 3)
        gstat["total_bytes"] = sum(v["bytes"] for v in gstat["adapters"].values())
        gstat["per_active_adapter_bytes"] = max(
            v["bytes"] for v in gstat["adapters"].values())
        gstat["ranks"] = sorted({v["rank"] for v in gstat["adapters"].values()})
        stats["granularities"][str(g)] = gstat
        print("  => g=%d: %d adapters, %.2fs total, %.1fMB total, ranks %s"
              % (g, len(parts), t_total, gstat["total_bytes"] / 1e6,
                 gstat["ranks"]), flush=True)

    with open(STATS, "w") as f:
        json.dump(stats, f, indent=2)
    print("wrote", STATS, flush=True)


if __name__ == "__main__":
    main()
