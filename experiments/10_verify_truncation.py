"""Did the 7303-token all-200 document actually reach the hypernetwork intact?

If any stage silently truncated, the all200 collapse would be an input-length
artifact and NOT a memory-scale result, and the paper would make a false claim.

Four independent checks, from the raw string down to the tensor the perceiver
consumed:

  1. Tokenize the document ourselves and count.
  2. Run D2L's own tokenize_ctx_text and compare counts. This is the exact
     function internalize() calls (hypernet.py:785-787), and it is configured
     truncation=False, but we verify rather than trust the flag.
  3. Instrument the encoder forward to record the ctx_ids tensor shape actually
     received, so any slicing between tokenization and consumption is visible.
  4. Content probe: check that facts from the START, MIDDLE and END of the
     document all survive into the token stream. A tail truncation would drop
     the end; a head truncation would drop the start.

Check 4 matters most for the claim: it is content-addressed, so it catches
truncation wherever it happens, including inside a stage we did not instrument.
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ctx_to_lora.modeling.hypernet as hn  # noqa: E402
from ctx_to_lora.data.processing import tokenize_ctx_text  # noqa: E402
from ctx_to_lora.model_loading import get_tokenizer  # noqa: E402
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel  # noqa: E402

CKPT = "/work/d2l_ckpt/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"
DATA = "/work/tofu_exp/data_v3/ceiling"
OUT = "/work/d2l_exp/truncation_check.json"

facts = json.load(open(os.path.join(DATA, "facts_small.json")))
doc = " ".join(facts)
res = {"n_facts": len(facts), "doc_chars": len(doc)}

sd_ck = torch.load(CKPT, weights_only=False, map_location="cpu")
model = ModulatedPretrainedModel.from_state_dict(
    sd_ck, train=False, use_sequence_packing=False
)
model.reset()
base_name = model.base_model.name_or_path
ctx_enc_name = model.ctx_encoder.base_model.name_or_path
res["base_model"] = base_name
res["ctx_encoder"] = ctx_enc_name
print("base=%s\nctx_encoder=%s" % (base_name, ctx_enc_name), flush=True)

# ---- 1 + 2: our count vs D2L's own tokenizer path ----
tok = get_tokenizer(ctx_enc_name)
plain = len(tok(doc).input_ids)
d2l_ids = tokenize_ctx_text(dict(context=[doc]), tok)["ctx_ids"]
d2l_n = len(d2l_ids[0])
res["plain_tokens"] = plain
res["d2l_tokenize_ctx_text_tokens"] = d2l_n
print("plain tokenize: %d | D2L tokenize_ctx_text: %d (chat template adds %d)"
      % (plain, d2l_n, d2l_n - plain), flush=True)

# ---- 3: what the encoder actually received ----
SEEN = {}
_orig_gw = ModulatedPretrainedModel.generate_weights


def traced_gw(self, ctx_ids, ctx_attn_mask=None, ctx_position_ids=None, *a, **k):
    SEEN["ctx_ids_shape"] = tuple(ctx_ids.shape)
    SEEN["ctx_ids_numel"] = int(ctx_ids.numel())
    if ctx_attn_mask is not None:
        SEEN["attn_sum"] = int(ctx_attn_mask.sum())
    return _orig_gw(self, ctx_ids, ctx_attn_mask, ctx_position_ids, *a, **k)


ModulatedPretrainedModel.generate_weights = traced_gw
model.internalize(doc)
ModulatedPretrainedModel.generate_weights = _orig_gw

res["encoder_received_shape"] = SEEN.get("ctx_ids_shape")
res["encoder_received_numel"] = SEEN.get("ctx_ids_numel")
res["encoder_attn_sum"] = SEEN.get("attn_sum")
print("encoder received ctx_ids shape=%s numel=%s attn_sum=%s"
      % (SEEN.get("ctx_ids_shape"), SEEN.get("ctx_ids_numel"),
         SEEN.get("attn_sum")), flush=True)

consumed = SEEN.get("ctx_ids_numel") == d2l_n
res["all_tokens_consumed"] = bool(consumed)

# ---- 4: content probe, start / middle / end ----
ids = d2l_ids[0]
decoded = tok.decode(ids)
probes = {
    "first_fact": facts[0],
    "middle_fact": facts[len(facts) // 2],
    "last_fact": facts[-1],
}
res["content_probe"] = {}
for k, v in probes.items():
    # Compare on a distinctive slice; whitespace/casing can differ after decode.
    needle = " ".join(v.split()[:6])
    present = needle.lower() in decoded.lower()
    res["content_probe"][k] = {"needle": needle, "present": bool(present)}
    print("  %-12s present=%-5s | %s" % (k, present, needle), flush=True)

all_present = all(v["present"] for v in res["content_probe"].values())
res["all_content_present"] = bool(all_present)

verdict = consumed and all_present
res["verdict"] = "NO_TRUNCATION" if verdict else "TRUNCATION_SUSPECTED"
print("=" * 70, flush=True)
print("tokens consumed by encoder: %s (%s of %s)"
      % (consumed, SEEN.get("ctx_ids_numel"), d2l_n), flush=True)
print("start+middle+end all present: %s" % all_present, flush=True)
print("VERDICT:", res["verdict"], flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print("wrote", OUT, flush=True)
sys.exit(0 if verdict else 2)
