"""Deterministic D2L -> PEFT adapter conversion, derived from D2L's own forward pass.

WHY THIS IS NOT A GUESS

Every constant here is read off D2L source, not assumed:

1. ORIENTATION. `lora_layer.py:34-35` is authoritative:
       delta = einsum(A, x, "tot_q r d_in, tot_q s d_in -> tot_q s r")
       delta = einsum(B, delta, "tot_q r d_out, tot_q s r -> tot_q s d_out")
   so D2L stores A as [r, d_in] and B as [r, d_out]. PEFT stores lora_A as
   [r, d_in] (same) and lora_B as [d_out, r] (TRANSPOSED). Confirmed empirically
   by the observed shapes: A [.., 8, 9728] with d_in=intermediate_size=9728, and
   B [.., 8, 2560] with d_out=hidden_size=2560, for down_proj.

2. SCALING. `hypernet.py:520` passes `scaling=self.peft_config.lora_alpha`
   DIRECTLY as the multiplier, and `model_loading.py:178` sets
   `lora_alpha = r**(3/2) * 2`. For r=8 that is 45.2548. PEFT instead computes
   scaling = lora_alpha / r. So to make PEFT reproduce D2L we must write
       lora_alpha_for_peft = d2l_scaling * r = 45.2548 * 8 = 362.0387
   Writing 45.2548 into adapter_config.json would silently apply 1/8th of the
   intended delta. This is the single most dangerous field in the conversion.

3. MODULE PATHS. `lora_layer.py:96-99` maps bare module names to
   `self_attn.{name}` or `mlp.{name}`. For down_proj that is `mlp.down_proj`.

4. LAYER INDEXING. `lora_layer.py:101-102` slices `[:, layer_idx]`, so dim 1 of
   the generated tensor is the layer axis and dim 0 is the context/batch axis.
"""
import argparse
import json
import math
import os

import torch
from safetensors.torch import save_file

# From lora_layer.py:96-99.
ATTN_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"}
MLP_MODULES = {"down_proj", "up_proj", "gate_proj"}


def module_subpath(mname):
    if mname in ATTN_MODULES:
        return "self_attn.%s" % mname
    if mname in MLP_MODULES:
        return "mlp.%s" % mname
    raise ValueError("unknown module name %r" % mname)


def convert(generated_loras, d2l_scaling, ctx_index=0):
    """D2L generated_loras -> (peft_state_dict, rank).

    generated_loras[m]["A"]: [n_ctx, n_layers, r, d_in]
    generated_loras[m]["B"]: [n_ctx, n_layers, r, d_out]
    """
    sd = {}
    rank = None
    for mname, mats in generated_loras.items():
        A_all = mats["A"].detach()
        B_all = mats["B"].detach()
        assert A_all.dim() == 4 and B_all.dim() == 4, (A_all.shape, B_all.shape)
        n_ctx, n_layers, r_a, d_in = A_all.shape
        n_ctx_b, n_layers_b, r_b, d_out = B_all.shape
        assert (n_ctx, n_layers, r_a) == (n_ctx_b, n_layers_b, r_b)
        if rank is None:
            rank = r_a
        assert rank == r_a, "mixed ranks across modules not supported"

        sub = module_subpath(mname)
        for li in range(n_layers):
            A = A_all[ctx_index, li]  # [r, d_in]  -> PEFT lora_A is [r, d_in]
            B = B_all[ctx_index, li]  # [r, d_out] -> PEFT lora_B is [d_out, r]
            base = "base_model.model.model.layers.%d.%s" % (li, sub)
            sd[base + ".lora_A.weight"] = A.contiguous().to(torch.float32)
            sd[base + ".lora_B.weight"] = B.t().contiguous().to(torch.float32)
    return sd, rank


def peft_alpha_for(d2l_scaling, rank):
    """PEFT applies scaling = lora_alpha / r; D2L applies `scaling` flat."""
    return float(d2l_scaling) * float(rank)


def write_adapter(out_dir, sd, rank, d2l_scaling, base_model, target_modules):
    os.makedirs(out_dir, exist_ok=True)
    save_file(sd, os.path.join(out_dir, "adapter_model.safetensors"))
    cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model,
        "r": int(rank),
        "lora_alpha": peft_alpha_for(d2l_scaling, rank),
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "target_modules": sorted(target_modules),
        "modules_to_save": None,
        "use_rslora": False,
        "use_dora": False,
    }
    with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-pt", required=True, help="torch.save of generated_loras")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scaling", type=float, required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--ctx-index", type=int, default=0)
    a = ap.parse_args()

    gl = torch.load(a.in_pt, map_location="cpu", weights_only=False)
    sd, rank = convert(gl, a.scaling, a.ctx_index)
    cfg = write_adapter(
        a.out_dir, sd, rank, a.scaling, a.base_model, list(gl.keys())
    )
    print(json.dumps(cfg, indent=2))
    print("tensors:", len(sd), "rank:", rank)
    print("d2l scaling %.6f -> peft lora_alpha %.6f (alpha/r = %.6f)" % (
        a.scaling, cfg["lora_alpha"], cfg["lora_alpha"] / rank))


if __name__ == "__main__":
    main()
