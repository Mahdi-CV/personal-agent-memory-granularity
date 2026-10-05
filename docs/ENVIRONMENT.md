# Environment

## Hardware used for the published runs

- AMD Instinct MI300X
- 192 GB HBM3
- ROCm 7.2.3
- One GPU per active model server

The experiment is not ROCm-specific. A different accelerator can be used if it
can serve Qwen3-4B-Instruct-2507 in bf16 with rank-16 LoRA adapters.

## Container

```text
rocm/hyperloom:vllm-v0.24.0-rocm7.2.0
```

The image tag says ROCm 7.2.0, while the installed runtime used by the reported
runs was ROCm 7.2.3.

## Main package versions

| Package | Version |
|---|---|
| vLLM | 0.24.0+rocm723 |
| PyTorch | 2.11.0+gitd0c8b1f |
| Transformers | 5.12.1 |
| PEFT | 0.19.1 |
| Datasets | 5.0.0 |
| Accelerate | 1.14.0 |
| Safetensors | 0.8.0 |

PyTorch is provided by the container and is intentionally not included in the
requirements files.
