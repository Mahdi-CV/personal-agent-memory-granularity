# Reproduction

## No-GPU reproduction

The committed per-item JSON files are sufficient to regenerate the statistical
analysis, every paper table, and the final PDF:

```bash
python3 paper/camera_ready_analysis.py
python3 paper/make_tables.py
python3 paper/make_figs.py
./paper/build.sh
```

This is the supported quick reproduction path.

## Full model rerun

### 1. Start the container

```bash
docker run -d --name memory-granularity \
  -v "$PWD":/work \
  -v /path/to/models:/models \
  --device /dev/kfd --device /dev/dri --group-add video \
  --ipc=host --shm-size 16g --security-opt seccomp=unconfined \
  rocm/hyperloom:vllm-v0.24.0-rocm7.2.0 sleep infinity

docker exec -it memory-granularity bash
cd /work
pip install -r requirements-train.txt
pip install -r requirements-granularity.txt
```

### 2. Install the external compiler

Clone the released Doc-to-LoRA implementation to `/work/d2l` and download its
Qwen 4B checkpoint so this file exists:

```text
/work/d2l_ckpt/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin
```

The expected base model is:

```text
/models/Qwen3-4B-Instruct-2507
```

### 3. Create the executed path layout

The released scripts are preserved close to the versions used for the paper:

```bash
ln -s /work/experiments /work/tofu_exp
ln -s /work/experiments /work/d2l_test
mkdir -p /work/d2l_exp
```

### 4. Build the data

```bash
python experiments/tofu_build_v3.py --out-dir /work/tofu_exp/data_v3
```

The primary condition contains 200 original question-answer chat records from
authors 0 through 9. Paraphrases are evaluation-only.

### 5. Validate and build compiled adapters

```bash
python experiments/05_build_adapters.py
python experiments/17_build_granularity.py
python experiments/10_verify_truncation.py
```

### 6. Build gradient adapters

Fixed 249 updates:

```bash
python experiments/29_build_sgd_seeded249.py
```

Equal 83-epoch exposure:

```bash
python experiments/21_build_sgd_granularity.py
```

### 7. Serve Qwen3-4B

```bash
HIP_VISIBLE_DEVICES=0 \
VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
vllm serve /models/Qwen3-4B-Instruct-2507 \
  --served-model-name qwen4bi \
  --max-model-len 4096 \
  --enforce-eager \
  --gpu-memory-utilization 0.85 \
  --enable-lora --max-loras 4 --max-cpu-loras 96 \
  --max-lora-rank 16 --port 8107
```

### 8. Gate, evaluate, and analyze

```bash
python experiments/14_load_and_gate.py

python experiments/25_gate_sgd_sweep.py
python experiments/24_eval_sgd_sweep.py
python experiments/23_analyze_sgd_granularity.py

python experiments/18_eval_granularity.py
python experiments/19_eval_router_sgd.py
python experiments/20_analyze_granularity.py

python experiments/07_eval_threeway.py
python experiments/09_analyze.py
python experiments/eval_full_context.py
```

Every adapter must pass the log-probability activity gate before its evaluation
is accepted.

## External assets

Qwen model weights, the TOFU dataset, and the Doc-to-LoRA source and checkpoint
are obtained from their original projects and are not copied into this
repository.
