# Personal-agent memory granularity

Minimal reproduction repository for:

> **How Should a Personal Agent Store Its Facts? Granularity, Compute, and
> Robustness in Parametric Memory**  
> SLM-Agents Workshop at NeurIPS 2026

## Result

The study fixes one 200-record personal memory and varies how many records share
each LoRA adapter.

| Construction regime | 20 records/adapter | 200 records/adapter |
|---|---:|---:|
| Compiled, original questions | 0.602 | 0.116 |
| Gradient, 249 updates | 0.746 | 0.354 |
| Gradient, 83 epochs | 0.773 | 0.757 |
| Gradient, 83 epochs, paraphrased | 0.564 | 0.387 |

The apparent granularity penalty on original questions depends on the training
budget. Under equal record exposure, original-question recall is nearly flat,
while paraphrase recall still declines. The direct interaction is 0.160 with an
author-clustered 95% interval of `[0.105, 0.212]`.

Full context reaches the highest recall while all 200 records fit, but uses
131.4 times the prompt tokens of the adapter arms.

## Repository contents

```text
paper/          camera-ready PDF, LaTeX, generated tables and figures
experiments/    minimal scripts for data, training, adapters, evaluation, and analysis
results/        exact frozen JSON files consumed by the paper build
docs/           environment, result map, and reproduction instructions
```

This repository intentionally excludes:

- model and adapter weights;
- the TOFU dataset;
- upstream Doc-to-LoRA source and checkpoint;
- internal development history;
- exploratory and superseded experiments;
- server-management logs and private infrastructure.

## Quick verification

No GPU is required:

```bash
python3 paper/camera_ready_analysis.py
python3 paper/make_tables.py
python3 paper/make_figs.py
./paper/build.sh
```

See [docs/REPRODUCE.md](docs/REPRODUCE.md) for the full GPU procedure and
[docs/RESULTS.md](docs/RESULTS.md) for the artifact map.

## Paper

[Camera-ready PDF](paper/build/main.pdf)

## License

Code is MIT licensed. Frozen result artifacts are released under CC BY 4.0.
Third-party assets remain under their original licenses and are not
redistributed.
