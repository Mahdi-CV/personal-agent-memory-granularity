# Frozen result map

Every manuscript number is generated from the committed JSON files by
`paper/make_tables.py`.

| Directory | Purpose |
|---|---|
| `results/results_gran/` | Compiled granularity ladder, router results, build statistics, and activity gates |
| `results/results_regimes/` | Fixed-update and equal-exposure gradient ladders and controls |
| `results/results_matched/` | Step-matched gradient operating point |
| `results/results_threeway/` | Base, retrieval, compiled, and gradient operating points |
| `results/results_full_context/` | Full-context baseline and prompt-length audit |
| `results/reanalysis/` | Direct question-form interaction and author-clustered uncertainty |

## Rebuild the reported values

```bash
python3 paper/camera_ready_analysis.py
python3 paper/make_tables.py
```

The table generator asserts:

- rank 16 at every compiled granularity;
- exactly 200 records in every memory organization;
- no compiler-input truncation;
- activity-gate success for every evaluated adapter;
- exactly 249 updates in the fixed-update ladder;
- the reported result ordering;
- prompt-length and no-truncation checks for full context.

## Two independent 20-record banks

Figure 1 reports 0.746 from the seed-controlled granularity bank. Table 1
reports 0.779 from an earlier independently trained bank used for routing and
systems accounting. Both use rank 16, `down_proj`, and 249 updates. They are
different adapter artifacts and evaluation passes.
