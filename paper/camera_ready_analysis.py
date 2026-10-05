#!/usr/bin/env python3
"""Camera-ready sensitivity analyses requested by the reviewers.

Uses only frozen per-item result artifacts. No model inference is required.

Primary output:
  - direct interaction between question form and granularity:
      (paraphrase g20-g200 difference) - (original g20-g200 difference)
  - item-bootstrap and author-clustered bootstrap confidence intervals
  - exact author-level sign-flip sensitivity test
"""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reanalysis" / "camera_ready_interaction.json"


def load_cells(path: str) -> dict:
    return json.loads((ROOT / path).read_text())["cells"]


def percentile_interval(values: list[float]) -> list[float]:
    values = sorted(values)
    n = len(values)
    return [values[int(0.025 * n)], values[int(0.975 * n) - 1]]


def analyze(name: str, q20: list[int], q200: list[int],
            pq20: list[int], pq200: list[int],
            *, n_boot: int = 50_000, seed: int = 42) -> dict:
    """Positive interaction means coarsening hurts paraphrases more."""
    item_effects: list[float] = []
    author_effects: list[float] = []

    for values in zip(q20, q200, pq20, pq200):
        if min(values) < 0:
            continue
        oq20, oq200, opq20, opq200 = values
        item_effects.append((opq20 - opq200) - (oq20 - oq200))

    for author_idx in range(10):
        effects: list[float] = []
        lo, hi = author_idx * 20, (author_idx + 1) * 20
        for i in range(lo, hi):
            values = q20[i], q200[i], pq20[i], pq200[i]
            if min(values) < 0:
                continue
            oq20, oq200, opq20, opq200 = values
            effects.append((opq20 - opq200) - (oq20 - oq200))
        author_effects.append(sum(effects) / len(effects))

    observed = sum(item_effects) / len(item_effects)
    rng = random.Random(seed)
    item_boot: list[float] = []
    author_boot: list[float] = []
    for _ in range(n_boot):
        sample = [item_effects[rng.randrange(len(item_effects))]
                  for _ in range(len(item_effects))]
        item_boot.append(sum(sample) / len(sample))

        author_ids = [rng.randrange(10) for _ in range(10)]
        author_boot.append(sum(author_effects[i] for i in author_ids) / 10)

    author_observed = sum(author_effects) / len(author_effects)
    sign_flips = [
        abs(sum(sign * effect for sign, effect in zip(signs, author_effects)) / 10)
        for signs in itertools.product((-1, 1), repeat=10)
    ]
    p_sign_flip = (
        sum(value >= abs(author_observed) - 1e-12 for value in sign_flips)
        / len(sign_flips)
    )

    return {
        "name": name,
        "definition": (
            "(paraphrase accuracy g20-g200) - "
            "(original accuracy g20-g200); positive means coarsening has a "
            "larger penalty on paraphrased questions"
        ),
        "n_items_gradable_in_all_four_cells": len(item_effects),
        "interaction": observed,
        "item_bootstrap_ci95": percentile_interval(item_boot),
        "author_effects": author_effects,
        "author_mean_interaction": author_observed,
        "author_clustered_bootstrap_ci95": percentile_interval(author_boot),
        "author_exact_sign_flip_p": p_sign_flip,
        "n_author_clusters": 10,
        "bootstrap_replicates": n_boot,
        "seed": seed,
    }


def endpoint(name: str, g20: list[int], g200: list[int],
             *, n_boot: int = 50_000, seed: int = 42) -> dict:
    """Author-normalized g20 minus g200 endpoint comparison."""
    item_effects: list[float] = []
    author_effects: list[float] = []

    for left, right in zip(g20, g200):
        if left < 0 or right < 0:
            continue
        item_effects.append(left - right)

    for author_idx in range(10):
        effects: list[float] = []
        lo, hi = author_idx * 20, (author_idx + 1) * 20
        for i in range(lo, hi):
            left, right = g20[i], g200[i]
            if left < 0 or right < 0:
                continue
            effects.append(left - right)
        author_effects.append(sum(effects) / len(effects))

    author_observed = sum(author_effects) / len(author_effects)
    rng = random.Random(seed)
    author_boot: list[float] = []
    for _ in range(n_boot):
        author_ids = [rng.randrange(10) for _ in range(10)]
        author_boot.append(sum(author_effects[i] for i in author_ids) / 10)

    sign_flips = [
        abs(sum(sign * effect for sign, effect in zip(signs, author_effects)) / 10)
        for signs in itertools.product((-1, 1), repeat=10)
    ]
    p_sign_flip = (
        sum(value >= abs(author_observed) - 1e-12 for value in sign_flips)
        / len(sign_flips)
    )

    return {
        "name": name,
        "definition": "author-normalized accuracy difference: g20 minus g200",
        "n_items_gradable_in_both_cells": len(item_effects),
        "item_mean_difference": sum(item_effects) / len(item_effects),
        "author_effects": author_effects,
        "author_mean_difference": author_observed,
        "author_clustered_bootstrap_ci95": percentile_interval(author_boot),
        "author_exact_sign_flip_p": p_sign_flip,
        "n_author_clusters": 10,
        "bootstrap_replicates": n_boot,
        "seed": seed,
    }


def main() -> None:
    compiled = load_cells("results/results_gran/granularity.json")
    fixed = load_cells("results/results_regimes/granularity_seeded249.json")
    exposure = load_cells("results/results_regimes/granularity_sgd_sweep.json")

    def h(cells: dict, key: str) -> list[int]:
        return cells[key]["hits"]["keyfacts"]

    result = {
        "compiled": analyze(
            "compiled",
            h(compiled, "g20/oracle/q"), h(compiled, "g200/oracle/q"),
            h(compiled, "g20/oracle/pq"), h(compiled, "g200/oracle/pq"),
        ),
        "fixed_249_updates": analyze(
            "fixed_249_updates",
            h(fixed, "g20s249/oracle/q"), h(fixed, "g200s249/oracle/q"),
            h(fixed, "g20s249/oracle/pq"), h(fixed, "g200s249/oracle/pq"),
        ),
        "equal_83_epochs": analyze(
            "equal_83_epochs",
            h(exposure, "g20e83/oracle/q"), h(exposure, "g200e83/oracle/q"),
            h(exposure, "g20e83/oracle/pq"), h(exposure, "g200e83/oracle/pq"),
        ),
        "endpoint_sensitivity": {
            "compiled_original": endpoint(
                "compiled_original",
                h(compiled, "g20/oracle/q"), h(compiled, "g200/oracle/q"),
            ),
            "fixed_249_original": endpoint(
                "fixed_249_original",
                h(fixed, "g20s249/oracle/q"),
                h(fixed, "g200s249/oracle/q"),
            ),
            "equal_83_original": endpoint(
                "equal_83_original",
                h(exposure, "g20e83/oracle/q"),
                h(exposure, "g200e83/oracle/q"),
            ),
            "equal_83_paraphrased": endpoint(
                "equal_83_paraphrased",
                h(exposure, "g20e83/oracle/pq"),
                h(exposure, "g200e83/oracle/pq"),
            ),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
