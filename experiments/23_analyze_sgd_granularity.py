"""Analyze the gradient granularity x training-exposure sweep.

Imports paired() and exact_binom_two_sided() directly from
20_analyze_granularity.py so this arm cannot silently use a different retention
or significance definition than the paper. Retention is both/(both+lost),
matching Table 1's caption; significance is an exact two-sided McNemar.

Reports the FULL learning curves, not just the best point, per
PREREGISTER-gradient-ladder.md. Where a best-over-grid number appears it is
labelled as a maximum over a predeclared grid, which is an optimistic
estimator, and the whole curve is printed beside it.

Per condition (granularity x exposure):
  original-question recall, paraphrase recall, retention, optimizer steps,
  epochs, examples seen, total and per-adapter build time, routing accuracy
  and routing delta.

Then:
  - paired lost/gained + exact McNemar along the granularity ladder at each
    exposure level, and at each granularity's best exposure
  - the ordering test: does g20 > g40 > g100 > g200 hold, at each exposure
  - exposure sensitivity per granularity (does recall rise or fall with more
    training)
  - compiled vs gradient at matched granularity
  - measured run-to-run noise floor from the duplicate endpoint passes
"""
import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


a20 = load("a20", "20_analyze_granularity.py")
paired = a20.paired
exact_binom_two_sided = a20.exact_binom_two_sided

GRANS = [20, 40, 100, 200]

# The paper's frozen per-adapter gradient budget: tables/macros.tex:57,
# \SgdOptSteps = 249 (\SgdEpochs = 83), stated as held constant across all four
# granularities. Ladder A must match this EXACTLY, not to within a few updates.
EXACT_REF = 249
EXACT_EXPOSURE_EPOCHS = 83  # ladder B: fixed exposure at every granularity


def ckey(g, e):
    return "g%d/e%d" % (g, e)


def cell(res, g, e, routing, qkey):
    """Sweep results are keyed g<g>e<e>/<routing>/<qkey>."""
    return res["cells"].get("g%de%d/%s/%s" % (g, e, routing, qkey))


def hits_of(c):
    return c["hits"]["keyfacts"] if c else None


def acc(c):
    return c["keyfacts_acc"] if c else None


def fmt(x, n=3):
    if x is None:
        return "-"
    return ("%." + str(n) + "f") % x if isinstance(x, float) else str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sgd", default="/work/d2l_exp/results/granularity_sgd_sweep.json")
    ap.add_argument("--d2l", default="/work/d2l_exp/results/granularity.json")
    ap.add_argument("--sgd-build", default="/work/d2l_exp/sgd_gran_build_stats.json")
    ap.add_argument("--d2l-build", default="/work/d2l_exp/gran_build_stats.json")
    ap.add_argument("--gate", default="/work/d2l_exp/activity_gate_sgdgr_.json")
    ap.add_argument("--repeat",
                    default="/work/d2l_exp/results/granularity_sgd_repeat.json",
                    help="duplicate endpoint passes, for the noise floor")
    ap.add_argument("--out",
                    default="/work/d2l_exp/results/SGD_SWEEP_ANALYSIS.json")
    args = ap.parse_args()

    sgd = json.load(open(args.sgd))
    build = json.load(open(args.sgd_build))
    d2l = json.load(open(args.d2l)) if os.path.exists(args.d2l) else None
    gate = json.load(open(args.gate)) if os.path.exists(args.gate) else None

    conds = build["conditions"]
    # exposure levels actually built, per granularity
    levels = {}
    for k, v in conds.items():
        levels.setdefault(v["facts_per_adapter"], []).append(v["epochs"])
    for g in levels:
        levels[g] = sorted(set(levels[g]))

    out = {
        "arm": "sgd_granularity_x_exposure",
        "design": build.get("design"),
        "preregistered": build.get("preregistered"),
        "exposure_grid_epochs": build.get("exposure_grid_epochs"),
        "hyperparams": {k: build.get(k) for k in
                        ("base", "rank", "alpha", "lr", "target_modules",
                         "grad_accum", "reference_optimizer_steps")},
        "conditions": {}, "curves": {}, "ladder_paired": {},
        "ordering": {}, "exposure_sensitivity": {}, "vs_compiled": {},
        "noise_floor": None,
    }

    # ---- per-condition table ----
    for g in GRANS:
        for e in levels.get(g, []):
            k = ckey(g, e)
            b = conds.get(k)
            if not b:
                continue
            oq = cell(sgd, g, e, "oracle", "q")
            opq = cell(sgd, g, e, "oracle", "pq")
            pq_ = cell(sgd, g, e, "predicted", "q")
            ppq = cell(sgd, g, e, "predicted", "pq")
            row = {
                "facts_per_adapter": g, "epochs": e,
                "n_adapters": b["n_adapters"],
                "optimizer_steps_per_adapter": b["optimizer_steps_per_adapter"],
                "examples_seen_per_adapter": b["examples_seen_per_adapter"],
                # Derived HERE, not read from the build stats, because the sweep
                # that produced these artifacts predates the is_exact_249 field.
                # Only g20/e83 hits 249 exactly; g40/e50=250, g100/e19=247,
                # g200/e10=250. "nearest epoch to ref" is NOT "exact 249".
                "is_nearest_epoch_to_ref": b["is_step_matched"],
                "is_exact_249": b["optimizer_steps_per_adapter"] == EXACT_REF,
                "steps_minus_ref": b["optimizer_steps_per_adapter"] - EXACT_REF,
                "total_build_seconds": b["total_build_seconds"],
                "per_adapter_build_seconds": round(
                    b["total_build_seconds"] / b["n_adapters"], 2),
                "total_MB": round(b["total_bytes"] / 1e6, 1),
                "resident_MB": round(b["per_active_adapter_bytes"] / 1e6, 1),
                "oracle_q": acc(oq), "oracle_pq": acc(opq),
                "predicted_q": acc(pq_), "predicted_pq": acc(ppq),
            }
            if oq and opq:
                pr = paired(hits_of(oq), hits_of(opq))
                row["retention_oracle"] = pr["retention"]
                row["retention_paired"] = {kk: pr[kk] for kk in
                                           ("n_paired", "lost", "gained",
                                            "both", "neither", "p_mcnemar")}
            if oq and oq.get("router_correct"):
                rc = oq["router_correct"]
                row["router_top1_q"] = sum(rc) / len(rc)
            if oq and pq_:
                row["routing_delta_q"] = round(acc(oq) - acc(pq_), 4)
            if opq and ppq:
                row["routing_delta_pq"] = round(acc(opq) - acc(ppq), 4)
            out["conditions"][k] = row

    # ---- learning curves per granularity ----
    for g in GRANS:
        pts = []
        for e in levels.get(g, []):
            r = out["conditions"].get(ckey(g, e))
            if r:
                pts.append({"epochs": e,
                            "steps": r["optimizer_steps_per_adapter"],
                            "oracle_q": r["oracle_q"],
                            "oracle_pq": r["oracle_pq"],
                            "retention": r.get("retention_oracle"),
                            "is_nearest_epoch_to_ref": r["is_nearest_epoch_to_ref"],
                            "is_exact_249": r["is_exact_249"],
                            "steps_minus_ref": r["steps_minus_ref"]})
        if not pts:
            continue
        vals = [p for p in pts if p["oracle_q"] is not None]
        best = max(vals, key=lambda p: p["oracle_q"]) if vals else None
        sm = next((p for p in pts if p["is_nearest_epoch_to_ref"]), None)
        out["curves"][str(g)] = {
            "points": pts,
            "best_epochs": best["epochs"] if best else None,
            "best_oracle_q": best["oracle_q"] if best else None,
            "best_is_max_over_predeclared_grid": True,
            "nearest_epoch_to_ref_epochs": sm["epochs"] if sm else None,
            "nearest_epoch_to_ref_steps": sm["steps"] if sm else None,
            "nearest_epoch_is_exact_249": bool(sm and sm["is_exact_249"]),
            "step_matched_epochs": sm["epochs"] if sm else None,   # legacy key
            "step_matched_oracle_q": sm["oracle_q"] if sm else None,
            "gain_from_extra_exposure": (
                round(best["oracle_q"] - sm["oracle_q"], 4)
                if best and sm and sm["oracle_q"] is not None else None),
        }

    # ---- paired tests along the ladder, at each exposure level ----
    common = sorted(set.intersection(*[set(levels[g]) for g in GRANS])
                    ) if all(g in levels for g in GRANS) else []
    for e in common:
        for a, b_ in [(20, 40), (40, 100), (100, 200), (20, 200)]:
            ca = cell(sgd, a, e, "oracle", "q")
            cb = cell(sgd, b_, e, "oracle", "q")
            if ca and cb:
                pr = paired(hits_of(ca), hits_of(cb))
                out["ladder_paired"]["e%d/g%d->g%d" % (e, a, b_)] = {
                    kk: v for kk, v in pr.items()
                    if kk not in ("idx_lost", "idx_gained")}

    # ---- ladder at each granularity's BEST exposure ----
    bestc = {}
    for g in GRANS:
        c = out["curves"].get(str(g))
        if c and c["best_epochs"] is not None:
            bestc[g] = cell(sgd, g, c["best_epochs"], "oracle", "q")
    for a, b_ in [(20, 40), (40, 100), (100, 200), (20, 200)]:
        if bestc.get(a) and bestc.get(b_):
            pr = paired(hits_of(bestc[a]), hits_of(bestc[b_]))
            out["ladder_paired"]["best/g%d->g%d" % (a, b_)] = {
                kk: v for kk, v in pr.items()
                if kk not in ("idx_lost", "idx_gained")}

    # ---- ordering test ----
    for label, getter in (("by_exposure", None), ("best", None)):
        pass
    for e in common:
        seq = [acc(cell(sgd, g, e, "oracle", "q")) for g in GRANS]
        if all(v is not None for v in seq):
            out["ordering"]["e%d" % e] = {
                "seq": seq,
                "monotone_decreasing": all(seq[i] >= seq[i + 1]
                                           for i in range(len(seq) - 1)),
                "g20_gt_g200": seq[0] > seq[-1],
            }
    seqb = [out["curves"][str(g)]["best_oracle_q"] for g in GRANS
            if str(g) in out["curves"]]
    if len(seqb) == len(GRANS) and all(v is not None for v in seqb):
        out["ordering"]["best"] = {
            "seq": seqb,
            "monotone_decreasing": all(seqb[i] >= seqb[i + 1]
                                       for i in range(len(seqb) - 1)),
            "g20_gt_g200": seqb[0] > seqb[-1],
        }

    # ---- exposure sensitivity per granularity ----
    for g in GRANS:
        c = out["curves"].get(str(g))
        if not c:
            continue
        vals = [(p["epochs"], p["oracle_q"]) for p in c["points"]
                if p["oracle_q"] is not None]
        if len(vals) < 2:
            continue
        lo, hi = vals[0], vals[-1]
        cl = cell(sgd, g, lo[0], "oracle", "q")
        ch = cell(sgd, g, hi[0], "oracle", "q")
        d = {"range_epochs": [lo[0], hi[0]],
             "range_oracle_q": [lo[1], hi[1]],
             "spread": round(max(v for _, v in vals) - min(v for _, v in vals), 4),
             "monotone_increasing": all(vals[i][1] <= vals[i + 1][1]
                                        for i in range(len(vals) - 1)),
             "peaks_interior": (max(vals, key=lambda t: t[1])[0]
                                not in (lo[0], hi[0]))}
        if cl and ch:
            pr = paired(hits_of(cl), hits_of(ch))
            d["paired_low_to_high"] = {kk: v for kk, v in pr.items()
                                       if kk not in ("idx_lost", "idx_gained")}
        out["exposure_sensitivity"][str(g)] = d

    # ---- compiled vs gradient ----
    if d2l:
        def d2lcell(g, routing, qkey):
            return d2l["cells"].get("g%d/%s/%s" % (g, routing, qkey))
        for g in GRANS:
            cd = d2lcell(g, "oracle", "q")
            c = out["curves"].get(str(g))
            if not (cd and c and c["best_epochs"] is not None):
                continue
            cs = cell(sgd, g, c["best_epochs"], "oracle", "q")
            csm = (cell(sgd, g, c["step_matched_epochs"], "oracle", "q")
                   if c["step_matched_epochs"] else None)
            entry = {"compiled_q": acc(cd),
                     "gradient_best_q": acc(cs),
                     "gradient_best_epochs": c["best_epochs"],
                     "gradient_step_matched_q": acc(csm) if csm else None}
            if cs:
                pr = paired(hits_of(cd), hits_of(cs))
                entry["paired_compiled_to_gradient_best"] = {
                    kk: v for kk, v in pr.items()
                    if kk not in ("idx_lost", "idx_gained")}
            out["vs_compiled"][str(g)] = entry

    # ---- measured noise floor from duplicate endpoint passes ----
    if os.path.exists(args.repeat):
        rep = json.load(open(args.repeat))
        nf = {}
        for k, c2 in rep["cells"].items():
            c1 = sgd["cells"].get(k)
            if not c1:
                continue
            h1, h2 = hits_of(c1), hits_of(c2)
            gens_diff = sum(1 for a, b_ in zip(c1["gens"], c2["gens"]) if a != b_)
            flips = sum(1 for a, b_ in zip(h1, h2) if a != b_)
            nf[k] = {"acc_pass1": acc(c1), "acc_pass2": acc(c2),
                     "abs_acc_delta": round(abs((acc(c1) or 0) - (acc(c2) or 0)), 4),
                     "generations_differing": gens_diff,
                     "graded_outcome_flips": flips,
                     "n": len(c1["gens"])}
        if nf:
            out["noise_floor"] = {
                "cells": nf,
                "max_abs_acc_delta": max(v["abs_acc_delta"] for v in nf.values()),
                "note": ("multi-LoRA decoding is not bitwise reproducible at "
                         "temperature 0 here; deltas at or below this are "
                         "unresolved, not effects"),
            }

    if gate:
        out["activity_gate"] = gate

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # ================= console report =================
    print("\nGRADIENT GRANULARITY x TRAINING-EXPOSURE SWEEP")
    print("=" * 108)
    print("%-6s %-7s %-6s %-7s %-8s %-8s %-9s %-9s %-8s %-8s" %
          ("g", "epochs", "n_ad", "steps", "orac_q", "orac_pq", "retention",
           "seen", "build_s", "per_ad_s"))
    print("-" * 108)
    for g in GRANS:
        for e in levels.get(g, []):
            r = out["conditions"].get(ckey(g, e))
            if not r:
                continue
            print("%-6s %-7s %-6s %-7s %-8s %-8s %-9s %-9s %-8s %-8s%s" % (
                g, e, r["n_adapters"], r["optimizer_steps_per_adapter"],
                fmt(r["oracle_q"]), fmt(r["oracle_pq"]),
                fmt(r.get("retention_oracle")), r["examples_seen_per_adapter"],
                fmt(r["total_build_seconds"], 1),
                fmt(r["per_adapter_build_seconds"], 1),
                "  [STEP-MATCHED]" if r["is_step_matched"] else ""))
        print("-" * 108)

    print("\nLEARNING CURVES (oracle, original question)")
    for g in GRANS:
        c = out["curves"].get(str(g))
        if not c:
            continue
        pts = " ".join("e%d:%s" % (p["epochs"], fmt(p["oracle_q"]))
                       for p in c["points"])
        print("  g=%-4d %s" % (g, pts))
        print("         step-matched e%s=%s   best e%s=%s (max over predeclared grid)"
              % (c["step_matched_epochs"], fmt(c["step_matched_oracle_q"]),
                 c["best_epochs"], fmt(c["best_oracle_q"])))

    print("\nORDERING TEST (g20 >= g40 >= g100 >= g200 ?)")
    for k, v in out["ordering"].items():
        print("  %-8s seq %s  monotone=%s  g20>g200=%s"
              % (k, [fmt(x) for x in v["seq"]], v["monotone_decreasing"],
                 v["g20_gt_g200"]))

    print("\nEXPOSURE SENSITIVITY")
    for g in GRANS:
        d = out["exposure_sensitivity"].get(str(g))
        if not d:
            continue
        p = d.get("paired_low_to_high", {})
        print("  g=%-4d e%d->%d  %s -> %s  spread %.3f  peak_interior=%s%s"
              % (g, d["range_epochs"][0], d["range_epochs"][1],
                 fmt(d["range_oracle_q"][0]), fmt(d["range_oracle_q"][1]),
                 d["spread"], d["peaks_interior"],
                 ("  lost %d gained %d p=%.3g" % (p["lost"], p["gained"], p["p_mcnemar"])
                  if p else "")))

    print("\nPAIRED ALONG THE LADDER")
    for k in sorted(out["ladder_paired"]):
        v = out["ladder_paired"][k]
        print("  %-16s lost %-4d gained %-4d  %.3f -> %.3f  p=%.3g"
              % (k, v["lost"], v["gained"], v["acc1"], v["acc2"], v["p_mcnemar"]))

    if out["vs_compiled"]:
        print("\nCOMPILED vs GRADIENT (same items, oracle, original question)")
        for g in GRANS:
            v = out["vs_compiled"].get(str(g))
            if not v:
                continue
            p = v.get("paired_compiled_to_gradient_best", {})
            print("  g=%-4d compiled %s | gradient step-matched %s | gradient best %s (e%s)%s"
                  % (g, fmt(v["compiled_q"]), fmt(v["gradient_step_matched_q"]),
                     fmt(v["gradient_best_q"]), v["gradient_best_epochs"],
                     ("  lost %d gained %d p=%.3g"
                      % (p["lost"], p["gained"], p["p_mcnemar"]) if p else "")))

    if out["noise_floor"]:
        print("\nMEASURED NOISE FLOOR (duplicate endpoint passes)")
        for k, v in out["noise_floor"]["cells"].items():
            print("  %-24s %.3f vs %.3f  delta %.3f  gens_diff %d/%d  flips %d"
                  % (k, v["acc_pass1"], v["acc_pass2"], v["abs_acc_delta"],
                     v["generations_differing"], v["n"], v["graded_outcome_flips"]))
        print("  max abs accuracy delta: %.3f"
              % out["noise_floor"]["max_abs_acc_delta"])

    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
