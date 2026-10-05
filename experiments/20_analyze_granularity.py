"""Analyze the granularity curve and the routing tax.

STATISTICAL POSITION

The same 200 TOFU items recur in every cell. Pooling cells as if they were
independent observations would count each item once per granularity and shrink
every interval by a factor that has nothing to do with evidence. This project
has already made that error once, so it is guarded against explicitly here:

  * Within a granularity, paraphrase robustness is a PAIRED item-level question
    (same item, two question forms, same adapter), so it is tested with exact
    McNemar on discordant pairs.
  * Across granularities, the comparison is also PAIRED at the item level (same
    item, two different adapters), so the curve is compared with paired tests
    rather than two independent proportions.
  * Any aggregate that mixes repeated measurements of the same items uses an
    ITEM-CLUSTERED bootstrap: resample ITEMS with replacement, and carry all of
    that item's measurements together. Resampling measurements instead of items
    would treat the repeats as independent, which is the error above.

Per the user's instruction, effect sizes and the shape of the curve lead; a
p-value is reported where a pairing genuinely exists, not manufactured at every
point.

THE ROUTING TAX IS MEASURED, NOT INFERRED

Because oracle and predicted routing run over identical items, the accuracy
difference decomposes exactly: only items the router misrouted can differ, so
the loss attributable to routing is computed directly from those items rather
than being estimated from the two headline rates.
"""
import json
import math
import os
import random
import sys

RES = "/work/d2l_exp/results"
GRAN = os.path.join(RES, "granularity.json")
RSGD = os.path.join(RES, "router_sgd.json")
STATS = "/work/d2l_exp/gran_build_stats.json"
OUT = os.path.join(RES, "GRANULARITY_ANALYSIS.json")


def binom_cdf(k, n, p=0.5):
    return sum(math.comb(n, i) * p ** n for i in range(k + 1))


def exact_binom_two_sided(a, b):
    """Exact McNemar on discordant counts a (lost) and b (gained)."""
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2.0 * binom_cdf(k, n))


def paired(h1, h2):
    """Paired comparison over items gradable in BOTH conditions.

    hit == -1 means the grader could not score that generation, so the item is
    dropped PAIRWISE. Treating ungradable as wrong would silently convert a
    grader abstention into an error against whichever arm abstains more.
    """
    lost = gained = both = neither = 0
    idx_lost, idx_gained = [], []
    for i, (a, b) in enumerate(zip(h1, h2)):
        if a < 0 or b < 0:
            continue
        if a and not b:
            lost += 1
            idx_lost.append(i)
        elif b and not a:
            gained += 1
            idx_gained.append(i)
        elif a and b:
            both += 1
        else:
            neither += 1
    n = lost + gained + both + neither
    return {"n_paired": n, "lost": lost, "gained": gained,
            "both": both, "neither": neither,
            "acc1": (both + lost) / n if n else None,
            "acc2": (both + gained) / n if n else None,
            "retention": (both / (both + lost)) if (both + lost) else None,
            "p_mcnemar": exact_binom_two_sided(lost, gained),
            "idx_lost": idx_lost, "idx_gained": idx_gained}


def item_clustered_bootstrap(cells, stat, n_boot=5000, seed=0):
    """Resample ITEMS (not measurements) and recompute `stat`.

    cells: list of per-item hit lists, all aligned on the same item order.
    stat:  callable(list_of_resampled_cells) -> float
    """
    rng = random.Random(seed)
    n = len(cells[0])
    assert all(len(c) == n for c in cells)
    out = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        out.append(stat([[c[i] for i in idx] for c in cells]))
    out.sort()
    lo = out[int(0.025 * len(out))]
    hi = out[int(0.975 * len(out)) - 1]
    return lo, hi


def acc(h):
    g = [x for x in h if x >= 0]
    return sum(g) / len(g) if g else None


def main():
    gran = json.load(open(GRAN))
    build = json.load(open(STATS))
    out = {"note": build.get("note"), "granularity": {}, "routing_tax": {},
           "curve_paired": {}, "sgd": {}}

    gs = sorted(build["granularities"], key=int)

    # ---- the curve, per granularity -------------------------------------
    for g in gs:
        b = build["granularities"][g]
        cell = lambda r, q: gran["cells"].get("g%s/%s/%s" % (g, r, q))  # noqa: E731
        oq, opq = cell("oracle", "q"), cell("oracle", "pq")
        pq_, ppq = cell("predicted", "q"), cell("predicted", "pq")
        if not (oq and opq):
            continue
        rob = paired(oq["hits"]["keyfacts"], opq["hits"]["keyfacts"])
        lo, hi = item_clustered_bootstrap(
            [oq["hits"]["keyfacts"]], lambda cs: acc(cs[0]) or 0.0)
        row = {
            "facts_per_adapter": int(g),
            "n_adapters": b["n_adapters"],
            "rank": b["ranks"],
            "per_adapter_bytes": b["per_active_adapter_bytes"],
            "total_memory_bytes": b["total_bytes"],
            "resident_bytes_one_active": b["per_active_adapter_bytes"],
            "total_build_seconds": b["total_build_seconds"],
            "per_adapter_build_seconds": round(
                b["total_build_seconds"] / b["n_adapters"], 3),
            "doc_tokens": sorted(a["doc_tokens_d2l"]
                                 for a in b["adapters"].values()),
            "all_tokens_consumed": all(a["all_tokens_consumed"]
                                       for a in b["adapters"].values()),
            "oracle_q": oq["keyfacts_acc"],
            "oracle_q_ci95_itemboot": [round(lo, 4), round(hi, 4)],
            "oracle_pq": opq["keyfacts_acc"],
            "paraphrase_paired_oracle": rob,
        }
        if pq_ and ppq:
            row["predicted_q"] = pq_["keyfacts_acc"]
            row["predicted_pq"] = ppq["keyfacts_acc"]
            row["router_top1_q"] = sum(pq_["router_correct"]) / pq_["n"]
            row["router_top1_pq"] = sum(ppq["router_correct"]) / ppq["n"]
            for qk, o, p in (("q", oq, pq_), ("pq", opq, ppq)):
                mis = [i for i, c in enumerate(p["router_correct"]) if not c]
                # Exact decomposition: only misrouted items can differ.
                diff = [i for i in range(p["n"])
                        if o["hits"]["keyfacts"][i] != p["hits"]["keyfacts"][i]]
                out["routing_tax"]["g%s/%s" % (g, qk)] = {
                    "n_misrouted": len(mis),
                    "misrouted_idx": mis,
                    "oracle_acc": o["keyfacts_acc"],
                    "predicted_acc": p["keyfacts_acc"],
                    "delta": (p["keyfacts_acc"] - o["keyfacts_acc"])
                    if (p["keyfacts_acc"] is not None
                        and o["keyfacts_acc"] is not None) else None,
                    "n_items_changed": len(diff),
                    "changed_idx": diff,
                    "all_changes_are_misroutes": set(diff).issubset(set(mis)),
                    "paired_oracle_vs_predicted": paired(
                        o["hits"]["keyfacts"], p["hits"]["keyfacts"]),
                }
        out["granularity"][g] = row

    # ---- paired comparison ALONG the curve -------------------------------
    # Same items, different adapters, so this is paired at the item level.
    for i in range(len(gs) - 1):
        a, b = gs[i], gs[i + 1]
        ca = gran["cells"].get("g%s/oracle/q" % a)
        cb = gran["cells"].get("g%s/oracle/q" % b)
        if ca and cb:
            out["curve_paired"]["g%s_vs_g%s/q" % (a, b)] = paired(
                ca["hits"]["keyfacts"], cb["hits"]["keyfacts"])
    if gs:
        ca = gran["cells"].get("g%s/oracle/q" % gs[0])
        cb = gran["cells"].get("g%s/oracle/q" % gs[-1])
        if ca and cb:
            out["curve_paired"]["g%s_vs_g%s/q" % (gs[0], gs[-1])] = paired(
                ca["hits"]["keyfacts"], cb["hits"]["keyfacts"])

    # ---- SGD under the same router ---------------------------------------
    if os.path.exists(RSGD):
        rs = json.load(open(RSGD))
        arm = rs["arm"]
        for qk in ("q", "pq"):
            o = rs["cells"].get("%s/oracle/%s" % (arm, qk))
            p = rs["cells"].get("%s/predicted/%s" % (arm, qk))
            if not (o and p):
                continue
            mis = [i for i, c in enumerate(p["router_correct"]) if not c]
            diff = [i for i in range(p["n"])
                    if o["hits"]["keyfacts"][i] != p["hits"]["keyfacts"][i]]
            out["sgd"][qk] = {
                "oracle_acc": o["keyfacts_acc"],
                "predicted_acc": p["keyfacts_acc"],
                "delta": p["keyfacts_acc"] - o["keyfacts_acc"],
                "router_top1": sum(p["router_correct"]) / p["n"],
                "n_misrouted": len(mis),
                "n_items_changed": len(diff),
                "all_changes_are_misroutes": set(diff).issubset(set(mis)),
                "paired_oracle_vs_predicted": paired(
                    o["hits"]["keyfacts"], p["hits"]["keyfacts"]),
            }
        oq = rs["cells"].get("%s/oracle/q" % arm)
        opq = rs["cells"].get("%s/oracle/pq" % arm)
        if oq and opq:
            out["sgd"]["paraphrase_paired_oracle"] = paired(
                oq["hits"]["keyfacts"], opq["hits"]["keyfacts"])
        pq_ = rs["cells"].get("%s/predicted/q" % arm)
        ppq = rs["cells"].get("%s/predicted/pq" % arm)
        if pq_ and ppq:
            out["sgd"]["paraphrase_paired_predicted"] = paired(
                pq_["hits"]["keyfacts"], ppq["hits"]["keyfacts"])

        # SGD vs D2L at matched granularity (g=20), paired, predicted routing:
        # the deployable head-to-head.
        d2l_p = gran["cells"].get("g20/predicted/q")
        if pq_ and d2l_p:
            out["sgd"]["vs_d2l_g20_predicted_q"] = paired(
                pq_["hits"]["keyfacts"], d2l_p["hits"]["keyfacts"])

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

    # ---- console table ---------------------------------------------------
    print("%-6s %-4s %-7s %-7s %-7s %-7s %-6s %-8s %-9s %-7s" % (
        "facts", "nad", "orc_q", "orc_pq", "prd_q", "prd_pq", "ret",
        "totMB", "residMB", "build_s"))
    for g in gs:
        r = out["granularity"].get(g)
        if not r:
            continue
        pr = r["paraphrase_paired_oracle"]
        print("%-6s %-4d %-7s %-7s %-7s %-7s %-6s %-8.1f %-9.1f %-7.2f" % (
            g, r["n_adapters"],
            "%.3f" % r["oracle_q"] if r["oracle_q"] is not None else "-",
            "%.3f" % r["oracle_pq"] if r["oracle_pq"] is not None else "-",
            "%.3f" % r["predicted_q"] if r.get("predicted_q") is not None else "-",
            "%.3f" % r["predicted_pq"] if r.get("predicted_pq") is not None else "-",
            "%.3f" % pr["retention"] if pr["retention"] is not None else "-",
            r["total_memory_bytes"] / 1e6,
            r["resident_bytes_one_active"] / 1e6,
            r["total_build_seconds"]))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
