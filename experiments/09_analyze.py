"""Paired analysis of the three-way comparison.

All arms answered the SAME 200 items, so every comparison here is PAIRED at the
item level. Two test families:

  WITHIN-ARM paraphrase drop: for one arm, q vs pq on the same item. Exact
  McNemar (binomial on discordant pairs). This is comparison A's core number.

  BETWEEN-ARM at fixed condition: arm X vs arm Y on the same item. Also exact
  McNemar. Used for D2L vs retrieval (B) and full vs matched SGD (D).

Items where the keyfact grader is ungradable (hit == -1) are dropped PAIRWISE,
not globally, so each test uses every item valid for that specific pair.

No pooling across arms: the same 200 items appear in every cell, so pooling
would count each item repeatedly. This is the same clustering error already
corrected once in this project.
"""
import json
import sys
from math import comb

RES = "/work/d2l_exp/results/threeway.json"
OUT = "/work/d2l_exp/results/ANALYSIS.json"


def exact_mcnemar(a_hits, b_hits):
    """Two-sided exact McNemar on paired binary outcomes. -1 = ungradable."""
    lost = gained = 0
    n = 0
    for x, y in zip(a_hits, b_hits):
        if x < 0 or y < 0:
            continue
        n += 1
        if x == 1 and y == 0:
            lost += 1
        elif x == 0 and y == 1:
            gained += 1
    d = lost + gained
    if d == 0:
        return {"n_paired": n, "lost": 0, "gained": 0, "p": 1.0}
    k = min(lost, gained)
    p = sum(comb(d, i) for i in range(k + 1)) / (2 ** d) * 2
    return {"n_paired": n, "lost": lost, "gained": gained, "p": min(1.0, p)}


def main():
    blob = json.load(open(RES))
    cells = blob["cells"]
    ARMS = ["base", "sgd7", "sgddp", "d2l_pa", "d2l_all", "retrieval"]
    out = {"base_model": blob["base_model"], "n_items": blob["n_trained_items"]}

    # ---- A/D: within-arm paraphrase drop ----
    print("=" * 78)
    print("PARAPHRASE DROP (same item, original q -> paraphrased pq)")
    print("%-10s %8s %8s %8s   %5s %6s  %s" %
          ("arm", "q", "pq", "drop", "lost", "gained", "p"))
    print("-" * 78)
    drops = {}
    for a in ARMS:
        cq, cp = cells["%s/q" % a], cells["%s/pq" % a]
        m = exact_mcnemar(cq["hits"]["keyfacts"], cp["hits"]["keyfacts"])
        d = (cq["keyfacts_acc"] or 0) - (cp["keyfacts_acc"] or 0)
        drops[a] = {"q": cq["keyfacts_acc"], "pq": cp["keyfacts_acc"],
                    "drop": d, **m}
        print("%-10s %8.3f %8.3f %+8.3f   %5d %6d  %.3g" %
              (a, cq["keyfacts_acc"] or 0, cp["keyfacts_acc"] or 0, -d,
               m["lost"], m["gained"], m["p"]))
    out["paraphrase_drop"] = drops

    # ---- between-arm comparisons that answer A-D ----
    KEY = [
        ("A  SGD7 vs D2L-perauthor", "sgd7", "d2l_pa"),
        ("A' SGD-dp vs D2L-perauthor", "sgddp", "d2l_pa"),
        ("B  D2L-perauthor vs retrieval", "d2l_pa", "retrieval"),
        ("D  SGD7 vs SGD-downproj", "sgd7", "sgddp"),
        ("   D2L-perauthor vs D2L-all200", "d2l_pa", "d2l_all"),
        ("   SGD7 vs retrieval", "sgd7", "retrieval"),
    ]
    print()
    print("=" * 78)
    print("BETWEEN-ARM, PAIRED (positive 'net' favours the FIRST arm)")
    btw = {}
    for qkey in ("q", "pq"):
        print("-- question = %s --" % qkey)
        for label, x, y in KEY:
            cx, cy = cells["%s/%s" % (x, qkey)], cells["%s/%s" % (y, qkey)]
            m = exact_mcnemar(cx["hits"]["keyfacts"], cy["hits"]["keyfacts"])
            net = m["lost"] - m["gained"]
            btw["%s|%s" % (label.strip(), qkey)] = {
                "x": x, "y": y, "x_acc": cx["keyfacts_acc"],
                "y_acc": cy["keyfacts_acc"], "net_favouring_x": net, **m}
            print("  %-32s %.3f vs %.3f  net %+4d  p=%.3g" %
                  (label, cx["keyfacts_acc"] or 0, cy["keyfacts_acc"] or 0,
                   net, m["p"]))
    out["between_arm"] = btw

    # ---- C: interference ----
    print()
    print("=" * 78)
    print("INTERFERENCE (vs base; negative = memory install damaged the model)")
    print("%-10s %10s %10s %10s %10s" % ("arm", "near", "d-near", "far", "d-far"))
    print("-" * 78)
    bn = cells["base/near"]["keyfacts_acc"] or 0
    bf = cells["base/far"]["far_acc"] or 0
    inter = {}
    for a in ARMS:
        nn = cells["%s/near" % a]["keyfacts_acc"] or 0
        ff = cells["%s/far" % a]["far_acc"] or 0
        mn = exact_mcnemar(cells["base/near"]["hits"]["keyfacts"],
                           cells["%s/near" % a]["hits"]["keyfacts"])
        mf = exact_mcnemar(cells["base/far"]["hits"]["far"],
                           cells["%s/far" % a]["hits"]["far"])
        inter[a] = {"near": nn, "near_delta": nn - bn, "near_p": mn["p"],
                    "far": ff, "far_delta": ff - bf, "far_p": mf["p"],
                    "near_test": mn, "far_test": mf}
        print("%-10s %10.3f %+10.3f %10.3f %+10.3f" % (a, nn, nn - bn, ff, ff - bf))
    out["interference"] = inter

    # ---- systems tradeoffs ----
    try:
        bs = json.load(open("/work/d2l_exp/build_stats.json"))
    except FileNotFoundError:
        bs = {}
    sysrows = {}
    for a in ARMS:
        sysrows[a] = {
            "mean_prompt_tokens_q": cells["%s/q" % a]["mean_prompt_tokens"],
            "mean_prompt_tokens_pq": cells["%s/pq" % a]["mean_prompt_tokens"],
        }
    out["systems"] = {"per_arm": sysrows, "d2l_build": bs.get("adapters", {})}
    print()
    print("prompt tokens/query: base %.1f | retrieval %.1f (%.2fx)" % (
        cells["base/q"]["mean_prompt_tokens"],
        cells["retrieval/q"]["mean_prompt_tokens"],
        cells["retrieval/q"]["mean_prompt_tokens"] / cells["base/q"]["mean_prompt_tokens"]))

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
