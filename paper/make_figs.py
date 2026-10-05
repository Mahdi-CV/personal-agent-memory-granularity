#!/usr/bin/env python3
"""Generate the paper figures from the frozen result JSON.

    ~/.venvs/paper/bin/python paper2/make_figs.py

Writes paper2/figs/*.pdf. Every value is read from the frozen artifacts; no
number is typed here.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
GRAN = os.path.join(RES, "results_gran")
MATCHED = os.path.join(RES, "results_matched")
THREE = os.path.join(RES, "results_threeway")
REGIMES = os.path.join(RES, "results_regimes")
OUT = os.path.join(HERE, "figs")

GS = [20, 40, 100, 200]

# Okabe-Ito, colorblind safe; the paper may be printed greyscale.
C_ORIG = "#0072B2"
C_PARA = "#D55E00"
C_RETR = "#009E73"
C_BASE = "#999999"
C_SGD = "#CC79A7"

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def load(p):
    with open(p) as f:
        return json.load(f)


def main():
    A = load(os.path.join(GRAN, "GRANULARITY_ANALYSIS.json"))
    T = load(os.path.join(THREE, "threeway.json"))
    g = {k: A["granularity"][str(k)] for k in GS}
    cells = T["cells"]

    x = list(range(len(GS)))
    q = [g[k]["oracle_q"] for k in GS]
    pq = [g[k]["oracle_pq"] for k in GS]
    base_q = cells["base/q"]["keyfacts_acc"]

    # ---- Figure 1: three construction regimes ----------------------------
    # The point of the figure is that the granularity relationship is NOT one
    # curve: it depends on how the adapters were built. Panel (a) is original
    # questions, where the compiled and fixed-update regimes decline and the
    # equal-exposure regime does not; panel (b) is paraphrases, where every
    # regime declines. Storage accounting moved to the tables to make room.
    SEED = load(os.path.join(REGIMES, "granularity_seeded249.json"))["cells"]
    SWEEP = load(os.path.join(REGIMES, "granularity_sgd_sweep.json"))["cells"]

    def acc(cells, key):
        return [cells[key % k]["keyfacts_acc"] for k in GS]

    regimes = [
        ("compiled", C_ORIG, "o", "-",
         q, pq),
        ("gradient, 249 updates/ad.", C_SGD, "s", "--",
         acc(SEED, "g%ds249/oracle/q"), acc(SEED, "g%ds249/oracle/pq")),
        ("gradient, 83 epochs", C_RETR, "^", "-.",
         acc(SWEEP, "g%de83/oracle/q"), acc(SWEEP, "g%de83/oracle/pq")),
    ]

    fig, ax = plt.subplots(1, 2, figsize=(6.6, 1.55))
    for j, title in enumerate(
            ["(a) original question", "(b) paraphrased question"]):
        a = ax[j]
        for name, col, mk, ls, yq, ypq in regimes:
            y = yq if j == 0 else ypq
            a.plot(x, y, marker=mk, ms=4, lw=1.5, ls=ls, color=col,
                   label=name, zorder=3)
        a.axhline(base_q, color=C_BASE, lw=1.0, ls=":", zorder=1)
        a.text(0.0, base_q + 0.03, "no memory", color=C_BASE, va="bottom",
               ha="left", fontsize=6.5, zorder=4)
        a.set_xticks(x)
        a.set_xticklabels(["%d\n(%d ad.)" % (k, g[k]["n_adapters"])
                           for k in GS])
        # No axis label: it is identical in both panels and the caption already
        # says the x axis is facts per adapter over one fixed memory.
        a.set_ylim(0, 1.02)
        a.set_title(title, loc="left")
    ax[0].set_ylabel("free-form factual recall")
    # One shared legend above both panels. Inside either panel it collides with
    # a curve or the base-rate label, and the three regimes are the point of the
    # figure, so the key gets its own uncontested strip.
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.10), handlelength=2.4,
               columnspacing=1.6)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "granularity.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "granularity.png"), bbox_inches="tight", dpi=170)
    plt.close(fig)

    # ---- Figure 2: systems tradeoff (grouped bars, not a radar) ----------
    fig, ax = plt.subplots(1, 3, figsize=(6.6, 2.0))

    names = ["retrieval", "compiled\n20 f/ad.", "gradient\n20 f/ad."]
    cols = [C_RETR, C_ORIG, C_SGD]

    accs = [cells["retrieval/q"]["keyfacts_acc"], g[20]["oracle_q"],
            A["sgd"]["q"]["oracle_acc"]]
    ax[0].bar(names, accs, color=cols, width=0.62)
    ax[0].axhline(cells["base/q"]["keyfacts_acc"], color=C_BASE, lw=1.0, ls=":")
    ax[0].set_ylabel("recall (original q)")
    ax[0].set_ylim(0, 0.95)
    ax[0].set_title("(a) accuracy", loc="left")
    for i, v in enumerate(accs):
        ax[0].text(i, v + 0.02, "%.3f" % v, ha="center", fontsize=6.5)

    # recurring per-query prompt-token cost
    sysm = load(os.path.join(THREE, "ANALYSIS.json"))["systems"]["per_arm"]
    toks = [sysm["retrieval"]["mean_prompt_tokens_q"],
            sysm["base"]["mean_prompt_tokens_q"],
            sysm["base"]["mean_prompt_tokens_q"]]
    ax[1].bar(names, toks, color=cols, width=0.62)
    ax[1].set_ylabel("prompt tokens / query")
    ax[1].set_ylim(0, 320)
    ax[1].set_title("(b) recurring cost", loc="left")
    for i, v in enumerate(toks):
        ax[1].text(i, v + 8, "%.0f" % v, ha="center", fontsize=6.5)

    # one-time construction cost, log scale (spans 3 orders of magnitude)
    sgd_stats = load(os.path.join(MATCHED, "sgd_perauthor_matched_stats.json"))
    builds = [0.0, g[20]["total_build_seconds"], sgd_stats["total_train_seconds"]]
    plot_b = [b if b > 0 else 0.02 for b in builds]
    ax[2].bar(names, plot_b, color=cols, width=0.62)
    ax[2].set_yscale("log")
    ax[2].set_ylabel("construction (s, log)")
    ax[2].set_ylim(0.02, 1e4)
    ax[2].set_title("(c) one-time cost", loc="left")
    for i, v in enumerate(builds):
        lab = "none" if v == 0 else ("%.1f" % v if v < 100 else "%.0f" % v)
        ax[2].text(i, plot_b[i] * 1.5, lab, ha="center", fontsize=6.5)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "tradeoff.pdf"), bbox_inches="tight")
    plt.close(fig)

    print("wrote", os.path.join(OUT, "granularity.pdf"))
    print("wrote", os.path.join(OUT, "tradeoff.pdf"))


if __name__ == "__main__":
    main()
