#!/usr/bin/env python3
"""Generate every table and numeric macro in the paper from the frozen results.

    python3 paper2/make_tables.py

No number in the PDF is typed by hand. This script also re-asserts the claims
the paper makes, so a results file that no longer supports the text fails the
build instead of silently producing a wrong paper.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
GRAN = os.path.join(RES, "results_gran")
MATCHED = os.path.join(RES, "results_matched")
THREE = os.path.join(RES, "results_threeway")
REGIMES = os.path.join(RES, "results_regimes")
FULLCTX = os.path.join(RES, "results_full_context")
REANALYSIS = os.path.join(RES, "reanalysis")
TAB = os.path.join(HERE, "tables")

GS = [20, 40, 100, 200]

# LaTeX control sequences may not contain digits, so every generated macro name
# spells its granularity out.
WORD = {20: "Twenty", 40: "Forty", 100: "Hundred", 200: "TwoHundred"}


def load(p):
    with open(p) as f:
        return json.load(f)


def exact_mcnemar(b, c):
    """Two-sided exact McNemar on the discordant pairs only."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired_hits(h1, h2):
    """Paired comparison over items gradable in BOTH conditions.

    hit == -1 marks a generation the grader could not score, so the item is
    dropped PAIRWISE. This is the same rule 20_analyze_granularity.paired uses,
    reproduced here so the regime tables and the frozen ladders cannot drift
    apart on how ungradable items are handled.
    """
    lost = gained = both = neither = 0
    for a, b in zip(h1, h2):
        if a < 0 or b < 0:
            continue
        if a and not b:
            lost += 1
        elif b and not a:
            gained += 1
        elif a and b:
            both += 1
        else:
            neither += 1
    n = lost + gained + both + neither
    return {"n_paired": n, "lost": lost, "gained": gained, "both": both,
            "neither": neither,
            "retention": (both / (both + lost)) if (both + lost) else None,
            "p_mcnemar": exact_mcnemar(lost, gained)}


def fmt_p(p):
    """LaTeX p-value: plain below 4 digits, scientific with a tidy exponent."""
    if p >= 1e-3:
        return "%.3f" % p
    mant, exp = ("%.1e" % p).split("e")
    return "$%s\\times10^{%d}$" % (mant, int(exp))


def fmt_p_narrow(p):
    """Same value as fmt_p, in the narrowest honest form.

    Used only where six p-values share one table row; the mantissa-times-ten
    form is 40pt too wide there. Exponent-only because every one of these is a
    round mantissa away from its order of magnitude anyway, and the prose gives
    the full values.
    """
    if p >= 1e-3:
        return "%.2f" % p
    mant, exp = ("%.1e" % p).split("e")
    return "$%s\\mathrm{e}{%d}$" % (mant, int(exp))


def main():
    A = load(os.path.join(GRAN, "GRANULARITY_ANALYSIS.json"))
    B = load(os.path.join(GRAN, "gran_build_stats.json"))
    T = load(os.path.join(THREE, "threeway.json"))
    TA = load(os.path.join(THREE, "ANALYSIS.json"))
    gate = load(os.path.join(GRAN, "activity_gate_gr.json"))
    SGD = load(os.path.join(MATCHED, "sgd_perauthor_matched_stats.json"))
    MS = load(os.path.join(MATCHED, "matched_steps.json"))
    CR = load(os.path.join(REANALYSIS, "camera_ready_interaction.json"))

    g = {k: A["granularity"][str(k)] for k in GS}
    cells = T["cells"]

    # ---- integrity assertions: the paper's claims, re-checked -------------
    # 1. per-adapter capacity really is constant along the ladder
    ranks = {r for k in GS for r in g[k]["rank"]}
    assert ranks == {16}, ranks
    per_ad = {g[k]["per_adapter_bytes"] for k in GS}
    assert len(per_ad) == 1, per_ad
    # 2. every document was fully consumed (no silent truncation)
    assert all(g[k]["all_tokens_consumed"] for k in GS)
    # 3. total facts held fixed at 200 across the ladder
    for k in GS:
        tot = sum(a["n_facts"] for a in B["granularities"][str(k)]["adapters"].values())
        assert tot == 200, (k, tot)
        assert g[k]["n_adapters"] * k == 200, k
    # 4. the curve is monotone non-increasing on the headline metric
    accs = [g[k]["oracle_q"] for k in GS]
    assert all(accs[i] >= accs[i + 1] for i in range(len(accs) - 1)), accs
    # 5. every adapter passed the activity gate
    assert all(v["active"] for v in gate["adapters"].values()), "inert adapter"

    # The compiled ladder's own gate covers the 18 adapters of that ladder. The
    # operating-point arms and both gradient regimes were gated separately, so
    # the manuscript must not quote 18 as if it covered every adapter in the
    # paper. Take the union of every gate artifact and require all of them
    # active. Each gate records a base-versus-base noise floor; require that
    # floor to be present so "gated against a noise floor" is literally true.
    GATE_FILES = [
        os.path.join(GRAN, "activity_gate_gr.json"),
        os.path.join(GRAN, "activity_gate_sgd_pam.json"),
        os.path.join(MATCHED, "activity_gate_sgd_pam.json"),
        os.path.join(MATCHED, "activity_gate_23adapters.json"),
        os.path.join(THREE, "activity_gate.json"),
        os.path.join(REGIMES, "activity_gate_sgd_seeded249.json"),
        os.path.join(REGIMES, "activity_gate_sgd_gran.json"),
    ]
    gated = set()
    for gp in GATE_FILES:
        gj = load(gp)
        for name, rec in gj["adapters"].items():
            assert rec["active"], (gp, name)
            assert rec["delta"] > gj.get("noise_floor", 0.0), (gp, name)
            gated.add(name)

    # The manuscript says EVERY evaluated parametric adapter passed. That is only
    # honest if the two gradient regimes the results section reports are actually
    # covered. Check the exact adapter groups the eval used, by name, rather than
    # trusting a total count: fixed-update is sgds_g{k}_s249_*, equal-exposure is
    # sgdgr_g{k}_e83_*, and each must contribute 200/k adapters.
    for tmpl in ("sgds_g%d_s249_", "sgdgr_g%d_e83_"):
        for k in GS:
            pre = tmpl % k
            n = len([a for a in gated if a.startswith(pre)])
            assert n == 200 // k, (pre, n, 200 // k)

    n_gate = len(gate["adapters"])
    n_gate_all = len(gated)

    # ---- macros -----------------------------------------------------------
    m = {}
    m["NPerArm"] = T["n_trained_items"]
    m["NGradable"] = cells["base/q"]["keyfacts_n_gradable"]
    m["NExcluded"] = int(m["NPerArm"]) - int(m["NGradable"])
    m["NAdaptersGated"] = n_gate
    m["NAdaptersGatedAll"] = n_gate_all
    m["GraderFloor"] = "%.4f" % T["grader_floor"]

    for k in GS:
        tag = "G" + WORD[k]
        m[tag + "Q"] = "%.3f" % g[k]["oracle_q"]
        m[tag + "PQ"] = "%.3f" % g[k]["oracle_pq"]
        m[tag + "Nad"] = g[k]["n_adapters"]
        m[tag + "TotMB"] = "%.1f" % (g[k]["total_memory_bytes"] / 1e6)
        m[tag + "Build"] = "%.2f" % g[k]["total_build_seconds"]
        m[tag + "Ret"] = "%.3f" % g[k]["paraphrase_paired_oracle"]["retention"]

    m["ResidentMB"] = "%.1f" % (g[20]["resident_bytes_one_active"] / 1e6)
    m["StorageRatio"] = "%.0f" % (g[20]["total_memory_bytes"]
                                  / g[200]["total_memory_bytes"])

    cp = A["curve_paired"]["g20_vs_g200/q"]
    m["CurveLost"] = cp["lost"]
    m["CurveGained"] = cp["gained"]
    m["CurveP"] = fmt_p(cp["p_mcnemar"]).strip("$")
    m["CurveNPaired"] = cp["n_paired"]

    for a, b in ((20, 40), (40, 100), (100, 200)):
        c = A["curve_paired"]["g%d_vs_g%d/q" % (a, b)]
        m["Step%sto%sLost" % (WORD[a], WORD[b])] = c["lost"]
        m["Step%sto%sGained" % (WORD[a], WORD[b])] = c["gained"]
        m["Step%sto%sP" % (WORD[a], WORD[b])] = fmt_p(c["p_mcnemar"]).strip("$")

    m["BaseQ"] = "%.3f" % cells["base/q"]["keyfacts_acc"]
    m["BasePQ"] = "%.3f" % cells["base/pq"]["keyfacts_acc"]
    m["RetrQ"] = "%.3f" % cells["retrieval/q"]["keyfacts_acc"]
    m["RetrPQ"] = "%.3f" % cells["retrieval/pq"]["keyfacts_acc"]
    m["SgdQ"] = "%.3f" % A["sgd"]["q"]["oracle_acc"]
    m["SgdPQ"] = "%.3f" % A["sgd"]["pq"]["oracle_acc"]
    m["SgdRet"] = "%.3f" % A["sgd"]["paraphrase_paired_oracle"]["retention"]

    # Paraphrase robustness is reported WITHIN each arm (does this arm lose
    # significantly when the question is reworded?) and not as a ranking between
    # arms. The between-arm contrasts are not resolvable at this n: on a
    # common-correct paired test they give p = 0.31 (compiled vs gradient),
    # 0.50 (compiled vs retrieval), 0.71 (retrieval vs gradient).
    # Stripped of "$" so these compose inside an existing math group, matching
    # the Step*P macros above.
    m["SgdRetP"] = fmt_p(A["sgd"]["paraphrase_paired_oracle"]["p_mcnemar"]).strip("$")
    m["GTwentyRetP"] = fmt_p(g[20]["paraphrase_paired_oracle"]["p_mcnemar"]).strip("$")

    # Retrieval retention on the SAME definition the rest of the paper uses,
    # both/(both+lost), recomputed from the per-item hits. COMMON_CORRECT.json
    # stores a different quantity (pq/q) under the same name, so it cannot be
    # read from there without changing the definition mid-paper.
    rq, rpq = cells["retrieval/q"]["hits"]["keyfacts"], cells["retrieval/pq"]["hits"]["keyfacts"]
    r_both = sum(1 for a, b in zip(rq, rpq) if a == 1 and b == 1)
    r_lost = sum(1 for a, b in zip(rq, rpq) if a == 1 and b == 0)
    r_gain = sum(1 for a, b in zip(rq, rpq) if a == 0 and b == 1)
    m["RetrRet"] = "%.3f" % (r_both / (r_both + r_lost))
    m["RetrRetP"] = fmt_p(exact_mcnemar(r_lost, r_gain)).strip("$")

    # The paper states that no PAIRWISE retention difference is resolvable, so
    # that statement is checked here rather than trusted. Restrict to items both
    # arms answered correctly on the original question, then test which arm
    # still survives the paraphrase. If any pair ever became significant the
    # text would be wrong and the build must fail.
    sgd_q = MS["cells"]["sgd_pam/q"]["hits"]["keyfacts"]
    sgd_pq = MS["cells"]["sgd_pam/pq"]["hits"]["keyfacts"]
    d_q = cells["d2l_pa/q"]["hits"]["keyfacts"]
    d_pq = cells["d2l_pa/pq"]["hits"]["keyfacts"]

    def common_correct_p(qa, pa, qb, pb):
        b = c = 0
        for i in range(len(qa)):
            if -1 in (qa[i], pa[i], qb[i], pb[i]) or not (qa[i] and qb[i]):
                continue
            if pa[i] and not pb[i]:
                b += 1
            elif pb[i] and not pa[i]:
                c += 1
        return exact_mcnemar(b, c)

    pairwise = {
        "compiled-vs-gradient": common_correct_p(d_q, d_pq, sgd_q, sgd_pq),
        "compiled-vs-retrieval": common_correct_p(d_q, d_pq, rq, rpq),
        "retrieval-vs-gradient": common_correct_p(rq, rpq, sgd_q, sgd_pq),
    }
    assert all(p > 0.05 for p in pairwise.values()), pairwise
    m["PairMinP"] = "%.2f" % min(pairwise.values())

    # Printed in the manuscript so the reproducibility answer does not depend on
    # a future code release. LR is not stored in the stats file; it is the
    # build script's default and is asserted against it at build time.
    m["SgdOptSteps"] = SGD["optimizer_steps_per_adapter"]
    m["SgdEpochs"] = SGD["epochs"]
    m["LoraAlpha"] = 32
    m["LearningRate"] = "2\\times10^{-4}"
    lr_src = os.path.join(ROOT, "connector", "d2l",
                          "15_build_sgd_perauthor_matched.py")
    if os.path.exists(lr_src):
        src = open(lr_src).read()
        assert '"--lr", type=float, default=2e-4' in src, "LR default changed"
        assert '"--alpha", type=int, default=32' in src, "alpha default changed"
    assert SGD["match_criterion"].startswith("optimizer steps"), SGD["match_criterion"]
    m["SgdFullQ"] = "%.3f" % cells["sgd7/q"]["keyfacts_acc"]
    m["SgdFullPQ"] = "%.3f" % cells["sgd7/pq"]["keyfacts_acc"]

    # near-domain (unrelated-question) control
    m["NearBase"] = "%.3f" % cells["base/near"]["keyfacts_acc"]
    m["NearGTwenty"] = "%.3f" % cells["d2l_pa/near"]["keyfacts_acc"]
    m["NearRetr"] = "%.3f" % cells["retrieval/near"]["keyfacts_acc"]

    # deployment cost
    sysm = TA["systems"]["per_arm"]
    ptr = sysm["retrieval"]["mean_prompt_tokens_q"] / sysm["base"]["mean_prompt_tokens_q"]
    m["PromptTokenRatio"] = "%.2f" % ptr
    m["RetrTokens"] = "%.1f" % sysm["retrieval"]["mean_prompt_tokens_q"]
    m["ParamTokens"] = "%.1f" % sysm["base"]["mean_prompt_tokens_q"]
    m["ExtraTokens"] = "%.1f" % (sysm["retrieval"]["mean_prompt_tokens_q"]
                                 - sysm["base"]["mean_prompt_tokens_q"])

    # Construction cost, steady state per adapter. The ladder is warmed before
    # the first timed build (see the build script), so no cold-compile outlier
    # has to be excluded by hand: every timing here is steady state.
    sgd_total = SGD["total_train_seconds"]
    d2l_per = g[20]["per_adapter_build_seconds"]
    sgd_per = sgd_total / g[20]["n_adapters"]
    # The gradient arm must cover the same bank the compiled arm does, or the
    # per-adapter ratio below is not like-for-like.
    assert SGD["n_adapters"] == g[20]["n_adapters"], (
        SGD["n_adapters"], g[20]["n_adapters"])
    m["SgdBuildTotal"] = "%.1f" % sgd_total
    m["SgdBuildPer"] = "%.1f" % sgd_per
    m["DtwoLBuildPer"] = "%.3f" % d2l_per
    m["BuildRatio"] = "%.1f" % (sgd_per / d2l_per)

    # routing
    m["RouterTopOne"] = "%.3f" % (100 * 0 + g[20]["router_top1_q"])
    m["RouterMisGTwenty"] = A["routing_tax"]["g20/q"]["n_misrouted"]
    m["MaxRoutingDelta"] = "%.3f" % max(
        abs(v["delta"]) for v in A["routing_tax"].values() if v["delta"] is not None)
    m["NondetChanged"] = A["routing_tax"]["g200/q"]["n_items_changed"]
    # At the coarsest granularity routing is provably identical for every item,
    # so any generation that still differs between two runs is pure engine
    # nondeterminism. Most of it is absorbed by grading; a little is not.
    nd = A["engine_nondeterminism"]["per_cell"]["g200/q"]
    assert nd["n_identically_routed"] == int(m["NPerArm"]), nd
    m["NondetGens"] = nd["gens_differing_despite_identical_routing"]
    m["NondetRouted"] = nd["n_identically_routed"]
    assert nd["graded_flips_despite_identical_routing"] == m["NondetChanged"], nd

    # ---- Three construction regimes over the same 200 facts ---------------
    # Compiled, gradient at a fixed optimizer-update budget, and gradient at
    # fixed per-fact exposure. These answer different questions and are NOT
    # three estimates of one curve, so each is emitted with its own macros and
    # its own paired tests rather than being averaged or overlaid as one line.
    SEED = load(os.path.join(REGIMES, "granularity_seeded249.json"))["cells"]
    SWEEP = load(os.path.join(REGIMES, "granularity_sgd_sweep.json"))["cells"]
    SB = load(os.path.join(REGIMES, "sgd_seeded249_build_stats.json"))
    WB = load(os.path.join(REGIMES, "sgd_gran_build_stats.json"))

    def seed_key(k, qk):
        return "g%ds249/oracle/%s" % (k, qk)

    def e83_key(k, qk):
        return "g%de83/oracle/%s" % (k, qk)

    # The seeded ladder exists to remove an initialization confound, so the
    # build must prove the initialization really was uniform. If a future
    # rebuild loses that property the paper's control claim is void and this
    # assertion fails the build rather than quietly weakening the text.
    assert SB["init_uniformity"]["uniform"] is True, SB["init_uniformity"]
    assert SB["init_uniformity"]["n_distinct_lora_A_first_draws"] == 1
    assert SB["exact_optimizer_steps"] == 249, SB["exact_optimizer_steps"]
    assert SB["seed"] == 42, SB["seed"]
    assert SB["target_modules"] == ["down_proj"], SB["target_modules"]
    assert SB["rank"] == 16 and SB["alpha"] == 32 and SB["grad_accum"] == 8
    # Every adapter must have stopped at exactly 249 optimizer updates, read
    # back from the Trainer rather than assumed from the configuration.
    for cname, cond in SB["conditions"].items():
        assert set(cond["verified_global_step"]) == {249}, (cname, cond)
    assert SB["n_adapters_total"] == 18, SB["n_adapters_total"]

    m["SeedSteps"] = SB["exact_optimizer_steps"]
    m["SeedSeed"] = SB["seed"]
    m["SeedNAd"] = SB["n_adapters_total"]
    m["ExpEpochs"] = 83

    for k in GS:
        tag = "G" + WORD[k]
        m["Seed" + tag + "Q"] = "%.3f" % SEED[seed_key(k, "q")]["keyfacts_acc"]
        m["Seed" + tag + "PQ"] = "%.3f" % SEED[seed_key(k, "pq")]["keyfacts_acc"]
        m["Exp" + tag + "Q"] = "%.3f" % SWEEP[e83_key(k, "q")]["keyfacts_acc"]
        m["Exp" + tag + "PQ"] = "%.3f" % SWEEP[e83_key(k, "pq")]["keyfacts_acc"]
        # Steps per adapter at 83 epochs grow with partition size; this is the
        # quantity the fixed-step ladder holds constant and this one does not.
        m["Exp" + tag + "Steps"] = WB["conditions"]["g%d/e83" % k][
            "optimizer_steps_per_adapter"]

    # Paired g20 -> g200 tests, one per regime and question type.
    def reg_paired(cells, keyfn, qk):
        return paired_hits(cells[keyfn(20, qk)]["hits"]["keyfacts"],
                           cells[keyfn(200, qk)]["hits"]["keyfacts"])

    seed_q = reg_paired(SEED, seed_key, "q")
    seed_pq = reg_paired(SEED, seed_key, "pq")
    exp_q = reg_paired(SWEEP, e83_key, "q")
    exp_pq = reg_paired(SWEEP, e83_key, "pq")
    # The compiled paraphrase test is not in GRANULARITY_ANALYSIS (which pairs
    # original questions only), so it is computed here from the same per-item
    # hits through the same rule.
    COMP = load(os.path.join(GRAN, "granularity.json"))["cells"]
    comp_pq = paired_hits(COMP["g20/oracle/pq"]["hits"]["keyfacts"],
                          COMP["g200/oracle/pq"]["hits"]["keyfacts"])
    m["CompCurvePQP"] = fmt_p(comp_pq["p_mcnemar"]).strip("$")

    m["SeedCurveP"] = fmt_p(seed_q["p_mcnemar"]).strip("$")
    m["SeedCurveLost"] = seed_q["lost"]
    m["SeedCurveGained"] = seed_q["gained"]
    m["SeedCurvePQP"] = fmt_p(seed_pq["p_mcnemar"]).strip("$")
    m["ExpCurveP"] = fmt_p(exp_q["p_mcnemar"]).strip("$")
    m["ExpCurveLost"] = exp_q["lost"]
    m["ExpCurveGained"] = exp_q["gained"]
    m["ExpCurvePQP"] = fmt_p(exp_pq["p_mcnemar"]).strip("$")
    m["ExpCurvePQLost"] = exp_pq["lost"]
    m["ExpCurvePQGained"] = exp_pq["gained"]

    # The equal-exposure original-question result is a NON-significant
    # comparison, so the text must not lean on the single largest p-value.
    # Emit the full range across all three coarsenings from g20.
    exp_q_ps = [paired_hits(SWEEP[e83_key(20, "q")]["hits"]["keyfacts"],
                            SWEEP[e83_key(k, "q")]["hits"]["keyfacts"])["p_mcnemar"]
                for k in (40, 100, 200)]
    m["ExpQPMin"] = "%.2f" % min(exp_q_ps)
    exp_q_accs = [SWEEP[e83_key(k, "q")]["keyfacts_acc"] for k in GS]
    m["ExpQLo"] = "%.3f" % min(exp_q_accs)
    m["ExpQHi"] = "%.3f" % max(exp_q_accs)

    # Direct question-form x granularity interaction requested during review.
    # Positive means the g20->g200 penalty is larger for paraphrased questions.
    eq_int = CR["equal_83_epochs"]
    assert eq_int["n_items_gradable_in_all_four_cells"] == int(m["NGradable"])
    assert eq_int["interaction"] > 0
    assert eq_int["author_clustered_bootstrap_ci95"][0] > 0
    m["ExpInteraction"] = "%.3f" % eq_int["interaction"]
    m["ExpInteractionCILo"] = "%.3f" % eq_int[
        "author_clustered_bootstrap_ci95"][0]
    m["ExpInteractionCIHi"] = "%.3f" % eq_int[
        "author_clustered_bootstrap_ci95"][1]
    m["ExpInteractionP"] = "%.4f" % eq_int["author_exact_sign_flip_p"]
    m["NAuthorClusters"] = eq_int["n_author_clusters"]

    endpoint = CR["endpoint_sensitivity"]
    m["CompAuthorQP"] = "%.3f" % endpoint[
        "compiled_original"]["author_exact_sign_flip_p"]
    m["SeedAuthorQP"] = "%.3f" % endpoint[
        "fixed_249_original"]["author_exact_sign_flip_p"]
    m["ExpAuthorQP"] = "%.3f" % endpoint[
        "equal_83_original"]["author_exact_sign_flip_p"]
    m["ExpAuthorPQP"] = "%.3f" % endpoint[
        "equal_83_paraphrased"]["author_exact_sign_flip_p"]

    # Bank construction times for the two gradient resource protocols.
    for k in GS:
        tag = WORD[k]
        m["SeedBuild" + tag] = "%.1f" % SB["conditions"]["g%d" % k][
            "total_build_seconds"]
        m["ExpBuild" + tag] = "%.1f" % WB["conditions"]["g%d/e83" % k][
            "total_build_seconds"]

    # The paper's central contrast: the fixed-step regime resolves a decline on
    # original questions and the equal-exposure regime does not, while BOTH
    # resolve one on paraphrases. If that ever stops holding, the interpretation
    # in the abstract and conclusion is wrong and the build must fail.
    assert seed_q["p_mcnemar"] < 0.001, seed_q
    assert min(exp_q_ps) > 0.05, exp_q_ps
    assert exp_pq["p_mcnemar"] < 0.001, exp_pq
    assert seed_pq["p_mcnemar"] < 0.01, seed_pq
    # Compiled must remain the steepest of the three on original questions.
    comp_drop = g[20]["oracle_q"] - g[200]["oracle_q"]
    seed_drop = (SEED[seed_key(20, "q")]["keyfacts_acc"]
                 - SEED[seed_key(200, "q")]["keyfacts_acc"])
    exp_drop = (SWEEP[e83_key(20, "q")]["keyfacts_acc"]
                - SWEEP[e83_key(200, "q")]["keyfacts_acc"])
    assert comp_drop > seed_drop > exp_drop, (comp_drop, seed_drop, exp_drop)

    # Initialization control, reported as one sentence in the main body.
    INIT = load(os.path.join(REGIMES, "init_sensitivity.json"))
    m["InitN"] = INIT["paired"]["q"]["n_paired"]
    m["InitQP"] = "%.2f" % INIT["paired"]["q"]["p_mcnemar"]
    m["InitPQP"] = "%.2f" % INIT["paired"]["pq"]["p_mcnemar"]

    # ---- Full context: all 200 facts in every prompt ----------------------
    # This is NOT retrieval: there is no ranking, filtering or top-k step, so
    # it is the no-selection reference point rather than a retrieval arm. The
    # assertions below are what let the manuscript state "no truncation" as a
    # measured fact rather than an assumption.
    FC = load(os.path.join(FULLCTX, "full_context.json"))
    fc_q = FC["cells"]["full_context/q"]
    fc_pq = FC["cells"]["full_context/pq"]

    assert FC["context_construction"]["n_facts"] == 200, FC["context_construction"]
    assert FC["context_construction"]["selection"].startswith("none")
    # Every fact must have been present in the prompt actually submitted, at
    # the start, the middle and the end of the context.
    for where, rec in FC["truncation_audit"].items():
        assert rec["present_in_q_prompt"] and rec["present_in_pq_prompt"], (where, rec)
    # The server had to be able to hold the longest prompt plus the generation
    # budget, or a silent truncation would be indistinguishable from a result.
    longest = max(fc_q["max_prompt_tokens"], fc_pq["max_prompt_tokens"])
    max_new = FC["generation"]["max_tokens"]
    assert longest + max_new < FC["server_max_model_len"], (longest, max_new)
    # vLLM's own prompt accounting must agree with the local tokenizer, which
    # is an independent check that nothing was dropped server-side.
    assert abs(fc_q["server_reported_prompt_tokens_mean"]
               - fc_q["mean_prompt_tokens"]) < 0.5, fc_q
    # Same grader and threshold as every other arm.
    assert abs(FC["grader_floor"] - T["grader_floor"]) < 1e-9, FC["grader_floor"]
    assert FC["generation"]["temperature"] == 0.0, FC["generation"]
    assert FC["base_model"].startswith("Qwen3-4B-Instruct"), FC["base_model"]

    m["FullCtxQ"] = "%.3f" % fc_q["keyfacts_acc"]
    m["FullCtxPQ"] = "%.3f" % fc_pq["keyfacts_acc"]
    m["FullCtxNHit"] = fc_q["keyfacts_n_hit"]
    m["FullCtxNGrad"] = fc_q["keyfacts_n_gradable"]
    m["FullCtxQCILo"] = "%.3f" % fc_q["keyfacts_ci95"][0]
    m["FullCtxQCIHi"] = "%.3f" % fc_q["keyfacts_ci95"][1]
    m["FullCtxPQNHit"] = fc_pq["keyfacts_n_hit"]
    m["FullCtxTokens"] = "{:,.1f}".format(fc_q["mean_prompt_tokens"])
    m["FullCtxNFacts"] = FC["context_construction"]["n_facts"]

    # Paired within-arm paraphrase test, recomputed here from the per-item hits
    # through the same rule the rest of the paper uses rather than trusted.
    fcp = paired_hits(fc_q["hits"]["keyfacts"], fc_pq["hits"]["keyfacts"])
    assert fcp["lost"] == FC["paired_q_vs_pq"]["lost"], (fcp, FC["paired_q_vs_pq"])
    assert fcp["gained"] == FC["paired_q_vs_pq"]["gained"], fcp
    assert abs(fcp["p_mcnemar"] - FC["paired_q_vs_pq"]["p_mcnemar"]) < 1e-9, fcp
    m["FullCtxLost"] = fcp["lost"]
    m["FullCtxGained"] = fcp["gained"]
    m["FullCtxRet"] = "%.3f" % fcp["retention"]
    m["FullCtxRetP"] = "%.3f" % fcp["p_mcnemar"]
    # This arm must be the one place the paper says the paraphrase drop is
    # unresolved, so a future rebuild that resolves it must fail the build.
    assert fcp["p_mcnemar"] > 0.05, fcp

    # Prompt-token ratios. Measured token counts only; the manuscript must not
    # turn these into latency or throughput ratios.
    fc_ratio_param = fc_q["mean_prompt_tokens"] / sysm["base"]["mean_prompt_tokens_q"]
    fc_ratio_retr = (fc_q["mean_prompt_tokens"]
                     / sysm["retrieval"]["mean_prompt_tokens_q"])
    m["FullCtxRatio"] = "%.1f" % fc_ratio_param
    m["FullCtxRatioRetr"] = "%.1f" % fc_ratio_retr
    # Full context must remain the highest-accuracy observed operating point,
    # since the abstract and conclusion both say so.
    assert fc_q["keyfacts_acc"] > cells["retrieval/q"]["keyfacts_acc"], fc_q
    assert fc_q["keyfacts_acc"] > A["sgd"]["q"]["oracle_acc"], fc_q

    with open(os.path.join(TAB, "macros.tex"), "w") as f:
        f.write("% Generated by paper2/make_tables.py. Do not edit by hand.\n")
        for k, v in m.items():
            f.write("\\newcommand{\\%s}{%s}\n" % (k, v))

    # ---- Table: the granularity curve ------------------------------------
    # The ladder and the paired step tests are one table. The step tests attach
    # as two extra columns on the row they start from, which keeps the table to
    # one row per granularity instead of interleaving step rows.
    with open(os.path.join(TAB, "granularity.tex"), "w") as f:
        f.write("\\setlength{\\tabcolsep}{4.5pt}\n")
        f.write("\\begin{tabular}{rrrrrrrcrr}\n\\toprule\n")
        f.write("& & \\multicolumn{5}{c}{memory} & "
                "\\multicolumn{2}{c}{paired step to next row} \\\\\n")
        f.write("\\cmidrule(lr){3-7}\\cmidrule(lr){9-10}\n")
        f.write("facts/ad. & ad. & orig.\\ q & para.\\ q & reten. & total MB"
                " & build s & & lost/gained & $p$ \\\\\n\\midrule\n")
        for i, k in enumerate(GS):
            r = g[k]
            if i + 1 < len(GS):
                c = A["curve_paired"]["g%d_vs_g%d/q" % (k, GS[i + 1])]
                step = "%d\\,/\\,%d & %s" % (
                    c["lost"], c["gained"], fmt_p(c["p_mcnemar"]))
            else:
                step = "-- & --"
            f.write("%d & %d & %.3f & %.3f & %.3f & %.1f & %.2f & & %s \\\\\n" % (
                k, r["n_adapters"], r["oracle_q"], r["oracle_pq"],
                r["paraphrase_paired_oracle"]["retention"],
                r["total_memory_bytes"] / 1e6, r["total_build_seconds"], step))
        c = A["curve_paired"]["g%d_vs_g%d/q" % (GS[0], GS[-1])]
        f.write("\\midrule\n")
        f.write("\\multicolumn{8}{r}{\\itshape end to end %d$\\to$%d} & "
                "%d\\,/\\,%d & %s \\\\\n" % (
                    GS[0], GS[-1], c["lost"], c["gained"],
                    fmt_p(c["p_mcnemar"])))
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ---- Table: paired steps along the curve ------------------------------
    # n_paired is constant across every row, so it is stated in the caption
    # rather than repeated as a column that would widen the table.
    with open(os.path.join(TAB, "curve_paired.tex"), "w") as f:
        f.write("\\setlength{\\tabcolsep}{4pt}\n")
        f.write("\\begin{tabular}{lrrl}\n\\toprule\n")
        f.write("comparison & lost & gained & $p$ \\\\\n\\midrule\n")
        order = [("20", "40"), ("40", "100"), ("100", "200"), ("20", "200")]
        npair = set()
        for a, b in order:
            c = A["curve_paired"]["g%s_vs_g%s/q" % (a, b)]
            npair.add(c["n_paired"])
            f.write("%s $\\to$ %s & %d & %d & %s \\\\\n" % (
                a, b, c["lost"], c["gained"], fmt_p(c["p_mcnemar"])))
        assert len(npair) == 1, npair
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ---- Table: operating points -----------------------------------------
    # The compiled row at 200 f/ad. is omitted here: it is the last row of the
    # granularity table and repeating it would add height without information.
    # Prompt tokens are shown as a ratio to the parametric prompt, not as both
    # ratio and absolute, so the full-context row fits without shrinking type;
    # the absolute full-context count is given in prose in the results section.
    # The gradient row is the frozen step-matched operating-point experiment
    # (SgdQ), a DIFFERENT experiment from the seeded granularity ladder, so its
    # value must not be swapped for the ladder's g20 number.
    assert abs(A["sgd"]["q"]["oracle_acc"] - 0.779) < 5e-4, A["sgd"]["q"]
    rows = [
        ("Base (no memory)", cells["base/q"]["keyfacts_acc"],
         cells["base/pq"]["keyfacts_acc"], "1$\\times$", "0", "0"),
        ("Full context (all \\NPerArm{} facts)", fc_q["keyfacts_acc"],
         fc_pq["keyfacts_acc"], "%.1f$\\times$" % fc_ratio_param, "0", "0"),
        ("Top-5 retrieval", cells["retrieval/q"]["keyfacts_acc"],
         cells["retrieval/pq"]["keyfacts_acc"], "%.2f$\\times$" % ptr, "0", "0"),
        ("Compiled, 20 f/ad.", g[20]["oracle_q"], g[20]["oracle_pq"],
         "1$\\times$", "%.1f" % (g[20]["total_memory_bytes"] / 1e6),
         "%.2f" % g[20]["total_build_seconds"]),
        ("Gradient, 20 f/ad.", A["sgd"]["q"]["oracle_acc"],
         A["sgd"]["pq"]["oracle_acc"], "1$\\times$",
         "%.1f" % (g[20]["total_memory_bytes"] / 1e6), "%.1f" % sgd_total),
    ]
    with open(os.path.join(TAB, "operating.tex"), "w") as f:
        f.write("\\setlength{\\tabcolsep}{4.5pt}\n")
        f.write("\\begin{tabular}{lrrrrr}\n\\toprule\n")
        # "bank", not "adapter": the compiled and gradient rows carry the total
        # over the ten-adapter g20 bank, not a per-adapter figure. Per-adapter
        # build seconds are given in the prose instead.
        f.write("memory organization & orig.\\ q & para.\\ q & prompt tok."
                " & bank MB & bank build s \\\\\n\\midrule\n")
        for name, q, pq, tok, mb, bs in rows:
            f.write("%s & %.3f & %.3f & %s & %s & %s \\\\\n"
                    % (name, q, pq, tok, mb, bs))
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ---- Table: the three construction regimes ----------------------------
    # One row per granularity, three column blocks for the three regimes. The
    # steps column is repeated only for the equal-exposure block because that
    # is the block where it varies; the other two hold it fixed by design.
    # This table absorbs the storage and compiled-build columns that used to sit
    # in a separate compiled-only ladder table, so the two are not printed twice.
    with open(os.path.join(TAB, "regimes.tex"), "w") as f:
        f.write("\\setlength{\\tabcolsep}{4pt}\n")
        f.write("\\begin{tabular}{rrrrrrrrrrr}\n\\toprule\n")
        f.write("& & \\multicolumn{2}{c}{compiled} & "
                "\\multicolumn{2}{c}{gradient, %d updates} & "
                "\\multicolumn{3}{c}{gradient, %d epochs} & "
                "\\multicolumn{2}{c}{bank} \\\\\n"
                % (SB["exact_optimizer_steps"], 83))
        f.write("\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-9}"
                "\\cmidrule(lr){10-11}\n")
        # The build-seconds column is the COMPILED generator's construction time
        # only. Labelling it plain "build s" under a shared "bank" header would
        # read as if it covered all three regimes; it does not.
        f.write("facts/ad. & ad. & orig. & para. & orig. & para."
                " & orig. & para. & upd. & MB & comp.\\ build s"
                " \\\\\n\\midrule\n")
        for k in GS:
            f.write("%d & %d & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %d"
                    " & %.1f & %.2f \\\\\n"
                    % (k, g[k]["n_adapters"],
                       g[k]["oracle_q"], g[k]["oracle_pq"],
                       SEED[seed_key(k, "q")]["keyfacts_acc"],
                       SEED[seed_key(k, "pq")]["keyfacts_acc"],
                       SWEEP[e83_key(k, "q")]["keyfacts_acc"],
                       SWEEP[e83_key(k, "pq")]["keyfacts_acc"],
                       WB["conditions"]["g%d/e83" % k][
                           "optimizer_steps_per_adapter"],
                       g[k]["total_memory_bytes"] / 1e6,
                       g[k]["total_build_seconds"]))
        f.write("\\midrule\n")
        # One row, not two: each regime block already has an orig. and a para.
        # column, so the end-to-end p-values sit under the accuracies they test.
        cp20 = A["curve_paired"]["g20_vs_g200/q"]
        f.write("\\multicolumn{2}{r}{\\itshape $20\\to200$, $p$} & "
                "%s & %s & %s & %s & %s & %s & & & \\\\\n"
                % (fmt_p_narrow(cp20["p_mcnemar"]),
                   fmt_p_narrow(comp_pq["p_mcnemar"]),
                   fmt_p_narrow(seed_q["p_mcnemar"]),
                   fmt_p_narrow(seed_pq["p_mcnemar"]),
                   fmt_p_narrow(exp_q["p_mcnemar"]),
                   fmt_p_narrow(exp_pq["p_mcnemar"])))
        f.write("\\bottomrule\n\\end{tabular}\n")

    print("wrote macros + 4 tables to", TAB)
    print("  compiled q:  ", [round(g[k]["oracle_q"], 3) for k in GS])
    print("  seeded249 q: ", [round(SEED[seed_key(k, "q")]["keyfacts_acc"], 3)
                              for k in GS])
    print("  e83 q:       ", [round(SWEEP[e83_key(k, "q")]["keyfacts_acc"], 3)
                              for k in GS])
    print("  e83 q p-values vs g20 (40/100/200):",
          ["%.4g" % p for p in exp_q_ps])
    print("  seeded q p(20->200) = %.4g   e83 pq p(20->200) = %.4g"
          % (seed_q["p_mcnemar"], exp_pq["p_mcnemar"]))
    print("  curve q:", [g[k]["oracle_q"] for k in GS])
    print("  full ctx: q=%.3f pq=%.3f  lost=%d gained=%d p=%.4f  %.1f tok "
          "(%.1fx param, %.1fx top-5)"
          % (fc_q["keyfacts_acc"], fc_pq["keyfacts_acc"], fcp["lost"],
             fcp["gained"], fcp["p_mcnemar"], fc_q["mean_prompt_tokens"],
             fc_ratio_param, fc_ratio_retr))
    print("  all assertions passed (rank fixed %s, 200 facts, %d ladder / %d "
          "total adapters gated)" % (ranks, n_gate, n_gate_all))


if __name__ == "__main__":
    sys.exit(main())
