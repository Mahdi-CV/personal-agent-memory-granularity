"""A deployable partition router, and the routing-accuracy measurement.

WHY THIS EXISTS

Every per-author result so far routes on the item's ground-truth author_idx.
That is an oracle: at inference time an agent does not know which partition of
the user's memory holds the answer, and handing the system that label gives it
information a deployed system would have to earn. So the partitioned-parametric
story is not testable until routing is done from the question alone.

DESIGN (deliberately the simplest fair thing)

The router scores the question against each partition's own factual document
using the SAME unigram word-overlap scorer the retrieval arm already uses
(tofu_freeform_v3.FreeFormEvaluator.retrieve). Reusing it matters twice:

  1. It is fair between arms. The identical router object routes both the SGD
     and the D2L per-author arms, so no routing advantage can accrue to either.
  2. It is fair against the retrieval BASELINE. Retrieval and routing then run
     on the same representation and the same matching rule, so a
     partitioned-parametric-vs-retrieval comparison is not secretly a
     comparison of two different retrievers.

No labels, no author_idx, no gold answers, no evaluation metadata are read. The
router sees exactly one thing: the question string, plus the partition
documents, which are built from the same facts every arm already receives.

SCORING RULE

Per partition, sum the per-fact overlap of the question's content words with
each fact in the partition, and also track the best single fact. Summing alone
would bias toward large partitions purely because they contain more facts,
which would make the router look better at coarse granularities for the wrong
reason. So the headline rule is MAX over facts (partition scored by its single
best-matching fact), which is size-invariant, and the sum rule is recorded
alongside as a sensitivity check.

STOPWORDS

The same STOP list the graders use is applied. Without it, question boilerplate
("what is the full name of the author") matches every partition equally and the
router degenerates toward the largest partition.
"""
import json
import os
import re
import sys

sys.path.insert(0, "/work/tofu_exp")
sys.path.insert(0, "/work/d2l_test")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tofu_freeform_v3 import STOP  # noqa: E402

WORD = re.compile(r"[a-z0-9]+")

# Facts per adapter for the granularity curve. 40 stands in for the requested
# 50 because TOFU author blocks are 20 facts wide; see build_partitions.
GRANULARITIES = (20, 40, 100, 200)


def words(s):
    return set(w for w in WORD.findall(s.lower()) if w not in STOP and len(w) > 1)


class PartitionRouter:
    """Route a question to one of N memory partitions, question-text only.

    partitions: list of dicts with keys
        name          served adapter suffix / identifier
        fact_indices  row indices of the facts this partition internalized
    facts: the same 200 answer strings every arm receives.
    """

    def __init__(self, partitions, facts, rule="max"):
        self.partitions = partitions
        self.facts = facts
        self.rule = rule
        self._pw = []
        for p in partitions:
            self._pw.append([words(facts[i]) for i in p["fact_indices"]])

    def scores(self, question):
        qw = words(question)
        out = []
        for fw in self._pw:
            per_fact = [len(qw & w) for w in fw]
            mx = max(per_fact) if per_fact else 0
            sm = sum(per_fact)
            out.append((mx, sm))
        return out

    def rank(self, question):
        """Partition indices, best first. Ties broken by the other statistic."""
        sc = self.scores(question)
        if self.rule == "max":
            key = lambda i: (-sc[i][0], -sc[i][1], i)  # noqa: E731
        else:
            key = lambda i: (-sc[i][1], -sc[i][0], i)  # noqa: E731
        return sorted(range(len(sc)), key=key)

    def route(self, question):
        return self.rank(question)[0]


def build_partitions(rows, facts, facts_per_adapter):
    """Group AUTHOR BLOCKS into partitions of ~facts_per_adapter facts.

    Partitioning is author-aware and deterministic: authors are taken in
    ascending author_idx order and packed into consecutive partitions, so every
    partition document is a whole number of coherent author blocks and no
    author is split across two adapters. That keeps each document readable
    prose (the input D2L was trained on) rather than an incoherent mix, and
    makes the partition set a pure function of facts_per_adapter with no
    randomness to seed or report.

    Returns a list of {name, author_ids, fact_indices}.
    """
    from collections import defaultdict
    by_author = defaultdict(list)
    for i, r in enumerate(rows):
        by_author[r["author_idx"]].append(i)
    authors = sorted(by_author)
    per_author = len(by_author[authors[0]])
    assert all(len(v) == per_author for v in by_author.values()), \
        "unequal author blocks; packing assumes uniform blocks"

    # TOFU's author blocks are 20 facts, so whole-author packing can only reach
    # partition sizes that are multiples of 20: 20/40/100/200 (10/5/2/1
    # adapters). A 50-fact partition would have to CUT an author in half,
    # putting the same author's facts in two different adapters. That would
    # break the coherent-document property D2L internalizes and would also make
    # routing artificially hard, because two partitions would then legitimately
    # contain the same author. The user asked for author-aware partitions where
    # they preserve coherent documents, so the requested 50 is replaced by 40
    # (the nearest reachable size) and the substitution is recorded here and in
    # the emitted stats rather than being silently rounded.
    assert facts_per_adapter % per_author == 0, \
        "facts_per_adapter %d is not a multiple of the %d-fact author block" % (
            facts_per_adapter, per_author)
    authors_per_part = facts_per_adapter // per_author

    parts = []
    for s in range(0, len(authors), authors_per_part):
        grp = authors[s:s + authors_per_part]
        idx = []
        for a in grp:
            idx.extend(by_author[a])
        parts.append({
            "name": "g%03d_p%02d" % (facts_per_adapter, len(parts)),
            "author_ids": grp,
            "fact_indices": idx,
        })
    tot = sum(len(p["fact_indices"]) for p in parts)
    assert tot == len(facts), (tot, len(facts))
    return parts


def gold_partition(parts, item_index):
    for pi, p in enumerate(parts):
        if item_index in p["fact_indices"]:
            return pi
    raise KeyError(item_index)


def main():
    DATA = "/work/tofu_exp/data_v3/ceiling"
    rows = [json.loads(l) for l in open(os.path.join(DATA, "test.jsonl"))]
    facts = json.load(open(os.path.join(DATA, "facts_small.json")))
    out = {"granularities": {}}

    # 40 replaces the requested 50: see build_partitions for why 50 is not
    # reachable without splitting a 20-fact author block across two adapters.
    for g in GRANULARITIES:
        parts = build_partitions(rows, facts, g)
        res = {"facts_per_adapter": g, "n_partitions": len(parts),
               "partitions": [{"name": p["name"], "author_ids": p["author_ids"],
                               "n_facts": len(p["fact_indices"])} for p in parts]}
        for rule in ("max", "sum"):
            r = PartitionRouter(parts, facts, rule=rule)
            for qkey in ("q", "pq"):
                t1 = t2 = 0
                pred = []
                for i, row in enumerate(rows):
                    rk = r.rank(row[qkey])
                    gold = gold_partition(parts, i)
                    pred.append(rk[0])
                    t1 += int(rk[0] == gold)
                    t2 += int(gold in rk[:2])
                res["%s/%s" % (rule, qkey)] = {
                    "top1": t1 / len(rows), "top2": t2 / len(rows),
                    "predicted": pred,
                }
                print("g=%3d n_part=%2d rule=%-3s %-2s top1=%.3f top2=%.3f"
                      % (g, len(parts), rule, qkey, t1 / len(rows),
                         t2 / len(rows)), flush=True)
        out["granularities"][str(g)] = res

    with open("/work/d2l_exp/router_accuracy.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote /work/d2l_exp/router_accuracy.json")


if __name__ == "__main__":
    main()
