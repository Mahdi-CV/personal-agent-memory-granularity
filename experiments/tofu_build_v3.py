"""Rebuild the TOFU experiment data with a proper held-out split.

Produces two evaluation conditions:

  heldout  - train on authors 0-9, evaluate on authors 10-19. The eval facts were
             never seen during training, so this measures whether an adapter can
             answer about facts it was not trained on.
  ceiling  - train and evaluate on authors 0-9 (the original setup). Every eval
             question is a training question, so adapter accuracy on the verbatim
             target is a memorization ceiling, not a generalization measure.

The retrieval corpus is padded with distractors drawn from the remaining ~180 TOFU
authors so that retrieval is searching a realistic corpus rather than an oracle
set containing only the gold answers.
"""
import argparse
import json
import os

from datasets import load_dataset

QA_PER_AUTHOR = 20
N_PERTURBED = 200  # authors available in forget10_perturbed / QA_PER_AUTHOR -> 20


def author_rows(ds, author_idx):
    lo = author_idx * QA_PER_AUTHOR
    return [ds[i] for i in range(lo, lo + QA_PER_AUTHOR)]


def build_condition(pt, train_authors, eval_authors):
    train_records, train_answers = [], []
    for a in train_authors:
        for row in author_rows(pt, a):
            train_records.append({"messages": [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]},
            ]})
            train_answers.append(row["answer"])

    test_items = []
    for a in eval_authors:
        for row in author_rows(pt, a):
            test_items.append({
                "author_idx": a,
                "q": row["question"],
                "pq": row["paraphrased_question"],
                "true": row["answer"],
                "para_true": row["paraphrased_answer"],
                "perturbed": list(row["perturbed_answer"]),
            })
    return train_records, train_answers, test_items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.environ.get("TOFU_DATA_DIR", "./data_v3"))
    ap.add_argument("--n-distractors", type=int, default=3800,
                    help="extra fact statements added to the retrieval corpus")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pt = load_dataset("locuslab/TOFU", "forget10_perturbed")["train"]
    full = load_dataset("locuslab/TOFU", "full")["train"]
    n_authors = len(pt) // QA_PER_AUTHOR

    conditions = {
        "heldout": (list(range(0, 10)), list(range(10, 20))),
        "ceiling": (list(range(0, 10)), list(range(0, 10))),
    }

    # Distractor pool: answers from `full` that are not in the perturbed pool at all.
    perturbed_questions = {pt[i]["question"] for i in range(len(pt))}
    distractors = [full[i]["answer"] for i in range(len(full))
                   if full[i]["question"] not in perturbed_questions]
    distractors = distractors[:args.n_distractors]

    summary = {}
    for name, (train_authors, eval_authors) in conditions.items():
        cdir = os.path.join(args.out_dir, name)
        os.makedirs(cdir, exist_ok=True)
        train_records, train_answers, test_items = build_condition(pt, train_authors, eval_authors)

        # The retrieval corpus is the user's memory store: it holds the facts the
        # user actually wrote down. It always contains the eval-relevant facts,
        # because the point of the comparison is substrate (weights vs store),
        # not whether the store happens to be populated.
        gold_facts = []
        for a in eval_authors:
            gold_facts.extend(r["answer"] for r in author_rows(pt, a))
        corpus_small = list(dict.fromkeys(gold_facts))
        corpus_padded = list(dict.fromkeys(gold_facts + distractors))

        with open(os.path.join(cdir, "train_sft.jsonl"), "w") as f:
            for r in train_records:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(cdir, "test.jsonl"), "w") as f:
            for r in test_items:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(cdir, "facts_small.json"), "w") as f:
            json.dump(corpus_small, f)
        with open(os.path.join(cdir, "facts_padded.json"), "w") as f:
            json.dump(corpus_padded, f)

        # Leakage audit: the gate that the previous build silently failed.
        trq = {r["messages"][0]["content"] for r in train_records}
        tra = {r["messages"][1]["content"] for r in train_records}
        teq = {t["q"] for t in test_items}
        tea = {t["true"] for t in test_items}
        audit = {
            "train_records": len(train_records),
            "test_items": len(test_items),
            "train_authors": train_authors,
            "eval_authors": eval_authors,
            "question_overlap": len(trq & teq),
            "answer_overlap": len(tra & tea),
            "corpus_small": len(corpus_small),
            "corpus_padded": len(corpus_padded),
        }
        with open(os.path.join(cdir, "audit.json"), "w") as f:
            json.dump(audit, f, indent=2)
        summary[name] = audit
        print("[%s] %s" % (name, json.dumps(audit)))

    # Hard gate: the held-out condition must have zero overlap.
    h = summary["heldout"]
    assert h["question_overlap"] == 0, "HELD-OUT LEAKAGE: %d questions" % h["question_overlap"]
    assert h["answer_overlap"] == 0, "HELD-OUT LEAKAGE: %d answers" % h["answer_overlap"]
    c = summary["ceiling"]
    assert c["question_overlap"] == c["test_items"], "ceiling condition should overlap fully"
    print("\nGATE PASSED: heldout overlap 0/0; ceiling overlap %d/%d (by design)"
          % (c["question_overlap"], c["test_items"]))
    print("authors in perturbed pool:", n_authors, "| distractors:", len(distractors))


if __name__ == "__main__":
    main()
