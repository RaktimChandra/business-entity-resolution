"""Score a matching_results.tsv against a ground-truth file with the official macro F0.5.

    python src/tools/score.py --pred output/matching_results.tsv --truth path/to/ground_truth.tsv
"""
from __future__ import annotations

import argparse
import csv


def _read(path, col):
    out = {}
    with open(path, encoding="utf-8") as fh:
        r = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(r)
        ci = header.index(col)
        for row in r:
            if not row:
                continue
            ids = row[ci] if len(row) > ci else ""
            out[row[0]] = {x for x in ids.split(",") if x}
    return out


def score(pred_path, truth_path):
    pred = _read(pred_path, "matched_entity_ids")
    truth = _read(truth_path, "matched_entity_ids")
    tot, n = 0.0, 0
    single_ok = single_n = 0
    for s1, t in truth.items():
        p = pred.get(s1, set())
        if not t:
            f = 1.0 if not p else 0.0
            single_n += 1
            single_ok += f
        else:
            tp = len(p & t)
            f = 1.25 * tp / (0.25 * len(t) + len(p)) if tp else 0.0
        tot += f
        n += 1
    return {"macro_f05": tot / max(n, 1), "entities": n,
            "singleton_accuracy": single_ok / max(single_n, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--truth", required=True)
    a = ap.parse_args()
    print(score(a.pred, a.truth))
