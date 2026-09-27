"""Official-metric implementation and blocking diagnostics.

Macro F0.5 per Source 1 entity with the closed form
    F0.5 = 1.25 * TP / (0.25 * |truth| + |pred|)
which equals (1.25 P R) / (0.25 P + R) whenever TP > 0. Singletons score 1.0
for an empty prediction and 0.0 otherwise, exactly as specified.
"""
from __future__ import annotations

import numpy as np


def pair_codes(s1_rows: np.ndarray, t_rows: np.ndarray, n_targets: int) -> np.ndarray:
    return s1_rows.astype(np.int64) * np.int64(n_targets) + t_rows.astype(np.int64)


class GroundTruth:
    """Ground-truth pairs indexed by (S1 row, target row)."""

    def __init__(self, s1_ids, t_ids, gt: dict[str, list[str]]):
        s1_pos = {k: i for i, k in enumerate(s1_ids)}
        t_pos = {k: i for i, k in enumerate(t_ids)}
        self.n_s1 = len(s1_ids)
        self.n_t = len(t_ids)
        rows, cols = [], []
        self.unknown_ids = 0
        for s1, lst in gt.items():
            r = s1_pos.get(s1)
            if r is None:
                continue
            for tid in lst:
                c = t_pos.get(tid)
                if c is None:
                    self.unknown_ids += 1
                    continue
                rows.append(r)
                cols.append(c)
        self.s1_rows = np.asarray(rows, dtype=np.int64)
        self.t_rows = np.asarray(cols, dtype=np.int64)
        self.codes = np.unique(pair_codes(self.s1_rows, self.t_rows, self.n_t))
        self.n_true = np.bincount(self.s1_rows, minlength=self.n_s1).astype(np.int64)

    def is_true(self, s1_rows, t_rows) -> np.ndarray:
        return np.isin(pair_codes(s1_rows, t_rows, self.n_t), self.codes)


def macro_f05(gt: GroundTruth, universe: np.ndarray, pred_s1: np.ndarray, pred_t: np.ndarray) -> dict:
    """Score predicted pairs over the S1 rows in ``universe``."""
    n = gt.n_s1
    in_u = np.zeros(n, dtype=bool)
    in_u[universe] = True
    keep = in_u[pred_s1]
    pred_s1, pred_t = pred_s1[keep], pred_t[keep]
    hit = gt.is_true(pred_s1, pred_t)
    tp = np.bincount(pred_s1, weights=hit, minlength=n)[universe]
    npred = np.bincount(pred_s1, minlength=n)[universe].astype(np.float64)
    ntrue = gt.n_true[universe].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(ntrue == 0, (npred == 0).astype(np.float64),
                     np.where(tp > 0, 1.25 * tp / (0.25 * ntrue + np.maximum(npred, 1e-9)), 0.0))
    has_pred = npred > 0
    has_true = ntrue > 0
    prec = np.where(has_pred, tp / np.maximum(npred, 1), np.nan)
    rec = np.where(has_true, tp / np.maximum(ntrue, 1), np.nan)
    single = ntrue == 0
    return {
        "macro_f05": float(f.mean()) if len(f) else 0.0,
        "macro_precision": float(np.nanmean(prec)) if has_pred.any() else float("nan"),
        "macro_recall": float(np.nanmean(rec)) if has_true.any() else float("nan"),
        "f05_singletons": float(f[single].mean()) if single.any() else float("nan"),
        "f05_non_singletons": float(f[~single].mean()) if (~single).any() else float("nan"),
        "n_entities": int(len(universe)),
        "n_singletons": int(single.sum()),
        "n_pred_pairs": int(npred.sum()),
    }


def blocking_stats(gt: GroundTruth | None, universe: np.ndarray, cand_s1: np.ndarray,
                   cand_t: np.ndarray, n_targets: int) -> dict:
    n_u = len(universe)
    n_pairs = len(cand_s1)
    stats = {
        "candidate_pairs": int(n_pairs),
        "avg_candidates_per_s1": float(n_pairs / max(n_u, 1)),
        "reduction_ratio": float(1.0 - n_pairs / max(n_u * n_targets, 1)),
    }
    if gt is not None:
        in_u = np.zeros(gt.n_s1, dtype=bool)
        in_u[universe] = True
        true_mask = in_u[gt.s1_rows]
        n_true_pairs = int(true_mask.sum())
        hit = gt.is_true(cand_s1, cand_t)
        stats["pair_recall"] = float(hit.sum() / max(n_true_pairs, 1))
        # recall ceiling for the macro metric: F0.5 attainable if the matcher were perfect
        tp = np.bincount(cand_s1, weights=hit, minlength=gt.n_s1)[universe]
        ntrue = gt.n_true[universe]
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(ntrue == 0, 1.0, np.where(tp > 0, 1.25 * tp / (0.25 * ntrue + tp), 0.0))
        stats["oracle_macro_f05"] = float(f.mean()) if len(f) else 0.0
        stats["entities_fully_covered"] = float(((tp == ntrue) & (ntrue > 0)).sum() / max((ntrue > 0).sum(), 1))
    return stats


class FastScorer:
    """Macro F0.5 for many candidate selections over a fixed pair set.

    Correctness of every pair is computed once; each evaluation is then a
    couple of bincounts, which makes decoder search ~100x faster than
    re-matching against the full ground truth for every policy.
    """

    def __init__(self, gt: GroundTruth, universe: np.ndarray, s1_rows: np.ndarray, t_rows: np.ndarray):
        pos = np.full(gt.n_s1, -1, dtype=np.int64)
        pos[universe] = np.arange(len(universe))
        self.loc = pos[s1_rows]
        self.inside = self.loc >= 0
        self.hit = gt.is_true(s1_rows, t_rows)
        self.ntrue = gt.n_true[universe].astype(np.float64)
        self.n = len(universe)

    def __call__(self, sel: np.ndarray) -> dict:
        m = sel & self.inside
        loc = self.loc[m]
        tp = np.bincount(loc, weights=self.hit[m], minlength=self.n)
        npred = np.bincount(loc, minlength=self.n).astype(np.float64)
        ntrue = self.ntrue
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(ntrue == 0, (npred == 0).astype(np.float64),
                         np.where(tp > 0, 1.25 * tp / (0.25 * ntrue + np.maximum(npred, 1e-9)), 0.0))
            prec = np.where(npred > 0, tp / np.maximum(npred, 1), np.nan)
            rec = np.where(ntrue > 0, tp / np.maximum(ntrue, 1), np.nan)
        return {"macro_f05": float(f.mean()) if self.n else 0.0,
                "macro_precision": float(np.nanmean(prec)) if (npred > 0).any() else float("nan"),
                "macro_recall": float(np.nanmean(rec)) if (ntrue > 0).any() else float("nan")}
