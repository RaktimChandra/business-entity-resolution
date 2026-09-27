"""Learned candidate pruner (the last blocking stage).

A small LightGBM scores every blocked pair from blocking-time signals only
(vote score, shared keys, rerank similarity, ranks, competition). Pairs below
a threshold are removed before the matcher runs. The threshold is the
most aggressive one whose out-of-fold loss of true pairs stays within
``prune_tolerance`` (relative to the pairs available before pruning).

Out-of-fold scores are produced for every training pair (folds grouped by
Source 1 entity), so the matcher is trained on exactly the distribution of
candidates it will see at inference.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import BLOCK_FEATURES
from .model import group_folds


def _params(cfg, k):
    return {"objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 100,
            "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "num_threads": cfg.n_jobs,
            "seed": cfg.seed + 500 + k, "deterministic": True, "force_row_wise": True, "verbosity": -1}


def _cap_rows(idx, cap, rng):
    return idx if len(idx) <= cap else np.sort(rng.choice(idx, size=cap, replace=False))


def fit_pruner(cand: pd.DataFrame, y: np.ndarray, cfg, log=None):
    """Return (oof_scores, final_model, threshold, stats)."""
    rng = np.random.default_rng(cfg.seed + 11)
    X = cand[BLOCK_FEATURES].astype(np.float32)
    folds = group_folds(cand["s1_row"].to_numpy(), cfg.n_folds, cfg.seed + 3)
    oof = np.zeros(len(cand), dtype=np.float64)
    for k in range(cfg.n_folds):
        tr = _cap_rows(np.flatnonzero(folds != k), cfg.pruner_max_rows, rng)
        va = np.flatnonzero(folds == k)
        if len(tr) == 0 or len(va) == 0:
            continue
        m = lgb.train(_params(cfg, k), lgb.Dataset(X.iloc[tr], y[tr]), num_boost_round=cfg.pruner_rounds)
        oof[va] = m.predict(X.iloc[va])
    all_idx = _cap_rows(np.arange(len(cand)), cfg.pruner_max_rows, rng)
    final = lgb.train(_params(cfg, 99), lgb.Dataset(X.iloc[all_idx], y[all_idx]),
                      num_boost_round=cfg.pruner_rounds)

    # most aggressive threshold that keeps (1 - tolerance) of the true pairs
    pos_scores = np.sort(oof[y == 1])
    n_pos = len(pos_scores)
    if n_pos == 0:
        thr = 0.0
    else:
        allowed_loss = int(np.floor(cfg.prune_tolerance * n_pos))
        thr = float(pos_scores[allowed_loss]) if allowed_loss < n_pos else 0.0
        thr = min(thr, cfg.prune_max_threshold)
    keep = oof >= thr
    stats = {
        "threshold": thr,
        "pairs_before": int(len(cand)),
        "pairs_after": int(keep.sum()),
        "true_pairs_before": int(n_pos),
        "true_pairs_after": int((keep & (y == 1)).sum()),
    }
    if log:
        log.info("pruner: threshold %.5f keeps %d/%d pairs (%.1f%%) and %d/%d true pairs",
                 thr, stats["pairs_after"], stats["pairs_before"],
                 100 * stats["pairs_after"] / max(1, stats["pairs_before"]),
                 stats["true_pairs_after"], n_pos)
    return oof, final, thr, stats


def apply_pruner(model: lgb.Booster, cand: pd.DataFrame, thr: float) -> np.ndarray:
    if len(cand) == 0:
        return np.zeros(0, dtype=bool)
    return model.predict(cand[BLOCK_FEATURES].astype(np.float32)) >= thr
