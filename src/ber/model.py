"""Gradient-boosted pair classifier with leakage-free out-of-fold predictions.

Folds are grouped by Source 1 entity so that no entity contributes pairs to
both the training and validation side of a fold.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd


def lgb_params(cfg, seed_offset: int = 0) -> dict:
    return {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "learning_rate": cfg.lgb_learning_rate,
        "num_leaves": cfg.lgb_num_leaves,
        "min_data_in_leaf": cfg.lgb_min_data_in_leaf,
        "feature_fraction": cfg.lgb_feature_fraction,
        "bagging_fraction": cfg.lgb_bagging_fraction,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "max_bin": 255,
        "num_threads": cfg.n_jobs,
        "seed": cfg.seed + seed_offset,
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
    }


def group_folds(groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Random assignment of whole groups to folds."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    fold_of = rng.integers(0, n_folds, size=len(uniq))
    return fold_of[np.searchsorted(uniq, groups)]


def train_oof(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, cfg, log=None, tag="stage"):
    folds = group_folds(groups, cfg.n_folds, cfg.seed)
    oof = np.zeros(len(y), dtype=np.float64)
    best_iters = []
    for k in range(cfg.n_folds):
        tr, va = folds != k, folds == k
        if va.sum() == 0 or tr.sum() == 0:
            continue
        dtr = lgb.Dataset(X[tr], y[tr], free_raw_data=True)
        dva = lgb.Dataset(X[va], y[va], reference=dtr, free_raw_data=True)
        booster = lgb.train(
            lgb_params(cfg, k), dtr, num_boost_round=cfg.lgb_rounds, valid_sets=[dva],
            callbacks=[lgb.early_stopping(cfg.lgb_early_stopping, verbose=False)],
        )
        it = booster.best_iteration or cfg.lgb_rounds
        best_iters.append(it)
        oof[va] = booster.predict(X[va], num_iteration=it)
        if log:
            log.info("  %s fold %d/%d: best_iter=%d", tag, k + 1, cfg.n_folds, it)
    n_final = int(np.ceil(np.mean(best_iters) * 1.1)) if best_iters else cfg.lgb_rounds
    full = lgb.train(lgb_params(cfg, 99), lgb.Dataset(X, y), num_boost_round=max(n_final, 20))
    return oof, full, {"best_iterations": best_iters, "final_rounds": max(n_final, 20)}


def importance(booster: lgb.Booster) -> pd.DataFrame:
    return pd.DataFrame({
        "feature": booster.feature_name(),
        "gain": booster.feature_importance("gain"),
        "split": booster.feature_importance("split"),
    }).sort_values("gain", ascending=False).reset_index(drop=True)
