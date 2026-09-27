"""Stage-2 collective features built on stage-1 match probabilities.

A pairwise model judges each pair in isolation. Entity resolution, however,
has structure that isolated pairs cannot see:

* **Competition** - Source 1 is deduplicated, so a target claimed with high
  probability by another Source 1 entity is unlikely to belong to this one.
* **Group context** - the same pair means different things if it is the
  clear winner for its entity or one of several near-identical candidates.
* **Cluster support** - true matches of one entity are near-duplicates of
  each other (the same business seen by Source 2 and Source 3); a candidate
  that looks like the entity's strongest candidate gains support.

These are computed from out-of-fold stage-1 probabilities during training
and from the full stage-1 model at inference, so there is no leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .blocking import _rank_within, add_competition_features, add_group_features

CONTEXT_FEATURES = [
    "p1", "p1_t_n", "p1_t_rank", "p1_t_margin", "p1_g_rank", "p1_g_gap", "p1_g_n", "p1_g_sum",
    "p1_g_n_hi", "sup_p", "sup_name", "sup_addr", "sup_score",
]


def add_context(cand: pd.DataFrame, p1: np.ndarray, fb) -> pd.DataFrame:
    c = cand[["s1_row", "t_row"]].copy()
    c["p1"] = p1.astype(np.float32)
    c = add_competition_features(c, "p1", "p1")
    c = add_group_features(c, "p1", "p1")
    s1 = c["s1_row"].to_numpy(np.int64)
    t = c["t_row"].to_numpy(np.int64)
    p = c["p1"].to_numpy(np.float32)
    hi = (p >= 0.5).astype(np.int32)
    uniq, inv = np.unique(s1, return_inverse=True)
    n_hi = np.bincount(inv, weights=hi, minlength=len(uniq))
    c["p1_g_n_hi"] = (n_hi[inv] - hi).astype(np.float32)  # other strong candidates

    # best and second-best candidate of each entity
    rank = _rank_within(s1, p)
    top1_t = np.full(len(uniq), -1, dtype=np.int64)
    top2_t = np.full(len(uniq), -1, dtype=np.int64)
    top1_p = np.zeros(len(uniq), dtype=np.float32)
    top2_p = np.zeros(len(uniq), dtype=np.float32)
    m1, m2 = rank == 0, rank == 1
    top1_t[inv[m1]], top1_p[inv[m1]] = t[m1], p[m1]
    top2_t[inv[m2]], top2_p[inv[m2]] = t[m2], p[m2]
    other_t = np.where(rank == 0, top2_t[inv], top1_t[inv])
    other_p = np.where(rank == 0, top2_p[inv], top1_p[inv])
    has = other_t >= 0
    sup_n = np.zeros(len(c), dtype=np.float32)
    sup_a = np.zeros(len(c), dtype=np.float32)
    if has.any():
        n_sim, a_sim = fb.target_pair_similarity(t[has], other_t[has])
        sup_n[has], sup_a[has] = n_sim / 100.0, a_sim / 100.0
    c["sup_p"] = np.where(has, other_p, -1.0).astype(np.float32)
    c["sup_name"] = np.where(has, sup_n, -1.0).astype(np.float32)
    c["sup_addr"] = np.where(has, sup_a, -1.0).astype(np.float32)
    c["sup_score"] = np.where(has, other_p * (0.5 * sup_n + 0.5 * sup_a), 0.0).astype(np.float32)
    return c[CONTEXT_FEATURES].astype(np.float32).reset_index(drop=True)
