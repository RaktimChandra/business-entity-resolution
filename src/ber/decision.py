"""Entity-level set decisions optimised for macro F0.5.

Two decoders are implemented and the better one on out-of-fold predictions
is selected automatically:

* ``threshold``: keep the best candidate if p >= t_top, plus any other
  candidate with p >= t_rest.
* ``expected_f``: choose the prediction set that maximises the *expected*
  F0.5 of the entity. With calibrated match probabilities p_i (sorted
  descending), predicting the top k gives approximately
      E[F_k] = 1.25 * S_k / (0.25 * S / (1 - P0) + k),
  where S_k is the sum of the top-k probabilities, S the sum over all
  candidates and P0 = prod(1 - p_i) the probability that the entity is a
  singleton; predicting nothing scores P0. The decoder picks the arg-max,
  so singleton handling falls out of the same objective.

An optional exclusivity step enforces that one target record is assigned to
at most one Source 1 entity (Source 1 is deduplicated); conflicts are
resolved in favour of the highest probability.
"""
from __future__ import annotations

import itertools

import numpy as np


def _sort_groups(s1: np.ndarray, p: np.ndarray):
    order = np.lexsort((-p, s1))
    g = s1[order]
    new = np.ones(len(g), dtype=bool)
    new[1:] = g[1:] != g[:-1]
    idx = np.arange(len(g))
    start = np.maximum.accumulate(np.where(new, idx, 0))
    rank = idx - start
    gid = np.cumsum(new) - 1
    return order, rank, gid, np.flatnonzero(new)


def decode_threshold(s1, p, t_top, t_rest):
    order, rank, gid, starts = _sort_groups(s1, p)
    ps = p[order]
    top_p = ps[starts][gid]
    sel_sorted = np.where(rank == 0, ps >= t_top, (ps >= t_rest) & (top_p >= t_top))
    sel = np.zeros(len(p), dtype=bool)
    sel[order] = sel_sorted
    return sel


def decode_expected_f(s1, p, alpha=1.0, empty_bias=1.0):
    q = np.clip(p, 1e-6, 1 - 1e-6) ** alpha
    order, rank, gid, starts = _sort_groups(s1, q)
    qs = q[order]
    S_k = np.cumsum(qs)
    base = np.concatenate(([0.0], S_k[:-1]))[starts][gid]
    S_k = S_k - base  # cumulative sum within group
    S = np.add.reduceat(qs, starts)[gid]
    log0 = np.add.reduceat(np.log1p(-qs), starts)[gid]
    P0 = np.exp(log0)
    k = rank + 1
    E = 1.25 * S_k / (0.25 * S / np.maximum(1 - P0, 1e-9) + k)
    best_E = np.maximum.reduceat(E, starts)[gid]
    # first rank achieving the group's best expectation
    is_best = E >= best_E - 1e-12
    best_rank = np.full(len(starts), np.iinfo(np.int64).max)
    np.minimum.at(best_rank, gid[is_best], rank[is_best])
    choose_nonempty = best_E > empty_bias * P0
    sel_sorted = choose_nonempty & (rank <= best_rank[gid])
    sel = np.zeros(len(p), dtype=bool)
    sel[order] = sel_sorted
    return sel


def enforce_exclusivity(s1, t, p, sel):
    """Keep each selected target only for the S1 entity with the highest probability."""
    idx = np.flatnonzero(sel)
    if len(idx) == 0:
        return sel
    order = np.lexsort((-p[idx], t[idx]))
    ti = t[idx][order]
    first = np.ones(len(ti), dtype=bool)
    first[1:] = ti[1:] != ti[:-1]
    out = np.zeros_like(sel)
    out[idx[order][first]] = True
    return out


def apply_policy(policy: dict, s1, t, p):
    if policy["kind"] == "threshold":
        sel = decode_threshold(s1, p, policy["t_top"], policy["t_rest"])
    else:
        sel = decode_expected_f(s1, p, policy["alpha"], policy["empty_bias"])
    if policy.get("exclusive"):
        sel = enforce_exclusivity(s1, t, p, sel)
    return sel


def candidate_policies(exclusive_options=(False, True)):
    for ex in exclusive_options:
        for t_top in np.round(np.arange(0.05, 0.96, 0.05), 3):
            for t_rest in np.round(np.arange(0.20, 0.96, 0.05), 3):
                if t_rest < t_top - 0.25:
                    continue
                yield {"kind": "threshold", "t_top": float(t_top), "t_rest": float(t_rest), "exclusive": ex}
        for alpha, bias in itertools.product([0.8, 1.0, 1.25, 1.5, 2.0], [0.8, 1.0, 1.2, 1.5, 2.0]):
            yield {"kind": "expected_f", "alpha": alpha, "empty_bias": bias, "exclusive": ex}


def tune_policy(score_fn, s1, t, p, exclusive_options):
    """Grid-search the decoders; ``score_fn(sel) -> dict`` with key 'macro_f05'."""
    best, best_score, table = None, -1.0, []
    for pol in candidate_policies(exclusive_options):
        sel = apply_policy(pol, s1, t, p)
        sc = score_fn(sel)
        table.append({**pol, **{k: sc[k] for k in ("macro_f05", "macro_precision", "macro_recall")}})
        if sc["macro_f05"] > best_score + 1e-12:
            best, best_score = pol, sc["macro_f05"]
    return best, best_score, table
