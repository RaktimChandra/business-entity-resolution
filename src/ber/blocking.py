"""Scalable candidate generation (blocking).

Pipeline
--------
1. **Multi-channel keys.** Every record emits hashed keys from complementary
   views: exact core name, name tokens, name token bigrams, phonetic skeletons,
   name prefix, numeric anchors (house / postal / phone numbers), alphanumeric
   codes, address bigrams and a name+number compound key. Each key is scoped
   by the country label (open set, never enumerated) when training data shows
   no cross-country matches.
2. **Inverted index over targets** (Source 2 + Source 3) stored as sorted
   uint64 arrays. Keys shared by more than ``key_df_cap`` targets are dropped,
   which bounds the work per query key (no all-pairs comparison anywhere).
3. **IDF-weighted vote.** A (S1, target) pair scores ``sum(w_channel * idf(key))``
   over shared keys, computed with fully vectorised numpy (sort + bincount).
4. **Two-level pruning.** Keep ``k_wide`` per (S1, source) by vote, rerank them
   with a cheap string similarity, then keep the final ``K`` per (S1, source).

Complexity is O(sum over query keys of df(key)) with df capped, i.e. linear in
the number of Source 1 records.
"""
from __future__ import annotations

from multiprocessing import Pool

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from .normalize import consonant_skeleton

# channel id -> (name, weight)
CHANNELS = {
    0: ("name_exact", 3.0),
    1: ("name_token", 1.0),
    2: ("name_prefix5", 1.0),
    3: ("name_phonetic_token", 0.6),
    4: ("name_bigram", 1.5),
    5: ("number", 1.0),
    6: ("alnum_code", 1.5),
    7: ("addr_bigram", 1.0),
    8: ("name_plus_number", 2.0),
    9: ("phonetic_exact", 2.0),
    10: ("addr_token", 0.4),
    11: ("number_plus_street", 2.0),
}
CHANNEL_WEIGHTS = np.array([CHANNELS[i][1] for i in range(len(CHANNELS))], dtype=np.float32)


def _record_keys(country, core, phon, addr, nums, codes):
    """All blocking keys for one record as (channel, value) string pairs."""
    keys = []
    c = country if country is not None else ""
    if core:
        toks = core.split()
        keys.append((0, core))
        for t in toks:
            if len(t) >= 3 and not t.isdigit():
                keys.append((1, t))
        compact = core.replace(" ", "")
        if len(compact) >= 4:
            keys.append((2, compact[:5]))
        for a, b in zip(toks, toks[1:]):
            keys.append((4, a + "_" + b))
    if phon:
        # vowel-free skeleton: robust to transliteration ("iunaited phuds" ~ "united foods")
        skel = consonant_skeleton(phon)
        for t in skel.split():
            if len(t) >= 2 and not t.isdigit():
                keys.append((3, t))
        keys.append((9, skel))
    long_nums = [n for n in nums.split() if len(n) >= 3] if nums else []
    for n in long_nums:
        keys.append((5, n))
    if codes:
        for cd in codes.split():
            keys.append((6, cd))
    if addr:
        atoks = addr.split()
        for a, b in zip(atoks, atoks[1:]):
            keys.append((7, a + "_" + b))
        alpha = []
        digits = []
        for t in atoks:
            if len(t) >= 4 and t.isalpha():
                keys.append((10, t))
                if len(alpha) < 4 and t not in alpha:
                    alpha.append(t)
            elif t.isdigit() and len(digits) < 2 and t not in digits:
                digits.append(t)
        # house number x street word: rare even when each part alone is common
        for d in digits:
            for a in alpha:
                keys.append((11, d + "#" + a))
    if core and long_nums:
        first = core.split()[0]
        for n in long_nums[:3]:
            keys.append((8, first + "#" + n))
    # de-duplicate within the record
    return list(dict.fromkeys(keys)), c


def _keys_chunk(args):
    rows, country, core, phon, addr, nums, codes, scope_country = args
    out_rows, out_ch, out_str = [], [], []
    for r, cc, co, ph, ad, nu, cd in zip(rows, country, core, phon, addr, nums, codes):
        keys, c = _record_keys(cc, co, ph, ad, nu, cd)
        prefix = c + "|" if scope_country else ""
        for ch, val in keys:
            out_rows.append(r)
            out_ch.append(ch)
            out_str.append(f"{prefix}{ch}|{val}")
    if not out_str:
        return (np.empty(0, np.int32), np.empty(0, np.int8), np.empty(0, np.uint64))
    hashes = pd.util.hash_array(np.asarray(out_str, dtype=object), categorize=False)
    return (np.asarray(out_rows, dtype=np.int32), np.asarray(out_ch, dtype=np.int8),
            hashes.astype(np.uint64))


def build_keys(df: pd.DataFrame, scope_country: bool, n_jobs: int = 1, chunk: int = 250_000):
    """Return (row, channel, key_hash) arrays for every record of ``df``.

    ``row`` is the positional index of the record inside ``df``.
    """
    n = len(df)
    cols = [df[c].tolist() for c in ("country", "name_core", "name_phon", "addr_norm", "nums", "codes")]
    tasks = []
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        tasks.append((list(range(s, e)),) + tuple(col[s:e] for col in cols) + (scope_country,))
    if n_jobs > 1 and len(tasks) > 1:
        with Pool(min(n_jobs, len(tasks))) as pool:
            parts = pool.map(_keys_chunk, tasks)
    else:
        parts = [_keys_chunk(t) for t in tasks]
    if not parts:
        return np.empty(0, np.int32), np.empty(0, np.int8), np.empty(0, np.uint64)
    rows = np.concatenate([p[0] for p in parts])
    ch = np.concatenate([p[1] for p in parts])
    keys = np.concatenate([p[2] for p in parts])
    return rows, ch, keys


class TargetIndex:
    """Inverted index: key hash -> sorted list of target rows."""

    def __init__(self, t_rows: np.ndarray, t_keys: np.ndarray, n_targets: int, df_cap: int):
        order = np.argsort(t_keys, kind="stable")
        keys_sorted = t_keys[order]
        rows_sorted = t_rows[order]
        uk, starts, counts = np.unique(keys_sorted, return_index=True, return_counts=True)
        keep = counts <= df_cap
        self.dropped_keys = int((~keep).sum())
        self.total_keys = int(len(uk))
        # compact storage: only postings of kept keys
        keep_mask_postings = np.repeat(keep, counts)
        self.rows = rows_sorted[keep_mask_postings].astype(np.int32)
        self.keys = uk[keep]
        self.counts = counts[keep].astype(np.int64)
        self.starts = np.zeros(len(self.keys), dtype=np.int64)
        if len(self.keys):
            self.starts[1:] = np.cumsum(self.counts)[:-1]
        self.idf = np.log1p(n_targets / self.counts).astype(np.float32)
        self.n_targets = n_targets

    def lookup(self, q_keys: np.ndarray):
        """Position in the index for each query key, or -1 if absent/dropped."""
        if len(self.keys) == 0:
            return np.full(len(q_keys), -1, dtype=np.int64)
        pos = np.searchsorted(self.keys, q_keys)
        pos_c = np.minimum(pos, len(self.keys) - 1)
        hit = self.keys[pos_c] == q_keys
        return np.where(hit, pos_c, -1)


def _vote(index: TargetIndex, q_rows, q_ch, q_keys):
    """IDF-weighted key voting for a batch of query keys -> aggregated pairs."""
    pos = index.lookup(q_keys)
    ok = pos >= 0
    pos, q_rows, q_ch = pos[ok], q_rows[ok], q_ch[ok]
    if len(pos) == 0:
        return (np.empty(0, np.int64),) * 2 + (np.empty(0, np.float32), np.empty(0, np.int16))
    lens = index.counts[pos]
    total = int(lens.sum())
    rep_q = np.repeat(q_rows.astype(np.int64), lens)
    w = (CHANNEL_WEIGHTS[q_ch] * index.idf[pos]).astype(np.float32)
    rep_w = np.repeat(w, lens)
    # postings positions: starts[pos] + 0..len-1
    offs = np.repeat(index.starts[pos] - np.concatenate(([0], np.cumsum(lens)[:-1])), lens)
    t = index.rows[np.arange(total, dtype=np.int64) + offs].astype(np.int64)
    code = rep_q * np.int64(index.n_targets) + t
    uniq, inv = np.unique(code, return_inverse=True)
    score = np.bincount(inv, weights=rep_w).astype(np.float32)
    nkeys = np.bincount(inv).astype(np.int16)
    return uniq // index.n_targets, uniq % index.n_targets, score, nkeys


def _rank_within(groups: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Dense 0-based rank of ``score`` (descending) inside each group; returns in input order."""
    order = np.lexsort((-score, groups))
    g_sorted = groups[order]
    new_group = np.ones(len(order), dtype=bool)
    new_group[1:] = g_sorted[1:] != g_sorted[:-1]
    idx = np.arange(len(order))
    group_start = np.maximum.accumulate(np.where(new_group, idx, 0))
    rank_sorted = idx - group_start
    rank = np.empty(len(order), dtype=np.int32)
    rank[order] = rank_sorted
    return rank


def cheap_similarity(q_core, t_core, q_addr, t_addr, q_phon, t_phon, n_jobs, q_nums=None, t_nums=None):
    """Fast rerank score in [0, 1] used only to order candidates.

    Two evidence paths, the stronger one wins:
    * name path  - name, phonetic and address similarity blended;
    * address path - address tokens plus exact numeric anchors. This keeps
      records whose *name* is unusable (other script, handle, placeholder) but
      whose address is identical.
    """
    w = n_jobs if n_jobs > 1 else 1
    s_name = process.cpdist(q_core, t_core, scorer=fuzz.token_set_ratio, workers=w) / 100.0
    s_addr = process.cpdist(q_addr, t_addr, scorer=fuzz.token_set_ratio, workers=w) / 100.0
    s_phon = process.cpdist(q_phon, t_phon, scorer=fuzz.ratio, workers=w) / 100.0
    q_len = np.fromiter((len(x) for x in q_addr), dtype=np.int32, count=len(q_addr))
    t_len = np.fromiter((len(x) for x in t_addr), dtype=np.int32, count=len(t_addr))
    addr_missing = (q_len == 0) | (t_len == 0)
    with_addr = 0.55 * s_name + 0.30 * s_addr + 0.15 * s_phon
    no_addr = 0.75 * s_name + 0.25 * s_phon
    name_path = np.where(addr_missing, no_addr, with_addr)
    if q_nums is not None:
        s_num = process.cpdist(q_nums, t_nums, scorer=fuzz.token_set_ratio, workers=w) / 100.0
        qn = np.fromiter((len(x) for x in q_nums), dtype=np.int32, count=len(q_nums))
        tn = np.fromiter((len(x) for x in t_nums), dtype=np.int32, count=len(t_nums))
        has_num = (qn > 0) & (tn > 0)
        addr_path = np.where(has_num, 0.6 * s_addr + 0.4 * s_num, 0.85 * s_addr)
        addr_path = np.where(addr_missing, 0.0, 0.95 * addr_path)
        name_path = np.maximum(name_path, addr_path)
    return name_path.astype(np.float32)


def generate_candidates(s1: pd.DataFrame, targets: pd.DataFrame, cfg, scope_country: bool, log=None):
    """Run blocking for all S1 rows against all targets.

    Returns a DataFrame with columns
    ``s1_row, t_row, key_score, n_keys, rerank, rank_src`` restricted to
    ``rank_src < cfg.k_max`` (final K is applied later), plus index stats.
    """
    n_jobs = cfg.n_jobs
    t_rows, t_ch, t_keys = build_keys(targets, scope_country, n_jobs)
    index = TargetIndex(t_rows, t_keys, len(targets), cfg.key_df_cap)
    del t_rows, t_ch, t_keys
    if log:
        log.info("index: %d target keys kept, %d dropped (df > %d)",
                 len(index.keys), index.dropped_keys, cfg.key_df_cap)

    q_rows, q_ch, q_keys = build_keys(s1, scope_country, n_jobs)
    order = np.argsort(q_rows, kind="stable")
    q_rows, q_ch, q_keys = q_rows[order], q_ch[order], q_keys[order]
    bounds = np.searchsorted(q_rows, np.arange(0, len(s1) + cfg.block_batch, cfg.block_batch))

    t_source = targets["source"].to_numpy(np.int8)
    t_core = targets["name_core"].to_numpy(object)
    t_addr = targets["addr_norm"].to_numpy(object)
    t_phon = targets["name_phon"].to_numpy(object)
    t_nums = targets["nums"].to_numpy(object)
    s_core = s1["name_core"].to_numpy(object)
    s_addr = s1["addr_norm"].to_numpy(object)
    s_phon = s1["name_phon"].to_numpy(object)
    s_nums = s1["nums"].to_numpy(object)

    parts = []
    n_voted = 0
    for b in range(len(bounds) - 1):
        lo, hi = bounds[b], bounds[b + 1]
        if hi <= lo:
            continue
        qr, tr, sc, nk = _vote(index, q_rows[lo:hi], q_ch[lo:hi], q_keys[lo:hi])
        n_voted += len(qr)
        if len(qr) == 0:
            continue
        src = t_source[tr]
        grp = qr * 4 + src
        r1 = _rank_within(grp, sc)
        keep = r1 < cfg.k_wide
        qr, tr, sc, nk, src, grp = qr[keep], tr[keep], sc[keep], nk[keep], src[keep], grp[keep]
        rr = cheap_similarity(s_core[qr].tolist(), t_core[tr].tolist(), s_addr[qr].tolist(),
                              t_addr[tr].tolist(), s_phon[qr].tolist(), t_phon[tr].tolist(), n_jobs,
                              s_nums[qr].tolist(), t_nums[tr].tolist())
        rr = rr + 0.02 * np.log1p(sc)
        r2 = _rank_within(grp, rr)
        keep = r2 < cfg.k_max
        parts.append(pd.DataFrame({
            "s1_row": qr[keep].astype(np.int32), "t_row": tr[keep].astype(np.int32),
            "key_score": sc[keep], "n_keys": nk[keep], "rerank": rr[keep].astype(np.float32),
            "rank_src": r2[keep].astype(np.int16), "t_source": src[keep],
        }))
        if log and (b % 10 == 0):
            log.info("  blocking batch %d/%d  (S1 rows %d..%d)", b + 1, len(bounds) - 1,
                     int(q_rows[lo]), int(q_rows[hi - 1]))
    cand = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        {c: pd.Series(dtype=t) for c, t in [("s1_row", np.int32), ("t_row", np.int32),
                                            ("key_score", np.float32), ("n_keys", np.int16),
                                            ("rerank", np.float32), ("rank_src", np.int16),
                                            ("t_source", np.int8)]})
    stats = {"index_keys_kept": int(len(index.keys)), "index_keys_dropped": index.dropped_keys,
             "pairs_voted": int(n_voted)}
    return cand, stats


def add_competition_features(cand: pd.DataFrame, score_col: str, prefix: str) -> pd.DataFrame:
    """For each pair, how strongly do *other* S1 entities claim the same target?

    Adds ``{prefix}_t_n`` (number of S1 claiming the target), ``{prefix}_t_rank``
    (rank of this S1 among them) and ``{prefix}_t_margin`` (this score minus the
    best score of any other S1 for that target).
    """
    t = cand["t_row"].to_numpy(np.int64)
    s = cand[score_col].to_numpy(np.float32)
    rank = _rank_within(t, s)
    uniq, inv, cnt = np.unique(t, return_inverse=True, return_counts=True)
    top1 = np.full(len(uniq), -np.inf, dtype=np.float32)
    np.maximum.at(top1, inv, s)
    s_masked = np.where(rank == 0, -np.inf, s).astype(np.float32)
    top2 = np.full(len(uniq), -np.inf, dtype=np.float32)
    np.maximum.at(top2, inv, s_masked)
    other_best = np.where(rank == 0, top2[inv], top1[inv])
    margin = np.where(np.isfinite(other_best), s - other_best, 1.0).astype(np.float32)
    cand[f"{prefix}_t_n"] = cnt[inv].astype(np.int16)
    cand[f"{prefix}_t_rank"] = rank.astype(np.int16)
    cand[f"{prefix}_t_margin"] = margin
    return cand


def add_group_features(cand: pd.DataFrame, score_col: str, prefix: str) -> pd.DataFrame:
    """Per-S1 context: rank of this candidate, gap to the best, group size/sum."""
    g = cand["s1_row"].to_numpy(np.int64)
    s = cand[score_col].to_numpy(np.float32)
    rank = _rank_within(g, s)
    uniq, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    top = np.full(len(uniq), -np.inf, dtype=np.float32)
    np.maximum.at(top, inv, s)
    tot = np.bincount(inv, weights=s).astype(np.float32)
    cand[f"{prefix}_g_rank"] = rank.astype(np.int16)
    cand[f"{prefix}_g_gap"] = (s - top[inv]).astype(np.float32)
    cand[f"{prefix}_g_n"] = cnt[inv].astype(np.int16)
    cand[f"{prefix}_g_sum"] = tot[inv]
    return cand
