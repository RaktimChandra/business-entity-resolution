"""Pairwise feature engineering for (Source 1, candidate) pairs.

All features are computed in vectorised batches:
* string similarities with RapidFuzz ``cpdist`` (C++, multi-threaded);
* character 3-gram cosine via feature hashing (no vocabulary to fit);
* rarity-aware token overlap (IDF-weighted Jaccard) and exact overlap of
  numeric anchors / postal codes / alphanumeric codes via sparse row products.

No feature reads the country label, so unseen countries are handled
identically to the training countries.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import HashingVectorizer

from .normalize import consonant_skeleton

# Raw-token vocabulary for "what was added to the name" features. Plain legal
# suffixes ("Co", "LLC") and decorative variants ("& Co", "[Inc]", "(Services)")
# behave very differently: the former often marks a *different* registered
# entity at the same address, the latter is formatting noise of the same one.
_PLAIN_LEGAL = {"co", "llc", "inc", "corp", "corporation", "company", "ltd", "limited", "pvt",
                "private", "llp", "pc", "lp", "pllc", "plc", "sarl", "sas", "sasu", "eurl",
                "gmbh", "incorporated", "pa", "sa"}
_GENERIC_ADD = {"services", "service", "center", "centre", "partners", "group", "enterprises",
                "solutions", "trading", "society", "council", "global", "the", "m/s", "shri",
                "sri", "mr", "mrs", "sons", "and", "&", "&-"}
_EXTRA_VOCAB = ["co", "llc", "inc", "corp", "corporation", "company", "ltd", "limited", "pvt",
                "private", "llp", "pc", "lp", "&", "and", "[inc]", "[co]", "[llc]", "[corp]",
                "[ltd]", "[llp]", "(services)", "services", "center", "partners", "group", "the",
                "shri", "m/s", "sons", "enterprises", "trading", "society"]
_EXTRA_ID = {t: i + 1 for i, t in enumerate(_EXTRA_VOCAB)}


def _raw_tokens(name: str) -> list[str]:
    return name.lower().replace(",", " ").replace(".", "").split()


def _extra_stats(qa: str, tb: str):
    """Tokens present in raw name ``tb`` but not in ``qa`` (multiset difference)."""
    q = _raw_tokens(qa)
    t = _raw_tokens(tb)
    rem = {}
    for x in q:
        rem[x] = rem.get(x, 0) + 1
    extra = []
    for x in t:
        if rem.get(x, 0) > 0:
            rem[x] -= 1
        else:
            extra.append(x)
    return extra

_WORD_HASH = HashingVectorizer(analyzer=str.split, n_features=2 ** 22, binary=True, norm=None,
                               alternate_sign=False, dtype=np.float32)
_CHAR_HASH = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=2 ** 20,
                               norm="l2", alternate_sign=False, dtype=np.float32)

BLOCK_FEATURES = ["key_score", "n_keys", "rerank", "rank_src", "is_s3",
                  "blk_t_n", "blk_t_rank", "blk_t_margin",
                  "blk_g_rank", "blk_g_gap", "blk_g_n", "blk_g_sum"]


def _cp(a, b, scorer, workers):
    return process.cpdist(a, b, scorer=scorer, workers=workers).astype(np.float32)


def _rowdot(A: sp.csr_matrix, B: sp.csr_matrix) -> np.ndarray:
    return np.asarray(A.multiply(B).sum(axis=1), dtype=np.float32).ravel()


def _rowsum(A: sp.csr_matrix) -> np.ndarray:
    return np.asarray(A.sum(axis=1), dtype=np.float32).ravel()


class RecordSide:
    """Column-oriented view of one side (S1 or targets) with sparse token matrices."""

    def __init__(self, df: pd.DataFrame):
        self.name_norm = df["name_norm"].to_numpy(object)
        self.name_core = df["name_core"].to_numpy(object)
        self.name_phon = df["name_phon"].to_numpy(object)
        self.addr = df["addr_norm"].to_numpy(object)
        self.raw_name = df["business_name"].to_numpy(object)
        self.translit = df["translit"].to_numpy(np.int8)
        self.addr_missing = df["addr_missing"].to_numpy(np.int8)
        self.name_len = np.array([len(s) for s in self.name_core], dtype=np.float32)
        self.name_ntok = np.array([s.count(" ") + 1 if s else 0 for s in self.name_core], dtype=np.float32)
        self.addr_ntok = np.array([s.count(" ") + 1 if s else 0 for s in self.addr], dtype=np.float32)
        self.W_name = _WORD_HASH.transform(self.name_core).tocsr()
        self.W_addr = _WORD_HASH.transform(self.addr).tocsr()
        self.W_nums = _WORD_HASH.transform(df["nums"].to_numpy(object)).tocsr()
        self.W_post = _WORD_HASH.transform(df["postal"].to_numpy(object)).tocsr()
        self.W_code = _WORD_HASH.transform(df["codes"].to_numpy(object)).tocsr()


class FeatureBuilder:
    def __init__(self, s1: pd.DataFrame, targets: pd.DataFrame, n_jobs: int = 1):
        self.q = RecordSide(s1)
        self.t = RecordSide(targets)
        self.workers = n_jobs if n_jobs > 1 else 1
        n_t = max(1, len(targets))
        # rarity weights from target document frequencies (unsupervised, same split)
        df_name = np.asarray((self.t.W_name > 0).sum(axis=0)).ravel()
        df_addr = np.asarray((self.t.W_addr > 0).sum(axis=0)).ravel()
        self.idf_name = np.log1p(n_t / (df_name + 1.0)).astype(np.float32)
        self.idf_addr = np.log1p(n_t / (df_addr + 1.0)).astype(np.float32)
        self.q.W_name_w = self.q.W_name @ sp.diags(self.idf_name)
        self.t.W_name_w = self.t.W_name @ sp.diags(self.idf_name)
        self.q.W_addr_w = self.q.W_addr @ sp.diags(self.idf_addr)
        self.t.W_addr_w = self.t.W_addr @ sp.diags(self.idf_addr)
        self.q.W_name_w = self.q.W_name_w.tocsr()
        self.t.W_name_w = self.t.W_name_w.tocsr()
        self.q.W_addr_w = self.q.W_addr_w.tocsr()
        self.t.W_addr_w = self.t.W_addr_w.tocsr()

    # ------------------------------------------------------------------
    def _weighted_jaccard(self, Aw, Bw, Bbin, qi, tj):
        a = Aw[qi]
        b = Bw[tj]
        shared = _rowdot(a, Bbin[tj])
        tot = _rowsum(a) + _rowsum(b) - shared
        return np.where(tot > 0, shared / np.maximum(tot, 1e-6), 0.0).astype(np.float32), shared

    def _overlap(self, A, B, qi, tj):
        a, b = A[qi], B[tj]
        shared = _rowdot(a, b)
        na, nb = _rowsum(a), _rowsum(b)
        union = na + nb - shared
        jac = np.where(union > 0, shared / np.maximum(union, 1), 0.0).astype(np.float32)
        conflict = ((na > 0) & (nb > 0) & (shared == 0)).astype(np.float32)
        return shared, na, nb, jac, conflict

    def _char_cos(self, q_strings, t_strings, qi, tj):
        uq, inv_q = np.unique(qi, return_inverse=True)
        ut, inv_t = np.unique(tj, return_inverse=True)
        Q = _CHAR_HASH.transform(q_strings[uq]).tocsr()
        T = _CHAR_HASH.transform(t_strings[ut]).tocsr()
        return _rowdot(Q[inv_q], T[inv_t])

    # ------------------------------------------------------------------
    def pair_features(self, qi: np.ndarray, tj: np.ndarray) -> pd.DataFrame:
        q, t, w = self.q, self.t, self.workers
        f: dict[str, np.ndarray] = {}
        qn, tn = q.name_norm[qi].tolist(), t.name_norm[tj].tolist()
        qc, tc = q.name_core[qi].tolist(), t.name_core[tj].tolist()
        qp, tp = q.name_phon[qi].tolist(), t.name_phon[tj].tolist()
        qa, ta = q.addr[qi].tolist(), t.addr[tj].tolist()

        f["n_ratio"] = _cp(qn, tn, fuzz.ratio, w)
        f["n_tset"] = _cp(qn, tn, fuzz.token_set_ratio, w)
        f["n_tsort"] = _cp(qn, tn, fuzz.token_sort_ratio, w)
        f["n_partial"] = _cp(qn, tn, fuzz.partial_ratio, w)
        f["n_jw"] = _cp(qn, tn, JaroWinkler.normalized_similarity, w)
        f["c_ratio"] = _cp(qc, tc, fuzz.ratio, w)
        f["c_tset"] = _cp(qc, tc, fuzz.token_set_ratio, w)
        f["c_partial_tset"] = _cp(qc, tc, fuzz.partial_token_set_ratio, w)
        f["c_jw"] = _cp(qc, tc, JaroWinkler.normalized_similarity, w)
        f["c_exact"] = (q.name_core[qi] == t.name_core[tj]).astype(np.float32)
        q_compact = [x.replace(" ", "") for x in qc]
        t_compact = [x.replace(" ", "") for x in tc]
        f["c_compact_ratio"] = _cp(q_compact, t_compact, fuzz.ratio, w)
        f["p_ratio"] = _cp(qp, tp, fuzz.ratio, w)
        f["p_tset"] = _cp(qp, tp, fuzz.token_set_ratio, w)
        q_sk = [consonant_skeleton(x) for x in qp]
        t_sk = [consonant_skeleton(x) for x in tp]
        f["sk_ratio"] = _cp(q_sk, t_sk, fuzz.ratio, w)
        f["sk_tset"] = _cp(q_sk, t_sk, fuzz.token_set_ratio, w)
        f["a_ratio"] = _cp(qa, ta, fuzz.ratio, w)
        f["a_tset"] = _cp(qa, ta, fuzz.token_set_ratio, w)
        f["a_tsort"] = _cp(qa, ta, fuzz.token_sort_ratio, w)
        f["a_partial"] = _cp(qa, ta, fuzz.partial_ratio, w)
        f["combo_tset"] = _cp([a + " " + b for a, b in zip(qc, qa)],
                              [a + " " + b for a, b in zip(tc, ta)], fuzz.token_set_ratio, w)

        f["n_char_cos"] = self._char_cos(q.name_norm, t.name_norm, qi, tj)
        f["a_char_cos"] = self._char_cos(q.addr, t.addr, qi, tj)

        f["n_idf_jac"], f["n_idf_shared"] = self._weighted_jaccard(q.W_name_w, t.W_name_w, t.W_name, qi, tj)
        f["a_idf_jac"], f["a_idf_shared"] = self._weighted_jaccard(q.W_addr_w, t.W_addr_w, t.W_addr, qi, tj)

        sh, na, nb, jac, conf = self._overlap(q.W_nums, t.W_nums, qi, tj)
        f["num_shared"], f["num_q"], f["num_t"], f["num_jac"], f["num_conflict"] = sh, na, nb, jac, conf
        sh, na, nb, _, conf = self._overlap(q.W_post, t.W_post, qi, tj)
        f["post_shared"], f["post_conflict"] = sh, conf
        f["post_any"] = ((na > 0) & (nb > 0)).astype(np.float32)
        sh, na, nb, _, conf = self._overlap(q.W_code, t.W_code, qi, tj)
        f["code_shared"], f["code_conflict"] = sh, conf

        q_first = [x.split(" ", 1)[0] for x in qc]
        t_first = [x.split(" ", 1)[0] for x in tc]
        f["first_eq"] = np.fromiter((a == b and a != "" for a, b in zip(q_first, t_first)),
                                    dtype=np.float32, count=len(qc))
        q_init = ["".join(tok[0] for tok in x.split()) for x in qc]
        t_init = ["".join(tok[0] for tok in x.split()) for x in tc]
        f["acronym"] = np.fromiter(
            ((len(a) > 1 and a == tcmp) or (len(b) > 1 and b == qcmp)
             for a, b, qcmp, tcmp in zip(q_init, t_init, q_compact, t_compact)),
            dtype=np.float32, count=len(qc))
        ql, tl = q.name_len[qi], t.name_len[tj]
        f["len_ratio"] = (np.minimum(ql, tl) / np.maximum(np.maximum(ql, tl), 1)).astype(np.float32)
        f["ntok_q"], f["ntok_t"] = q.name_ntok[qi], t.name_ntok[tj]
        f["antok_q"], f["antok_t"] = q.addr_ntok[qi], t.addr_ntok[tj]
        f["addr_miss_q"] = q.addr_missing[qi].astype(np.float32)
        f["addr_miss_t"] = t.addr_missing[tj].astype(np.float32)
        f["translit_q"] = q.translit[qi].astype(np.float32)
        f["translit_t"] = t.translit[tj].astype(np.float32)
        f["translit_mix"] = (q.translit[qi] != t.translit[tj]).astype(np.float32)

        # what exactly was added to / removed from the raw name
        n = len(qc)
        feats = np.zeros((n, 11), dtype=np.float32)
        q_raw, t_raw = q.raw_name[qi], t.raw_name[tj]
        for i in range(n):
            ex_t = _extra_stats(q_raw[i], t_raw[i])
            ex_q = _extra_stats(t_raw[i], q_raw[i])
            feats[i, 0] = len(ex_t)
            feats[i, 1] = len(ex_q)
            feats[i, 2] = any(x in _PLAIN_LEGAL for x in ex_t)
            feats[i, 3] = any(x in _PLAIN_LEGAL for x in ex_q)
            feats[i, 4] = any(("[" in x or "(" in x) for x in ex_t)
            feats[i, 5] = any(x in ("&", "and", "&-") for x in ex_t)
            feats[i, 6] = any(x in _GENERIC_ADD for x in ex_t)
            feats[i, 7] = _EXTRA_ID.get(ex_t[0], len(_EXTRA_VOCAB) + 1) if ex_t else 0
            feats[i, 8] = _EXTRA_ID.get(ex_q[0], len(_EXTRA_VOCAB) + 1) if ex_q else 0
            tr_ = t_raw[i]
            feats[i, 9] = ("@" in tr_) or (".com" in tr_.lower()) or ("#" in tr_) or ("www" in tr_.lower())
            feats[i, 10] = (not ex_t) and (not ex_q)
        for j, name in enumerate(["rx_t_n", "rx_q_n", "rx_t_legal", "rx_q_legal", "rx_t_bracket",
                                  "rx_t_amp", "rx_t_generic", "rx_t_first_id", "rx_q_first_id",
                                  "rx_t_handle", "rx_same_tokens"]):
            f[name] = feats[:, j]

        # interactions that trees otherwise need many splits for
        f["name_x_addr"] = (f["c_tset"] * f["a_tset"] / 100.0).astype(np.float32)
        f["best_name"] = np.maximum.reduce([f["n_tset"], f["c_tset"], f["p_tset"],
                                            f["c_compact_ratio"]]).astype(np.float32)
        return pd.DataFrame(f)

    def target_pair_similarity(self, ti: np.ndarray, tj: np.ndarray):
        """Similarity between two targets (used for cluster-support features)."""
        t, w = self.t, self.workers
        n = _cp(t.name_core[ti].tolist(), t.name_core[tj].tolist(), fuzz.token_set_ratio, w)
        a = _cp(t.addr[ti].tolist(), t.addr[tj].tolist(), fuzz.token_set_ratio, w)
        return n, a


def build_features(fb: FeatureBuilder, cand: pd.DataFrame, batch: int, log=None) -> pd.DataFrame:
    """Compute pair features for every row of ``cand`` (keeps row order)."""
    out = []
    qi_all = cand["s1_row"].to_numpy(np.int64)
    tj_all = cand["t_row"].to_numpy(np.int64)
    n = len(cand)
    for s in range(0, n, batch):
        e = min(n, s + batch)
        out.append(fb.pair_features(qi_all[s:e], tj_all[s:e]))
        if log:
            log.info("  features %d/%d pairs", e, n)
    feats = pd.concat(out, ignore_index=True) if out else fb.pair_features(
        np.empty(0, np.int64), np.empty(0, np.int64))
    blk = cand[[c for c in BLOCK_FEATURES + ["p0"] if c in cand.columns]].reset_index(drop=True)
    return pd.concat([blk.astype(np.float32), feats], axis=1)
