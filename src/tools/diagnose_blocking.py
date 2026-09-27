"""Explain why true training pairs are lost during blocking.

Uses the normalised parquet cache written by a previous run (work/train_*.parquet),
samples matched Source 1 entities and classifies every ground-truth pair as:

  kept            - in the final top-K candidates
  lost_topk       - blocked, but ranked beyond K after reranking
  lost_vote       - shares an indexed key, but never reached the top-k_wide vote set
  lost_dfcap      - shares keys only through keys dropped by the df cap
  no_shared_key   - shares no blocking key at all

and reports which key channels the lost pairs share, plus examples for review.

    python src/tools/diagnose_blocking.py --work-dir work --data-dir DATA --sample 20000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ber import blocking as B  # noqa: E402
from ber.config import Config  # noqa: E402
from ber.io_utils import read_ground_truth  # noqa: E402


def record_keys(df, rows, scope):
    r, ch, k = B.build_keys(df.iloc[rows].reset_index(drop=True), scope, 1)
    out = {}
    for rr, cc, kk in zip(r, ch, k):
        out.setdefault(int(rows[rr]), {})[int(kk)] = int(cc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--k", type=int, default=None, help="K per source (default: from artifacts or 19)")
    ap.add_argument("--df-cap", type=int, default=400)
    ap.add_argument("--k-wide", type=int, default=40)
    ap.add_argument("--n-jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="blocking_diagnosis.tsv")
    a = ap.parse_args()

    s1 = pd.read_parquet(os.path.join(a.work_dir, "train_s1.parquet"))
    t = pd.read_parquet(os.path.join(a.work_dir, "train_targets.parquet"))
    gt = read_ground_truth(a.data_dir)
    k = a.k
    art = os.path.join(a.work_dir, "artifacts.json")
    if k is None and os.path.exists(art):
        k = json.load(open(art))["k"]
    k = k or 19

    s1_pos = {x: i for i, x in enumerate(s1["entity_id"])}
    t_pos = {x: i for i, x in enumerate(t["entity_id"])}
    matched = [s for s, v in gt.items() if v and s in s1_pos]
    rng = np.random.default_rng(0)
    pick = rng.choice(len(matched), size=min(a.sample, len(matched)), replace=False)
    sample_ids = [matched[i] for i in pick]
    s_rows = np.array(sorted(s1_pos[s] for s in sample_ids))
    pairs = [(s1_pos[s], t_pos[x]) for s in sample_ids for x in gt[s] if x in t_pos]
    print(f"sampled {len(s_rows)} matched entities, {len(pairs)} true pairs")

    # 1) run the real blocking for the sample (all targets indexed)
    cfg = Config(n_jobs=a.n_jobs, key_df_cap=a.df_cap, k_wide=a.k_wide, k_max=a.k_wide)
    sub = s1.iloc[s_rows].reset_index(drop=True)
    cand, _ = B.generate_candidates(sub, t, cfg, True)
    cand["s1_row"] = s_rows[cand["s1_row"].to_numpy()]
    rank = {(int(a_), int(b_)): int(r) for a_, b_, r in zip(cand["s1_row"], cand["t_row"], cand["rank_src"])}

    # 2) shared keys and their document frequencies
    t_rows_needed = np.array(sorted({p[1] for p in pairs}))
    qk = record_keys(s1, s_rows, True)
    tk = record_keys(t, t_rows_needed, True)
    all_t_rows, _, all_t_keys = B.build_keys(t, True, a.n_jobs)
    uk, cnt = np.unique(all_t_keys, return_counts=True)
    del all_t_rows, all_t_keys

    def df_of(key):
        i = np.searchsorted(uk, key)
        return int(cnt[i]) if i < len(uk) and uk[i] == key else 0

    cats = Counter()
    ch_lost = Counter()
    rows = []
    for s, tt in pairs:
        shared = set(qk.get(s, {})) & set(tk.get(tt, {}))
        dfs = {kk: df_of(kk) for kk in shared}
        r = rank.get((s, tt))
        if r is not None and r < k:
            cat = "kept"
        elif r is not None:
            cat = "lost_topk"
        elif not shared:
            cat = "no_shared_key"
        elif all(d > a.df_cap for d in dfs.values()):
            cat = "lost_dfcap"
        else:
            cat = "lost_vote"
        cats[cat] += 1
        if cat != "kept":
            for kk in shared:
                ch_lost[(cat, B.CHANNELS[qk[s][kk]][0])] += 1
            if sum(1 for x in rows if x["category"] == cat) < 60:
                rows.append({
                    "category": cat, "rank_src": r,
                    "shared_channels": ",".join(sorted({B.CHANNELS[qk[s][kk]][0] for kk in shared})),
                    "min_df": min(dfs.values()) if dfs else "",
                    "s1_name": s1["business_name"].iat[s], "s1_addr": s1["business_address"].iat[s],
                    "t_name": t["business_name"].iat[tt], "t_addr": t["business_address"].iat[tt],
                    "s1_core": s1["name_core"].iat[s], "t_core": t["name_core"].iat[tt],
                    "s1_addr_norm": s1["addr_norm"].iat[s], "t_addr_norm": t["addr_norm"].iat[tt],
                    "country": s1["country"].iat[s],
                })
    total = sum(cats.values())
    print("\n=== true pairs by outcome ===")
    for c in ["kept", "lost_topk", "lost_vote", "lost_dfcap", "no_shared_key"]:
        print(f"{c:15s} {cats[c]:8d}  {100 * cats[c] / max(total, 1):6.2f}%")
    print("\n=== channels shared by lost pairs (top 25) ===")
    for (c, ch), n in ch_lost.most_common(25):
        print(f"{c:15s} {ch:22s} {n}")
    if rows:
        lost = pd.DataFrame(rows)
        print("\n=== lost pairs by country ===")
        print(lost["country"].value_counts().to_string())
        lost.to_csv(a.out, sep="\t", index=False)
        print(f"\nexamples written to {a.out}")


if __name__ == "__main__":
    main()
