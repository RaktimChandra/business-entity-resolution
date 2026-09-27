"""Builds the methodology document from the measured metrics of a run.

Every number in the document is read from ``work/run_log.json`` and the
artefacts written by the pipeline; nothing is typed in by hand.
"""
from __future__ import annotations

import datetime as dt
import json
import os

import pandas as pd

from .blocking import CHANNELS


def _pct(x, d=2):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if v != v else f"{100 * v:.{d}f}%"


def _f(x, d=4):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if v != v else f"{v:.{d}f}"


def build_document(cfg, template_path: str, out_path: str) -> None:
    with open(os.path.join(cfg.work_dir, "run_log.json"), encoding="utf-8") as fh:
        log = json.load(fh)
    m = log["metrics"]
    with open(os.path.join(cfg.work_dir, "artifacts.json"), encoding="utf-8") as fh:
        art = json.load(fh)
    prof = m.get("train_profile", {})
    tb = m.get("train_blocking", {})
    teb = m.get("test_blocking", {})
    val = m.get("validation", {})
    oof = val.get("oof", {})
    pol = val.get("policy", art.get("policy", {}))
    s1 = m.get("stage1", {})
    tp = m.get("test_predictions", {})
    tprof = m.get("test_profile", {})
    ep = m.get("error_patterns", {})
    pre = tb.get("before_pruning", {}) or {}
    pr = tb.get("pruner", {}) or {}

    imp_path = os.path.join(cfg.work_dir, "feature_importance.csv")
    imp_rows = ""
    if os.path.exists(imp_path):
        imp = pd.read_csv(imp_path).head(15)
        tot = imp["gain"].sum() or 1.0
        imp_rows = "\n".join(f"| {i + 1} | `{r.feature}` | {100 * r.gain / tot:.1f}% |"
                             for i, r in imp.iterrows())

    curve_rows = ""
    for row in m.get("train_k_curve", []):
        if row["k"] in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20) or row["k"] == art["k"]:
            mark = " **(selected)**" if row["k"] == art["k"] else ""
            curve_rows += (f"| {row['k']}{mark} | {_pct(row['pair_recall'])} | {_f(row['oracle_macro_f05'])} "
                           f"| {row['avg_candidates_per_s1']:.2f} |\n")

    pol_path = os.path.join(cfg.work_dir, "policy_search.csv")
    pol_rows = ""
    if os.path.exists(pol_path):
        pt = pd.read_csv(pol_path)
        for kind in ("threshold", "expected_f"):
            sub = pt[pt["kind"] == kind].head(3)
            for _, r in sub.iterrows():
                params = (f"t_top={r['t_top']}, t_rest={r['t_rest']}" if kind == "threshold"
                          else f"alpha={r['alpha']}, empty_bias={r['empty_bias']}")
                pol_rows += (f"| {kind} | {params} | {bool(r['exclusive'])} | {_f(r['macro_f05'])} "
                             f"| {_f(r['macro_precision'])} | {_f(r['macro_recall'])} |\n")

    err_rows = ""
    for view, groups in m.get("error_analysis", {}).items():
        if len(groups) < 2:
            continue  # a view with a single group carries no information
        for g, r in sorted(groups.items(), key=lambda kv: -kv[1]["entities"]):
            if g.startswith("0 (singleton)"):
                err_rows += (f"| {view} | {g} | {r['entities']:,} | {_f(r['macro_f05'])} | — | — |\n")
                continue
            err_rows += (f"| {view} | {g} | {r['entities']:,} | {_f(r['macro_f05'])} | "
                         f"{_f(r['precision'])} | {_f(r['recall'])} |\n")

    stage_rows = "\n".join(f"| {s['stage']} | {s['seconds']:.0f} s | {s['peak_rss_mb']:.0f} MB |"
                           for s in log.get("stages", []))
    channels = "\n".join(f"| `{v[0]}` | {v[1]} |" for v in CHANNELS.values())
    if pol.get("kind") == "threshold":
        policy_text = (f"threshold decoder with t_top = {pol.get('t_top')} and t_rest = {pol.get('t_rest')}"
                       f"{' plus one-target-one-entity exclusivity' if pol.get('exclusive') else ''}")
    else:
        policy_text = (f"expected-F0.5 decoder with calibration exponent alpha = {pol.get('alpha')} and "
                       f"empty-set bias = {pol.get('empty_bias')}"
                       f"{' plus one-target-one-entity exclusivity' if pol.get('exclusive') else ''}")

    values = {
        "TEAM_NAME": cfg.team_name,
        "TEAM_MEMBERS": ", ".join(cfg.team_members),
        "DATE": dt.date.today().strftime("%B %d, %Y"),
        "N_S1": f"{prof.get('n_source1', 0):,}",
        "N_T": f"{prof.get('n_targets', 0):,}",
        "N_S2": f"{prof.get('n_source2', 0):,}",
        "N_S3": f"{prof.get('n_source3', 0):,}",
        "SINGLETON": _pct(prof.get("singleton_rate")),
        "MEAN_MATCH": _f(prof.get("matches_per_matched_entity_mean"), 2),
        "MAX_MATCH": str(prof.get("matches_per_entity_max", "n/a")),
        "CROSS": _pct(prof.get("cross_country_match_rate"), 4),
        "MULTI": _pct(prof.get("target_multi_claim_rate"), 4),
        "TRANSLIT_S1": _pct(prof.get("translit_rate_source1")),
        "TRANSLIT_T": _pct(prof.get("translit_rate_targets")),
        "ADDR_MISS_S1": _pct(prof.get("addr_missing_rate_source1")),
        "ADDR_MISS_T": _pct(prof.get("addr_missing_rate_targets")),
        "SCOPE": "yes" if art.get("scope_country") else "no",
        "K": str(art.get("k")),
        "TRAIN_RECALL": _pct(tb.get("pair_recall")),
        "TRAIN_ORACLE": _f(tb.get("oracle_macro_f05")),
        "TRAIN_CPS": _f(tb.get("avg_candidates_per_s1"), 2),
        "TRAIN_RR": _pct(tb.get("reduction_ratio"), 6),
        "TEST_PAIRS": f"{teb.get('candidate_pairs', 0):,}",
        "TEST_CPS": _f(teb.get("avg_candidates_per_s1"), 2),
        "TEST_RR": _pct(teb.get("reduction_ratio"), 6),
        "TEST_N_S1": f"{tprof.get('n_source1', 0):,}",
        "TEST_N_T": f"{tprof.get('n_targets', 0):,}",
        "DF_CAP": str(cfg.key_df_cap),
        "K_WIDE": str(cfg.k_wide),
        "SAMPLE_E": f"{m.get('train_sample', {}).get('entities', 0):,}",
        "SAMPLE_P": f"{m.get('train_sample', {}).get('pairs', 0):,}",
        "POS_RATE": _pct(m.get("train_pairs_positive_rate")),
        "N_FOLDS": str(cfg.n_folds),
        "F1_STAGE1": _f(val.get("stage1_macro_f05")),
        "F1_STAGE2": _f(val.get("stage2_macro_f05")),
        "USED_STAGE": "stage 2" if val.get("use_stage2") else "stage 1",
        "OOF_F": _f(oof.get("macro_f05")),
        "OOF_P": _f(oof.get("macro_precision")),
        "OOF_R": _f(oof.get("macro_recall")),
        "OOF_SING": _f(oof.get("f05_singletons")),
        "OOF_NONSING": _f(oof.get("f05_non_singletons")),
        "SAMPLE_ORACLE": _f(val.get("sample_blocking", {}).get("oracle_macro_f05")),
        "POLICY": policy_text,
        "FINAL_ROUNDS": str(s1.get("final_rounds", "n/a")),
        "TEST_PRED_PAIRS": f"{tp.get('predicted_pairs', 0):,}",
        "TEST_PRED_SING": _pct(tp.get("predicted_singleton_rate")),
        "TEST_MEAN_MATCH": _f(tp.get("mean_matches_per_entity"), 2),
        "IMP_ROWS": imp_rows,
        "PRE_RECALL": _pct(pre.get("pair_recall")),
        "PRE_CPS": _f(pre.get("avg_candidates_per_s1"), 2),
        "PRE_ORACLE": _f(pre.get("oracle_macro_f05")),
        "PRUNE_THR": _f(pr.get("threshold"), 5),
        "PRUNE_KEEP": _pct(pr.get("pairs_after", 0) / max(pr.get("pairs_before", 1), 1)),
        "PRUNE_TRUE_KEEP": _pct(pr.get("true_pairs_after", 0) / max(pr.get("true_pairs_before", 1), 1)),
        "TRAIN_PAIRS": f"{tb.get('candidate_pairs', 0):,}",
        "TRAIN_PRE_PAIRS": f"{pre.get('candidate_pairs', 0):,}",
        "TEST_PRE_PAIRS": f"{teb.get('pairs_before_pruning', 0):,}",
        "FP_N": f"{ep.get('false_positive_pairs', 0):,}",
        "FP_SAME": _pct(ep.get("fp_share_identical_core_name")),
        "FP_NUM": _pct(ep.get("fp_share_numeric_conflict")),
        "FP_ADDR": _pct(ep.get("fp_share_address_missing")),
        "FN_N": f"{ep.get('false_negative_pairs_in_candidates', 0):,}",
        "FN_MIX": _pct(ep.get("fn_share_mixed_script")),
        "FN_ADDR": _pct(ep.get("fn_share_address_missing")),
        "FN_LOW": _pct(ep.get("fn_share_low_name_similarity")),
        "FN_BLOCK": f"{ep.get('true_pairs_missed_by_blocking', 0):,}",
        "FN_TOTAL": f"{ep.get('true_pairs_total', 0):,}",
        "CURVE_ROWS": curve_rows.rstrip("\n"),
        "POLICY_ROWS": pol_rows.rstrip("\n"),
        "STAGE_ROWS": stage_rows,
        "ERROR_ROWS": err_rows.rstrip("\n"),
        "CHANNEL_ROWS": channels,
        "N_STAGE1_FEATS": str(len(art.get("stage1_features", []))),
        "N_STAGE2_FEATS": str(len(art.get("stage2_features", []))),
        "LR": str(cfg.lgb_learning_rate),
        "LEAVES": str(cfg.lgb_num_leaves),
        "MIN_LEAF": str(cfg.lgb_min_data_in_leaf),
    }
    with open(template_path, encoding="utf-8") as fh:
        text = fh.read()
    for key, val_ in values.items():
        text = text.replace("{{" + key + "}}", val_)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)
