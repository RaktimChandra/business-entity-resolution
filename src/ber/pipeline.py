"""End-to-end pipeline: data -> normalisation -> blocking -> matching -> outputs.

Stages (each tracked with timings and peak memory in ``work/run_log.json``)
------------------------------------------------------------------------
1. load + normalise all sources (cached as parquet)
2. data profiling on the training split (drives automatic decisions)
3. training-split blocking for every Source 1 entity, K selection from the
   recall curve, blocking diagnostics
4. competition-preserving training sample, pair features
5. stage-1 LightGBM (out-of-fold) -> stage-2 collective features ->
   stage-2 LightGBM (out-of-fold)
6. decoder selection on out-of-fold predictions (macro F0.5)
7. test-split blocking with the chosen K -> candidate_pairs.tsv
8. test inference (stage 1 -> context -> stage 2) -> decoder -> matching_results.tsv
9. validation (built-in checker + official script when present), report
"""
from __future__ import annotations

import json
import os
import shutil

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import blocking as B
from .config import Config
from .context import CONTEXT_FEATURES, add_context
from .decision import apply_policy, tune_policy
from .features import BLOCK_FEATURES as B_FEATS
from .features import FeatureBuilder, build_features
from .io_utils import read_ground_truth, read_source, write_id_lists
from .metrics import GroundTruth, blocking_stats, macro_f05
from .model import importance, train_oof
from .normalize import normalize_frame
from .pruner import fit_pruner
from .tracking import Tracker
from .validate import run_official, validate


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_split(cfg: Config, split: str, tracker: Tracker):
    cache_s1 = os.path.join(cfg.work_dir, f"{split}_s1.parquet")
    cache_t = os.path.join(cfg.work_dir, f"{split}_targets.parquet")
    if os.path.exists(cache_s1) and os.path.exists(cache_t):
        tracker.log.info("using cached normalised %s data", split)
        return pd.read_parquet(cache_s1), pd.read_parquet(cache_t)
    s1 = read_source(cfg.data_dir, split, 1)
    t = pd.concat([read_source(cfg.data_dir, split, 2), read_source(cfg.data_dir, split, 3)],
                  ignore_index=True)
    for name, df in (("source1", s1), ("targets", t)):
        dup = int(df["entity_id"].duplicated().sum())
        if dup:
            tracker.log.warning("%s %s has %d duplicated entity_id rows; keeping first", split, name, dup)
    s1 = s1.drop_duplicates("entity_id").reset_index(drop=True)
    t = t.drop_duplicates("entity_id").reset_index(drop=True)
    tracker.log.info("%s: %d source-1 records, %d target records", split, len(s1), len(t))
    s1 = normalize_frame(s1, cfg.n_jobs)
    t = normalize_frame(t, cfg.n_jobs)
    s1.to_parquet(cache_s1, index=False)
    t.to_parquet(cache_t, index=False)
    return s1, t


def profile_training(s1, t, gt_dict) -> dict:
    s1_country = dict(zip(s1["entity_id"], s1["country"]))
    t_country = dict(zip(t["entity_id"], t["country"]))
    n_match = np.array([len(v) for v in gt_dict.values()])
    cross, total = 0, 0
    claims: dict[str, int] = {}
    for k, lst in gt_dict.items():
        c = s1_country.get(k)
        for x in lst:
            total += 1
            if t_country.get(x) != c:
                cross += 1
            claims[x] = claims.get(x, 0) + 1
    multi_claim = sum(1 for v in claims.values() if v > 1)
    return {
        "n_source1": int(len(s1)),
        "n_targets": int(len(t)),
        "n_source2": int((t["source"] == 2).sum()),
        "n_source3": int((t["source"] == 3).sum()),
        "countries_source1": s1["country"].value_counts().to_dict(),
        "countries_targets": t["country"].value_counts().to_dict(),
        "gt_entities": int(len(gt_dict)),
        "singleton_rate": float((n_match == 0).mean()) if len(n_match) else 0.0,
        "matches_per_entity_mean": float(n_match.mean()) if len(n_match) else 0.0,
        "matches_per_matched_entity_mean": float(n_match[n_match > 0].mean()) if (n_match > 0).any() else 0.0,
        "matches_per_entity_max": int(n_match.max()) if len(n_match) else 0,
        "match_count_distribution": {int(k): int(v) for k, v in
                                     zip(*np.unique(np.minimum(n_match, 12), return_counts=True))},
        "cross_country_match_rate": float(cross / max(total, 1)),
        "target_multi_claim_rate": float(multi_claim / max(len(claims), 1)),
        "translit_rate_source1": float(s1["translit"].mean()),
        "translit_rate_targets": float(t["translit"].mean()),
        "addr_missing_rate_source1": float(s1["addr_missing"].mean()),
        "addr_missing_rate_targets": float(t["addr_missing"].mean()),
    }


# ----------------------------------------------------------------------------
# blocking helpers
# ----------------------------------------------------------------------------
def choose_k(cand, gt: GroundTruth, universe, cfg, n_targets):
    curve = []
    for k in range(1, cfg.k_max + 1):
        m = cand["rank_src"].to_numpy() < k
        st = blocking_stats(gt, universe, cand["s1_row"].to_numpy()[m], cand["t_row"].to_numpy()[m], n_targets)
        curve.append({"k": k, **st})
    if cfg.k_final != "auto":
        return int(cfg.k_final), curve
    best = curve[-1]["pair_recall"]
    target = best - cfg.k_recall_tolerance
    for row in curve:
        if row["pair_recall"] >= target:
            return row["k"], curve
    return cfg.k_max, curve


def finalize_candidates(cand: pd.DataFrame, k: int) -> pd.DataFrame:
    c = cand[cand["rank_src"] < k].reset_index(drop=True)
    c["is_s3"] = (c["t_source"] == 3).astype(np.float32)
    c = B.add_competition_features(c, "rerank", "blk")
    c = B.add_group_features(c, "rerank", "blk")
    return c


def competition_sample(cand: pd.DataFrame, n_s1: int, cfg: Config, rng) -> np.ndarray:
    """Random seed entities + every entity competing with them for a target."""
    n_seed = min(cfg.train_seed_entities, n_s1)
    seeds = rng.choice(n_s1, size=n_seed, replace=False)
    seed_mask = np.zeros(n_s1, dtype=bool)
    seed_mask[seeds] = True
    s1 = cand["s1_row"].to_numpy()
    t = cand["t_row"].to_numpy()
    seed_targets = np.unique(t[seed_mask[s1]])
    competitors = np.unique(s1[np.isin(t, seed_targets)])
    extra = np.setdiff1d(competitors, seeds)
    budget = max(0, cfg.train_max_entities - n_seed)
    if len(extra) > budget:
        extra = rng.choice(extra, size=budget, replace=False)
    return np.sort(np.concatenate([seeds, extra]))


# ----------------------------------------------------------------------------
# main entry points
# ----------------------------------------------------------------------------
def train(cfg: Config, tracker: Tracker) -> dict:
    log = tracker.log
    rng = np.random.default_rng(cfg.seed)
    with tracker.stage("load+normalise train"):
        s1, t = load_split(cfg, "train", tracker)
        gt_dict = read_ground_truth(cfg.data_dir)
    with tracker.stage("profile train"):
        prof = profile_training(s1, t, gt_dict)
        tracker.record("train_profile", prof)
        log.info("singleton rate %.3f | mean matches (matched) %.2f | cross-country %.5f | multi-claim %.5f",
                 prof["singleton_rate"], prof["matches_per_matched_entity_mean"],
                 prof["cross_country_match_rate"], prof["target_multi_claim_rate"])
    scope_country = prof["cross_country_match_rate"] <= 0.001
    if cfg.enforce_exclusivity == "on":
        ex_opts = (True,)
    elif cfg.enforce_exclusivity == "off":
        ex_opts = (False,)
    else:
        ex_opts = (False, True) if prof["target_multi_claim_rate"] <= cfg.exclusivity_max_violation else (False,)
    gt = GroundTruth(s1["entity_id"].tolist(), t["entity_id"].tolist(), gt_dict)
    all_s1 = np.arange(len(s1))

    with tracker.stage("blocking train"):
        cand, bstats = B.generate_candidates(s1, t, cfg, scope_country, log)
        k, curve = choose_k(cand, gt, all_s1, cfg, len(t))
        tracker.record("train_k_curve", curve)
        cand = finalize_candidates(cand, k)
        pre = blocking_stats(gt, all_s1, cand["s1_row"].to_numpy(), cand["t_row"].to_numpy(), len(t))
        log.info("K=%d | pair recall %.4f | oracle F0.5 %.4f | %.2f cands/S1 (before pruning)",
                 k, pre["pair_recall"], pre["oracle_macro_f05"], pre["avg_candidates_per_s1"])

    pruner_model, prune_thr, prune_stats = None, None, None
    if cfg.use_pruner:
        with tracker.stage("learned pruner"):
            y_c = gt.is_true(cand["s1_row"].to_numpy(), cand["t_row"].to_numpy()).astype(np.int8)
            p0, pruner_model, prune_thr, prune_stats = fit_pruner(cand, y_c, cfg, log)
            cand["p0"] = p0.astype(np.float32)
            cand = cand[p0 >= prune_thr].reset_index(drop=True)

    with tracker.stage("blocking diagnostics"):
        bst = blocking_stats(gt, all_s1, cand["s1_row"].to_numpy(), cand["t_row"].to_numpy(), len(t))
        bst.update(bstats)
        bst["k_per_source"] = k
        bst["country_scoped_keys"] = scope_country
        bst["before_pruning"] = pre
        bst["pruner"] = prune_stats
        tracker.record("train_blocking", bst)
        log.info("final candidates | pair recall %.4f | oracle F0.5 %.4f | %.2f cands/S1 | RR %.8f",
                 bst["pair_recall"], bst["oracle_macro_f05"], bst["avg_candidates_per_s1"],
                 bst["reduction_ratio"])

    with tracker.stage("training sample"):
        sample = competition_sample(cand, len(s1), cfg, rng)
        in_sample = np.zeros(len(s1), dtype=bool)
        in_sample[sample] = True
        cs = cand[in_sample[cand["s1_row"].to_numpy()]].reset_index(drop=True)
        tracker.record("train_sample", {"entities": int(len(sample)), "pairs": int(len(cs))})
        log.info("training sample: %d entities, %d pairs", len(sample), len(cs))

    with tracker.stage("features train"):
        fb = FeatureBuilder(s1, t, cfg.n_jobs)
        X1 = build_features(fb, cs, cfg.feature_batch_pairs, log)
        y = gt.is_true(cs["s1_row"].to_numpy(), cs["t_row"].to_numpy()).astype(np.int8)
        groups = cs["s1_row"].to_numpy()
        tracker.record("train_pairs_positive_rate", float(y.mean()) if len(y) else 0.0)

    def score_fn(s1r, tr):
        def f(sel):
            return macro_f05(gt, sample, s1r[sel], tr[sel])
        return f

    s1r, tr = cs["s1_row"].to_numpy(), cs["t_row"].to_numpy()
    with tracker.stage("stage-1 model"):
        oof1, m1, info1 = train_oof(X1, y, groups, cfg, log, "stage1")
        pol1, f1, _ = tune_policy(score_fn(s1r, tr), s1r, tr, oof1, ex_opts)
        tracker.record("stage1", {**info1, "oof_best_policy": pol1, "oof_macro_f05": f1})
        log.info("stage-1 OOF macro F0.5 = %.5f with %s", f1, pol1)

    with tracker.stage("stage-2 model"):
        ctx = add_context(cs, oof1, fb)
        X2 = pd.concat([X1, ctx], axis=1)
        oof2, m2, info2 = train_oof(X2, y, groups, cfg, log, "stage2")
        pol2, f2, table = tune_policy(score_fn(s1r, tr), s1r, tr, oof2, ex_opts)
        log.info("stage-2 OOF macro F0.5 = %.5f with %s", f2, pol2)

    use_stage2 = f2 >= f1
    policy = pol2 if use_stage2 else pol1
    final_p = oof2 if use_stage2 else oof1
    sel = apply_policy(policy, s1r, tr, final_p)
    oof_metrics = macro_f05(gt, sample, s1r[sel], tr[sel])
    sample_block = blocking_stats(gt, sample, s1r, tr, len(t))
    tracker.record("validation", {"use_stage2": bool(use_stage2), "policy": policy,
                                  "stage1_macro_f05": f1, "stage2_macro_f05": f2,
                                  "oof": oof_metrics, "sample_blocking": sample_block})
    tracker.record("error_analysis", error_breakdown(gt, sample, s1, s1r[sel], tr[sel]))
    tracker.record("error_patterns", error_patterns(X1, y, sel, gt, sample, cs))
    dump_error_examples(os.path.join(cfg.work_dir, "error_examples.tsv"), s1, t, cs, y, sel, final_p)
    pd.DataFrame(table).sort_values("macro_f05", ascending=False).to_csv(
        os.path.join(cfg.work_dir, "policy_search.csv"), index=False)
    importance(m2 if use_stage2 else m1).to_csv(os.path.join(cfg.work_dir, "feature_importance.csv"), index=False)

    mdir = os.path.join(cfg.work_dir, "models")
    os.makedirs(mdir, exist_ok=True)
    if pruner_model is not None:
        pruner_model.save_model(os.path.join(mdir, "pruner.txt"))
    m1.save_model(os.path.join(mdir, "stage1.txt"))
    m2.save_model(os.path.join(mdir, "stage2.txt"))
    artifacts = {"k": int(k), "scope_country": bool(scope_country), "policy": policy,
                 "prune_threshold": prune_thr,
                 "use_stage2": bool(use_stage2), "stage1_features": list(X1.columns),
                 "stage2_features": list(X2.columns)}
    with open(os.path.join(cfg.work_dir, "artifacts.json"), "w") as fh:
        json.dump(artifacts, fh, indent=2)
    return artifacts


def predict(cfg: Config, tracker: Tracker, artifacts: dict | None = None) -> dict:
    log = tracker.log
    if artifacts is None:
        with open(os.path.join(cfg.work_dir, "artifacts.json")) as fh:
            artifacts = json.load(fh)
    mdir = os.path.join(cfg.work_dir, "models")
    m1 = lgb.Booster(model_file=os.path.join(mdir, "stage1.txt"))
    m2 = lgb.Booster(model_file=os.path.join(mdir, "stage2.txt"))
    k = artifacts["k"]

    with tracker.stage("load+normalise test"):
        s1, t = load_split(cfg, "test", tracker)
        tracker.record("test_profile", {
            "n_source1": int(len(s1)), "n_targets": int(len(t)),
            "countries_source1": s1["country"].value_counts().to_dict(),
            "translit_rate_source1": float(s1["translit"].mean()),
        })

    with tracker.stage("blocking test"):
        cfg_k = Config(**{**cfg.__dict__, "k_max": k})
        cand, bstats = B.generate_candidates(s1, t, cfg_k, artifacts["scope_country"], log)
        cand = finalize_candidates(cand, k)
        n_before = len(cand)
        if artifacts.get("prune_threshold") is not None:
            pruner = lgb.Booster(model_file=os.path.join(mdir, "pruner.txt"))
            p0 = (pruner.predict(cand[B_FEATS].astype(np.float32)) if len(cand)
                  else np.zeros(0)).astype(np.float32)
            cand["p0"] = p0
            cand = cand[p0 >= artifacts["prune_threshold"]].reset_index(drop=True)
        bst = blocking_stats(None, np.arange(len(s1)), cand["s1_row"].to_numpy(), cand["t_row"].to_numpy(), len(t))
        bst.update(bstats)
        bst["pairs_before_pruning"] = int(n_before)
        tracker.record("test_blocking", bst)
        log.info("test candidates: %d pairs, %.2f per S1, RR %.8f", bst["candidate_pairs"],
                 bst["avg_candidates_per_s1"], bst["reduction_ratio"])

    s1_ids = s1["entity_id"].to_numpy(object)
    t_ids = t["entity_id"].to_numpy(object)
    os.makedirs(cfg.out_dir, exist_ok=True)
    cand_lists = _lists_per_s1(len(s1), cand["s1_row"].to_numpy(), cand["t_row"].to_numpy(),
                               cand["rerank"].to_numpy(), t_ids)
    cand_path = os.path.join(cfg.out_dir, "candidate_pairs.tsv")
    write_id_lists(cand_path, s1_ids, cand_lists, "candidate_entity_ids")

    with tracker.stage("features+stage-1 test"):
        fb = FeatureBuilder(s1, t, cfg.n_jobs)
        feat_dir = os.path.join(cfg.work_dir, "test_features")
        shutil.rmtree(feat_dir, ignore_errors=True)
        os.makedirs(feat_dir)
        p1 = np.zeros(len(cand), dtype=np.float64)
        bs = cfg.feature_batch_pairs
        for bi, s in enumerate(range(0, len(cand), bs)):
            e = min(len(cand), s + bs)
            Xb = build_features(fb, cand.iloc[s:e], bs)
            Xb = Xb[artifacts["stage1_features"]]
            p1[s:e] = m1.predict(Xb)
            Xb.to_parquet(os.path.join(feat_dir, f"part_{bi:05d}.parquet"), index=False)
            log.info("  test stage-1 %d/%d pairs", e, len(cand))

    with tracker.stage("stage-2 test"):
        if artifacts["use_stage2"]:
            ctx = add_context(cand, p1, fb)
            p = np.zeros(len(cand), dtype=np.float64)
            for bi, s in enumerate(range(0, len(cand), bs)):
                e = min(len(cand), s + bs)
                Xb = pd.read_parquet(os.path.join(feat_dir, f"part_{bi:05d}.parquet"))
                Xb = pd.concat([Xb, ctx.iloc[s:e].reset_index(drop=True)], axis=1)
                p[s:e] = m2.predict(Xb[artifacts["stage2_features"]])
        else:
            p = p1
        shutil.rmtree(feat_dir, ignore_errors=True)

    with tracker.stage("decode + write"):
        s1r, tr = cand["s1_row"].to_numpy(), cand["t_row"].to_numpy()
        sel = apply_policy(artifacts["policy"], s1r, tr, p)
        match_lists = _lists_per_s1(len(s1), s1r[sel], tr[sel], p[sel], t_ids)
        match_path = os.path.join(cfg.out_dir, "matching_results.tsv")
        write_id_lists(match_path, s1_ids, match_lists, "matched_entity_ids")
        n_pred = np.array([len(x) for x in match_lists])
        tracker.record("test_predictions", {
            "predicted_pairs": int(n_pred.sum()),
            "predicted_singleton_rate": float((n_pred == 0).mean()),
            "mean_matches_per_entity": float(n_pred.mean()),
        })
        pd.DataFrame({"s1": s1_ids[s1r], "t": t_ids[tr], "p": p.astype(np.float32)}).to_parquet(
            os.path.join(cfg.work_dir, "test_scores.parquet"), index=False)

    with tracker.stage("validate outputs"):
        test_dir = os.path.join(cfg.data_dir, "test")
        errors = validate(match_path, cand_path, test_dir)
        ran, passed, out = run_official(cfg.official_validator, match_path, cand_path, test_dir)
        tracker.record("validation_builtin", {"passed": not errors, "errors": errors[:20]})
        tracker.record("validation_official", {"ran": ran, "passed": passed, "output": out[-2000:]})
        if errors:
            log.error("built-in validator FAILED:\n%s", "\n".join(errors[:20]))
        else:
            log.info("built-in validator: PASS")
        if ran:
            log.info("official validator: %s", "PASS" if passed else "FAIL\n" + out)
    return {"matching": match_path, "candidate": cand_path, "errors": errors}


def error_breakdown(gt: GroundTruth, universe, s1: pd.DataFrame, ps, pt) -> dict:
    """Out-of-fold macro F0.5 split by record characteristics (for error analysis)."""
    out = {}
    views = {
        "country": s1["country"].to_numpy(object),
        "non_latin_script": np.where(s1["translit"].to_numpy() > 0, "yes", "no"),
        "address_missing": np.where(s1["addr_missing"].to_numpy() > 0, "yes", "no"),
        "true_match_count": np.where(gt.n_true == 0, "0 (singleton)",
                                     np.where(gt.n_true == 1, "1",
                                              np.where(gt.n_true <= 3, "2-3", "4+"))),
    }
    for vname, lab in views.items():
        rows = {}
        for val in pd.unique(lab[universe]):
            u = universe[lab[universe] == val]
            if len(u) == 0:
                continue
            r = macro_f05(gt, u, ps, pt)
            rows[str(val)] = {"entities": int(len(u)), "macro_f05": r["macro_f05"],
                              "precision": r["macro_precision"], "recall": r["macro_recall"]}
        out[vname] = rows
    return out


def error_patterns(X: pd.DataFrame, y: np.ndarray, sel: np.ndarray, gt: GroundTruth, universe, cs) -> dict:
    """Measured characteristics of false positives and false negatives (out-of-fold)."""
    fp = sel & (y == 0)
    fn = (~sel) & (y == 1)
    either_missing = (X["addr_miss_q"].to_numpy() + X["addr_miss_t"].to_numpy()) > 0

    def share(mask, cond):
        return float(cond[mask].mean()) if mask.any() else float("nan")

    in_u = np.zeros(gt.n_s1, dtype=bool)
    in_u[universe] = True
    true_in_universe = int(in_u[gt.s1_rows].sum())
    return {
        "false_positive_pairs": int(fp.sum()),
        "fp_share_identical_core_name": share(fp, X["c_exact"].to_numpy() > 0),
        "fp_share_numeric_conflict": share(fp, X["num_conflict"].to_numpy() > 0),
        "fp_share_address_missing": share(fp, either_missing),
        "false_negative_pairs_in_candidates": int(fn.sum()),
        "fn_share_mixed_script": share(fn, X["translit_mix"].to_numpy() > 0),
        "fn_share_address_missing": share(fn, either_missing),
        "fn_share_low_name_similarity": share(fn, X["c_tset"].to_numpy() < 60),
        "true_pairs_missed_by_blocking": int(true_in_universe - int(y.sum())),
        "true_pairs_total": true_in_universe,
    }


def dump_error_examples(path, s1, t, cs, y, sel, p, n=40):
    fp = np.flatnonzero(sel & (y == 0))
    fn = np.flatnonzero((~sel) & (y == 1))
    fp = fp[np.argsort(-p[fp])][:n]
    fn = fn[np.argsort(p[fn])][:n]
    rows = []
    for kind, idx in (("false_positive", fp), ("false_negative", fn)):
        for i in idx:
            a, b = int(cs["s1_row"].iat[i]), int(cs["t_row"].iat[i])
            rows.append({"type": kind, "prob": round(float(p[i]), 4),
                         "s1_id": s1["entity_id"].iat[a], "s1_name": s1["business_name"].iat[a],
                         "s1_address": s1["business_address"].iat[a],
                         "cand_id": t["entity_id"].iat[b], "cand_name": t["business_name"].iat[b],
                         "cand_address": t["business_address"].iat[b]})
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _lists_per_s1(n_s1, s1_rows, t_rows, score, t_ids):
    """Group target ids per S1 row, ordered by score (descending), de-duplicated."""
    lists = [[] for _ in range(n_s1)]
    if len(s1_rows) == 0:
        return lists
    order = np.lexsort((-np.asarray(score), s1_rows))
    seen_prev = None
    for r, tt in zip(s1_rows[order], t_rows[order]):
        key = (r, tt)
        if key == seen_prev:
            continue
        seen_prev = key
        lists[r].append(t_ids[tt])
    return [list(dict.fromkeys(x)) for x in lists]
