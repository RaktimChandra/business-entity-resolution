"""Command-line entry point.

    python src/main.py run      --data-dir DATA --out-dir OUT --work-dir WORK
    python src/main.py train    ...   # fit models only
    python src/main.py predict  ...   # reuse saved models, write outputs
    python src/main.py report   ...   # rebuild the methodology document
    python src/main.py package  ...   # build the submission zip

``run`` = train + predict + report + package.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ber.config import Config  # noqa: E402
from ber.tracking import Tracker  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "ber", "assets", "methodology_template.md")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Business entity resolution pipeline")
    ap.add_argument("command", choices=["run", "train", "predict", "report", "package"])
    ap.add_argument("--data-dir", default="dataset", help="folder containing train/ and test/")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--train-seed-entities", type=int, default=None)
    ap.add_argument("--train-max-entities", type=int, default=None)
    ap.add_argument("--k-final", default=None, help="'auto' or an integer")
    ap.add_argument("--k-max", type=int, default=None)
    ap.add_argument("--key-df-cap", type=int, default=None)
    ap.add_argument("--k-wide", type=int, default=None, help="candidates kept per (S1, source) after the key vote")
    ap.add_argument("--block-batch", type=int, default=None, help="S1 entities per blocking batch (memory knob)")
    ap.add_argument("--num-leaves", type=int, default=None)
    ap.add_argument("--n-folds", type=int, default=None)
    ap.add_argument("--lgb-rounds", type=int, default=None)
    ap.add_argument("--official-validator", default=None,
                    help="path to utils/validate_submission.py (auto-detected next to the data folder)")
    ap.add_argument("--doc-out", default=None, help="where to write the filled methodology document")
    ap.add_argument("--zip", default=None, help="path of the submission zip to create")
    ap.add_argument("--code-dir", default=os.path.dirname(HERE),
                    help="business_entity_resolution folder to package")
    return ap.parse_args(argv)


def make_config(a) -> Config:
    cfg = Config(data_dir=a.data_dir, out_dir=a.out_dir, work_dir=a.work_dir)
    for arg, field in [("n_jobs", "n_jobs"), ("train_seed_entities", "train_seed_entities"),
                       ("train_max_entities", "train_max_entities"), ("k_max", "k_max"),
                       ("key_df_cap", "key_df_cap"), ("n_folds", "n_folds"), ("lgb_rounds", "lgb_rounds"),
                       ("k_wide", "k_wide"), ("block_batch", "block_batch"), ("num_leaves", "lgb_num_leaves")]:
        v = getattr(a, arg)
        if v is not None:
            setattr(cfg, field, v)
    if a.k_final is not None:
        cfg.k_final = a.k_final if a.k_final == "auto" else int(a.k_final)
    if a.official_validator:
        cfg.official_validator = a.official_validator
    else:
        guess = os.path.join(os.path.dirname(os.path.abspath(a.data_dir)), "utils", "validate_submission.py")
        cfg.official_validator = guess
    return cfg


def main(argv=None):
    a = parse_args(argv)
    cfg = make_config(a)
    os.makedirs(cfg.work_dir, exist_ok=True)
    tracker = Tracker(cfg.work_dir)
    cfg.to_json(os.path.join(cfg.work_dir, "config.json"))
    doc_out = a.doc_out or os.path.join(cfg.work_dir, "Documentation_template.md")

    from ber import pipeline, report
    from tools.package_submission import build_zip

    artifacts = None
    if a.command in ("run", "train"):
        artifacts = pipeline.train(cfg, tracker)
    if a.command in ("run", "predict"):
        res = pipeline.predict(cfg, tracker, artifacts)
        if res["errors"]:
            tracker.log.error("outputs failed validation; not packaging")
            return 1
    if a.command in ("run", "report"):
        report.build_document(cfg, TEMPLATE, doc_out)
        tracker.log.info("methodology document written to %s", doc_out)
    if a.command in ("run", "package"):
        zip_path = a.zip or os.path.join(os.path.dirname(os.path.abspath(cfg.out_dir)),
                                         cfg.team_name.replace(" ", "_") + "_submission.zip")
        build_zip(zip_path, cfg.out_dir, a.code_dir, doc_out, cfg.data_dir)
        tracker.log.info("submission package: %s", zip_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
