"""Central configuration for the entity resolution pipeline.

Every tunable lives here so that runs are reproducible from a single object,
and the resolved config is written next to the outputs of every run.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field


@dataclass
class Config:
    # ---- paths -----------------------------------------------------------
    data_dir: str = "dataset"
    out_dir: str = "output"
    work_dir: str = "work"

    # ---- compute ---------------------------------------------------------
    n_jobs: int = max(1, (os.cpu_count() or 2))
    seed: int = 42

    # ---- blocking --------------------------------------------------------
    # Keys whose document frequency among targets exceeds this cap are
    # considered non-discriminative and are not used to generate pairs.
    key_df_cap: int = 1000
    # Number of source-1 entities processed per blocking batch.
    block_batch: int = 10000
    # Candidates kept per (S1 entity, target source) after key scoring.
    k_wide: int = 80
    # Final candidates per (S1 entity, target source) fed to the matcher.
    # "auto" selects the smallest K that keeps (1 - k_recall_tolerance) of the
    # recall available at k_wide, measured on the training data.
    k_final: str | int = "auto"
    k_recall_tolerance: float = 0.002
    k_max: int = 20
    # Rerank candidates with a relative score threshold: a candidate is dropped
    # when its rerank score is below (best score in its group - rerank_margin).
    rerank_margin: float | None = None

    # ---- learned pruner (last blocking stage) ----------------------------
    use_pruner: bool = True
    prune_tolerance: float = 0.004     # max share of true pairs the pruner may drop
    prune_max_threshold: float = 0.5   # safety: never prune pairs scored >= this
    pruner_rounds: int = 300
    pruner_max_rows: int = 3_000_000

    # ---- training --------------------------------------------------------
    # Number of seed S1 entities used to train the matcher. The sample is
    # expanded with every S1 entity competing for the same targets so that
    # competition features see realistic conflicts.
    train_seed_entities: int = 200000
    train_max_entities: int = 700000
    n_folds: int = 4
    lgb_rounds: int = 1500
    lgb_early_stopping: int = 75
    lgb_learning_rate: float = 0.05
    lgb_num_leaves: int = 255
    lgb_min_data_in_leaf: int = 60
    lgb_feature_fraction: float = 0.8
    lgb_bagging_fraction: float = 0.8

    # ---- feature batch ---------------------------------------------------
    feature_batch_pairs: int = 1_000_000

    # ---- decision --------------------------------------------------------
    enforce_exclusivity: str = "auto"  # "auto" | "on" | "off"
    exclusivity_max_violation: float = 0.01

    # ---- misc ------------------------------------------------------------
    team_name: str = "Raktim Chandra"
    team_members: list = field(default_factory=lambda: ["Raktim Chandra"])
    official_validator: str = "utils/validate_submission.py"

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2, default=str)
