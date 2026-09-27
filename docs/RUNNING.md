# Business Entity Resolution — Run Guide

End-to-end, fully offline pipeline that links every Source 1 business to its matching Source 2 and Source 3 records:

**normalisation → multi-channel blocking → learned pruner → two-stage LightGBM matcher → macro-F0.5-optimal set decoder**

It produces both `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the raw challenge data with a single command. The candidate file is exactly the set of pairs the matcher runs inference on, so every match is a subset of the candidates.

## 1. Environment

* Python 3.11 or 3.12 on Linux or macOS. Windows works too; use WSL if multiprocessing misbehaves.
* Dependencies are pinned in `requirements.txt`. All of them are permissively licensed: LightGBM (MIT), RapidFuzz (MIT), anyascii (ISC), scikit-learn/NumPy/SciPy/pandas (BSD), PyArrow (Apache 2.0).
* No GPU, no internet access and no external data are needed at any point.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt   # Python 3.11/3.12
# Python 3.9/3.10: pip install -r requirements-compat.txt  (verified with the same test-suite)
```

**Recommended hardware for the full challenge data** (~12M training and ~12M test records): 16+ vCPUs and 64 GB RAM, e.g. AWS `r6i.4xlarge` / `m6i.8xlarge`. With 32 GB RAM, lower the training sample with `--train-max-entities 200000 --train-seed-entities 60000`.

## 2. Data layout

```
<root>/
├── dataset/
│   ├── train/ train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
│   └── test/  test_source1.tsv   test_source2.tsv   test_source3.tsv
└── utils/validate_submission.py      (official validator; auto-detected if present)
```

## 3. Reproduce everything (one command)

Run this from `<root>` (the folder that contains `dataset/`):

```bash
python code/business_entity_resolution/src/main.py run \
    --data-dir dataset --out-dir output --work-dir work
```

The command runs these steps in order:

1. Normalises all sources and caches them in `work/*.parquet`.
2. Profiles the training data. Country scoping of blocking keys and the exclusivity constraint are decided from the measured statistics.
3. Blocks all training entities, picks K from the recall curve, and trains the learned pruner out-of-fold.
4. Trains the stage-1 and stage-2 LightGBM matchers with entity-grouped out-of-fold validation, then selects the decoder on out-of-fold macro F0.5.
5. Blocks and prunes the test split and writes `output/candidate_pairs.tsv`.
6. Scores the test split and writes `output/matching_results.tsv`.
7. Validates both files with the built-in checker, and with `utils/validate_submission.py` when it is present.
8. Writes the methodology document, with all numbers measured in this run, to `work/Documentation_template.md`.
9. Builds `submission.zip` in the required structure.

The individual steps can also be run separately:

```bash
python code/business_entity_resolution/src/main.py train   --data-dir dataset --work-dir work
python code/business_entity_resolution/src/main.py predict --data-dir dataset --work-dir work --out-dir output
python code/business_entity_resolution/src/main.py report  --data-dir dataset --work-dir work
python code/business_entity_resolution/src/main.py package --data-dir dataset --work-dir work --out-dir output
```

Validate independently:

```bash
python code/business_entity_resolution/src/ber/validate.py \
    --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
python utils/validate_submission.py \
    --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## 4. Outputs and run artefacts

| File | Content |
|---|---|
| `output/matching_results.tsv` | final matches, one row per test Source 1 entity |
| `output/candidate_pairs.tsv` | final candidate set (post-pruning), the exact matcher inference set |
| `work/run_log.json` | every measured metric, stage timings and peak memory |
| `work/run.log` | human-readable log |
| `work/config.json` | full resolved configuration of the run |
| `work/artifacts.json`, `work/models/` | chosen K, pruner threshold, decoder policy, LightGBM models |
| `work/feature_importance.csv`, `work/policy_search.csv` | model diagnostics |

## 5. Determinism

All randomness (sampling, folds, LightGBM) is seeded (`seed=42`), and LightGBM runs with `deterministic=True`. Re-running with the same data and configuration reproduces the same outputs.

## 6. Code map

| Module | Responsibility |
|---|---|
| `src/main.py` | CLI |
| `src/ber/config.py` | all hyper-parameters |
| `src/ber/io_utils.py` | TSV reading/writing (tab-separated, no quoting, empty strings preserved) |
| `src/ber/normalize.py` | transliteration, abbreviation expansion, legal suffixes, phonetic skeleton, numeric anchors |
| `src/ber/blocking.py` | hashed multi-channel keys, inverted index, IDF voting, rerank, competition/group features |
| `src/ber/pruner.py` | learned last-stage candidate pruner |
| `src/ber/features.py` | vectorised pair features |
| `src/ber/model.py` | LightGBM with entity-grouped out-of-fold training |
| `src/ber/context.py` | stage-2 collective features |
| `src/ber/decision.py` | expected-F0.5 and threshold decoders, exclusivity |
| `src/ber/metrics.py` | exact macro F0.5, recall ceiling, reduction ratio |
| `src/ber/validate.py` | independent submission checker |
| `src/ber/report.py` + `ber/assets/` | methodology document generation |
| `src/tools/` | packaging, scoring, synthetic data for tests |
| `src/tests/` | unit and end-to-end tests (`python -m pytest src/tests`) |

## 7. Compliance

* No external databases, APIs, geocoders or internet data are used. All abbreviation tables are generic linguistic rules written into the code.
* The country label is treated as an open set of strings. It is never enumerated, filtered or used as a model feature.
* The only learned models are LightGBM gradient-boosted trees (MIT licence, far below 8B parameters).
