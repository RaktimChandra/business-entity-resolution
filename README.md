# Business Entity Resolution at Scale

[![CI](https://github.com/RaktimChandra/business-entity-resolution/actions/workflows/ci.yml/badge.svg)](https://github.com/RaktimChandra/business-entity-resolution/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A CPU-only, fully offline pipeline that links **1.7M business records** to their duplicates in a pool of **10M+ noisy records**. The records span the US, India and France, and many names are written in Devanagari, Bengali, Tamil and other Indic scripts. It was built for the **Amazon ML Challenge 2026** (business entity resolution task), and the score is macro **F0.5** per entity.

| | Round 1 | Round 2 | Round 3 |
|---|---|---|---|
| Public leaderboard (macro F0.5) | 0.9141 | **0.9628** | — (finished after the portal closed) |
| Out-of-fold validation F0.5 | 0.8594 | 0.9553 | **0.9617** |
| Blocking recall (final candidates) | 87.50% | 94.97% | **95.59%** |
| Candidates per entity (test) | 5.73 | 6.53 | 6.87 |
| Precision (OOF) | 0.993 | 0.995 | 0.995 |

Details for each round are in [`experiments/`](experiments/), and the full write-up is in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

---

## Architecture

```
raw TSV (S1, S2, S3)
  │
  ├─ Normalisation ─── transliteration (Indic → Latin) · abbreviation & state-code folding
  │                    legal-suffix stripping · phonetic + vowel-free skeleton · numeric anchors
  ├─ Blocking ──────── 12 hashed key channels → inverted index over 10M targets
  │                    IDF-weighted vote → two-path rerank (name path / address path) → top-K
  ├─ Learned pruner ── LightGBM on blocking signals: ~5× fewer pairs, 99.6% of true pairs kept
  ├─ Stage-1 matcher ─ ~85 vectorised pair features → LightGBM (entity-grouped OOF)
  ├─ Stage-2 matcher ─ collective features: competition, group context, cluster support
  └─ Decoder ──────── closed-form expected-F0.5 set selection (+ exclusivity), tuned on OOF
```

## Key ideas

1. **Cascade blocking with a learned pruner.** High recall first, then a cheap model removes about 80% of the pairs. Only about 6 candidates per entity reach the matcher, a reduction ratio of 99.99993%.
2. **Diagnosis-driven iteration.** A blocking diagnostic (`src/tools/diagnose_blocking.py`) sorts every lost true pair by cause. It showed that 12% of matches were lost because their *name* was unusable (another script, a placeholder, a social handle) even though the address was identical. An address-evidence rerank path and house-number × street keys fixed this: **+0.049 on the leaderboard in one iteration**.
3. **Decoy-aware features.** The data contains near-identical decoys at the same address ("Program" vs "Program **Co**"). Raw name-variation features separate a *different registered entity* from *formatting noise* ("& Co", "[Inc]", "(Services)").
4. **A metric-optimal decoder.** Per entity, E[F0.5_k] ≈ 1.25·S_k / (0.25·S/(1−P₀) + k). The set size, including "no match", comes from a single objective.
5. **Leakage-free validation.** Folds are grouped by entity, stage-2 features are built from out-of-fold stage-1 predictions, and every threshold is tuned on OOF only.

## Engineering

- **Scale:** 12.5M training and 11.7M test records on a 16-thread laptop with 28 GB RAM. A full run takes about 2.5 h.
- **Performance:**
  - Blocking is linear in the number of queries: sorted `uint64` inverted index, vectorised `bincount` voting, capped key frequencies.
  - Decoder search was cut from **57 min to seconds** by precomputing pair correctness once (`FastScorer`).
- **Reproducibility:**
  - One CLI command runs everything.
  - Seeded, deterministic LightGBM.
  - Pinned dependencies, with a compatibility set for Python 3.9/3.10.
- **Observability:** every stage logs wall time, peak RSS and metrics to `run_log.json`, and the methodology document is generated from those measured numbers.
- **Quality:**
  - Unit and end-to-end tests on a synthetic generator that mimics the noise patterns.
  - CI on every push.
  - An independent validator mirrors the submission rules.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q src/tests                      # includes an end-to-end run on synthetic data

# full pipeline (data not included: competition data cannot be redistributed)
python src/main.py run --data-dir dataset --out-dir output --work-dir work
python src/tools/diagnose_blocking.py --work-dir work --data-dir dataset --sample 20000
```

## Repository layout

```
src/ber/          normalize · blocking · pruner · features · model · context · decision · metrics · pipeline · report
src/tools/        synthetic data generator · scorer · blocking diagnostic · packager
src/tests/        unit + end-to-end tests
experiments/      per-round configs, measured metrics, feature importance, notes
docs/             methodology write-up generated from the final run
```

## Tech

Python · NumPy · pandas · SciPy · scikit-learn · LightGBM · RapidFuzz · anyascii · PyArrow · GitHub Actions

---

*The competition data is not included, in line with the challenge terms. All numbers above were measured on the official data.*
