# ML Challenge 2026: Business Entity Resolution Solution

**Author:** Raktim Chandra
**Submission Date:** September 27, 2026

---

## 1. Executive Summary

We link Source 1 businesses to their Source 2 and Source 3 records with a cascade:
- script-agnostic normalization,
- multi-channel IDF-weighted blocking,
- a learned candidate pruner,
- a two-stage LightGBM matcher that reasons collectively about competing entities.

The pruner leaves **5.71 candidates per entity** while keeping 95.59% of true pairs. A decoder then maximizes expected F0.5 per entity, which reaches **macro F0.5 = 0.9617** (precision 0.9957) on leakage-free, entity-grouped validation. The system uses no external data or country-specific rules; its only learned models are LightGBM ensembles.

---

## 2. Methodology

### 2.1 Problem Analysis

We profiled the full training split before designing anything. Every number below is measured by `ber/pipeline.py::profile_training`.

| Property | Value |
|---|---|
| Source 1 records (train) | 2,206,821 |
| Target records (train): Source 2 / Source 3 | 5,034,616 / 5,285,603 |
| Singleton rate (Source 1 entities with no match) | 5.58% |
| Mean matches per matched entity / maximum | 3.67 / 11 |
| True matches that cross country labels | 0.0000% |
| Target records claimed by more than one Source 1 entity | 0.0000% |
| Records in a non-Latin script (Source 1 / targets) | 0.00% / 14.75% |
| Records with a missing address (Source 1 / targets) | 0.00% / 3.34% |

These measurements led to the following design decisions. One more finding shaped the matcher: **legal-suffix decoys.** A name with a plain legal suffix added (e.g. "Colonial Program" vs "Colonial Program Co") at the *same* address is often a different registered entity. The same name with decorative variation ("& Co", "[Inc]", "(Services)") is formatting noise of the same entity.

* **The metric is set-valued and averaged per entity.** Per entity, F0.5 = 1.25·TP / (0.25·|truth| + |predicted|). One false match on a single-match entity drops its score from 1.0 to 0.56, while one missed match on a two-match entity only drops it to 0.83. Singletons score exactly 1 or 0. We therefore treat each entity as a *set prediction* problem rather than as independent pair classifications (Section 4.3).
* **Country as a partition.** The measured share of true matches that cross country labels is 0.0000%, so the pipeline decides automatically whether to scope blocking keys by the country *string* (decision for this run: yes). Country is never enumerated, one-hot encoded or used as a model feature, so the test-only country goes through exactly the same code path.
* **Source 1 is deduplicated.** Only 0.0000% of target records belong to more than one Source 1 entity, so competition between Source 1 entities for the same target is informative. We exploit this in blocking features, stage-2 features and an optional exclusivity constraint.
* **Names change script, or disappear.** 0.00% of Source 1 records and 14.75% of target records are written in a non-Latin script (Devanagari, Bengali, Tamil, Gujarati, Odia and others). Many target records also carry a placeholder, social handle or domain instead of the business name, while their address is intact. Transliteration alone does not make "एसएस फूड प्राइवेट लिमिटेड" equal to "SS Food Private Limited", so numeric anchors (house numbers, phone digits, postal codes) and phonetic skeletons provide the missing bridge.

### 2.2 Solution Strategy

```
 raw TSV (S1, S2, S3)
        │
        ▼
 [1] Normalisation ── transliteration · abbreviation expansion · legal-suffix stripping
        │              phonetic skeleton · numeric anchors · alphanumeric codes
        ▼
 [2] Blocking ─────── 12 hashed key channels → inverted index over S2∪S3
        │              IDF-weighted vote → top-120 → cheap rerank → top-16 per source
        ▼
 [2b] Learned pruner ─ LightGBM on blocking signals → keeps 17.86% of pairs
        │               and 99.60% of true pairs → candidate_pairs.tsv
        ▼
 [3] Stage-1 matcher ─ 74 pair features → LightGBM (entity-grouped OOF)
        │
        ▼
 [4] Stage-2 matcher ─ + competition / group-context / cluster-support features → LightGBM
        │
        ▼
 [5] Decoder ──────── expected-F0.5 or threshold set selection, tuned on OOF macro F0.5
        │
        ▼
 matching_results.tsv  +  candidate_pairs.tsv (exact inference set)
```

**Approach Type:** Hybrid. Multi-channel blocking, a learned pruner, a two-stage gradient-boosted classifier, and metric-optimal set decoding.

**Core Innovation:**
1. **Cascade blocking with a learned pruner.** This makes the candidate set small, which the organisers rank explicitly, while keeping almost all recall.
2. **Collective second stage.** Pairwise probabilities are revised using how strongly other entities claim the same record, where the candidate stands within its own entity's group, and whether it resembles the entity's strongest candidate.
3. **Closed-form expected-F0.5 decoder.** It selects each entity's match set, and decides "no match" (singletons), from a single objective.
4. **Competition-preserving training sample.**
5. **Decoy-aware name-variation features.** They separate a different registered entity at the same address from formatting noise of the same entity. Random seed entities are augmented with every entity that competes with them for a record, so competition features have the same distribution in training as at inference.

---

## 3. Candidate Generation (Blocking)

**Blocking keys used.** Each record emits hashed keys from the channels below, and every key is prefixed with the record's country label. A pair's blocking score is Σ weight(channel) × log(1 + N / df(key)) over the keys the two records share.

| Channel | Weight |
|---|---|
| `name_exact` | 3.0 |
| `name_token` | 1.0 |
| `name_prefix5` | 1.0 |
| `name_phonetic_token` | 0.6 |
| `name_bigram` | 1.5 |
| `number` | 1.0 |
| `alnum_code` | 1.5 |
| `addr_bigram` | 1.0 |
| `name_plus_number` | 2.0 |
| `phonetic_exact` | 2.0 |
| `addr_token` | 0.4 |
| `number_plus_street` | 2.0 |

**Scalability.** Keys are 64-bit hashes stored in a sorted inverted index over all Source 2 and Source 3 records. Keys shared by more than 1500 targets are dropped as non-discriminative, which bounds the work done per query key. Votes are aggregated with vectorised sort and bincount operations in batches of Source 1 entities. The cost is therefore linear in the number of Source 1 records, and no all-pairs comparison happens anywhere.

**Two-level pruning.** For each (entity, target source) we keep the top 120 targets by vote, then rerank them with two evidence paths:
- a **name path**, which blends token-set similarity on core name and address with phonetic similarity;
- an **address path**, which combines address tokens with exact numeric anchors.

The stronger path wins. This keeps records whose name is unusable (other script, placeholder, handle) but whose address is identical. After reranking we keep the top **K = 16**. K is chosen automatically as the smallest value within 0.2 percentage points of the recall available at the maximum K. The recall curve on the full training split was:

| K per source | Pair recall | Oracle macro F0.5 | Candidates per S1 |
|---|---|---|---|
| 1 | 41.97% | 0.7615 | 2.00 |
| 2 | 67.70% | 0.8900 | 4.00 |
| 3 | 81.64% | 0.9384 | 6.00 |
| 4 | 88.52% | 0.9604 | 8.00 |
| 5 | 91.78% | 0.9712 | 10.00 |
| 6 | 93.43% | 0.9769 | 12.00 |
| 8 | 94.90% | 0.9818 | 16.00 |
| 10 | 95.43% | 0.9836 | 20.00 |
| 12 | 95.69% | 0.9845 | 24.00 |
| 15 | 95.92% | 0.9853 | 30.00 |
| 16 **(selected)** | 95.97% | 0.9855 | 32.00 |
| 20 | 96.14% | 0.9862 | 39.99 |

*Oracle macro F0.5 is the score a perfect matcher would reach on these candidates, i.e. the ceiling that blocking imposes.*

**Learned pruner (last blocking stage).** A LightGBM model is trained out-of-fold on blocking-time signals only: vote score, shared keys, rerank similarity, ranks and competition. It uses threshold 0.01333, the most aggressive threshold that loses at most 0.4% of the true pairs available after K-selection. It keeps 17.86% of the pairs and 99.60% of the true pairs, which cuts candidates per entity from 32.00 to 5.71. The pruner's score is also passed to the matcher as a feature.

| Training split, all entities | Pair recall | Oracle macro F0.5 | Candidates per S1 |
|---|---|---|---|
| After K-selection (K = 16 per source) | 95.97% | 0.9855 | 32.00 |
| **After learned pruner (final)** | **95.59%** | **0.9842** | **5.71** |

**Candidate pairs generated (final, exactly the set the matcher scores).**

| Split | Source 1 entities | Pairs before pruner | Final candidate pairs | Candidates per S1 | Reduction ratio |
|---|---|---|---|---|---|
| Train | 2,206,821 | 70,612,251 | 12,611,048 | 5.71 | 99.999945% |
| Test | 1,732,544 | 55,437,129 | 11,910,008 | 6.87 | 99.999931% |

**How true matches were not lost.** The channels are complementary:

* A typo in the name is caught by the numeric, address-bigram and phonetic channels.
* A name in a different script is caught by the vowel-free phonetic skeleton ("iunaited phuds" and "united foods" both become `ntd fds`), by house-number × street keys, and by alphanumeric codes such as `af0684`.
* A missing address is caught by the exact-name, name-bigram and name-plus-number channels.
* A generic name is caught by the address channels.

The IDF weighting keeps rare, informative keys dominant, so a large K is unnecessary. `candidate_pairs.tsv` is exactly the set the matcher runs inference on, and every predicted match is a subset of it.

---

## 4. Matching Model

### 4.1 Features Used

Stage 1 uses 74 features, computed in vectorised batches (RapidFuzz C++ `cpdist`, sparse row products). None of them uses the country label.

* **Name features.** On the normalized full name we compute Levenshtein ratio, token-set, token-sort, partial and Jaro-Winkler similarity. On the core name (legal suffixes removed) we compute ratio, token-set, partial token-set, Jaro-Winkler, exact equality and compact (space-free) ratio. On the phonetic skeleton we compute ratio and token-set. We also compute character-trigram cosine, IDF-weighted token Jaccard and shared IDF mass, first-token equality, an acronym test, and token counts and length ratio.
* **Name-variation features.** These describe exactly which raw tokens were added to, or removed from, the name: a plain legal suffix, a bracketed or ampersand variant, a generic word such as "Services" or "Center", a social handle or domain, or identical tokens. They separate legal-suffix decoys from formatting noise.
* **Address features.** We compute ratio, token-set, token-sort, partial ratio, character-trigram cosine, IDF-weighted token Jaccard and a combined name+address token-set.
* **Other: numeric anchors.** For house, postal and phone numbers we compute shared count, Jaccard and an explicit **conflict flag**, set when both records carry numbers but none agree. Postal codes and alphanumeric codes also get shared-count and conflict flags.
* **Other: data-quality flags.** Missing address on either side, non-Latin script on either side, and a mixed-script flag.
* **Other: blocking context.** Vote score, number of shared keys, rerank score, rank within source, the target's source, and how many other entities claim the same target together with this entity's rank and margin among them.
* **Other: stage-2 collective features**, 87 features in total. These are the stage-1 probability, its rank, gap and sum within the entity's group, the number of other strong candidates, competition for the target on the probability scale, and **cluster support**: the similarity of the candidate to the entity's strongest other candidate, weighted by that candidate's probability.

### 4.2 Model Type

* **LightGBM binary classifier** (gradient-boosted trees, MIT licence) with `num_leaves=255`, `learning_rate=0.05`, `min_data_in_leaf=60`, feature and bagging fraction 0.8, and L2 regularisation of 1.0.
* **Training data.** 700,000 Source 1 entities (4,786,406 candidate pairs, 46.93% positive). Every negative comes from our own blocking, so the model learns exactly the distinctions it must make at inference, e.g. the same brand at a different branch or a different number on the same street.
* **Validation.** 4-fold cross-validation with folds grouped by Source 1 entity, with early stopping on each fold. Stage-2 features are built from out-of-fold stage-1 predictions, so no label leaks into stage 2. The final models are refit on all sampled pairs with the averaged best iteration count (1126 rounds for stage 1).

### 4.3 Threshold Selection Method

We do not use a single global threshold. Instead, two set decoders are evaluated on out-of-fold probabilities against the exact competition metric, and the best one is kept.

* **Threshold decoder.** The top candidate is accepted if p ≥ t_top, and any further candidate if p ≥ t_rest.
* **Expected-F0.5 decoder.** For an entity with calibrated probabilities p₁ ≥ p₂ ≥ …, predicting the top k yields approximately E[F_k] = 1.25·S_k / (0.25·S/(1−P₀) + k). Here S_k is the sum of the top k probabilities, S the sum over all candidates, and P₀ = Π(1−pᵢ) the probability that the entity is a singleton. Predicting nothing scores P₀. The decoder picks the argmax, so singleton handling and multi-match set size come from one objective.

Both decoders are searched over their parameters, with and without a one-record-one-entity exclusivity constraint. The best configuration by out-of-fold macro F0.5 is the **expected-F0.5 decoder with calibration exponent alpha = 1.0 and empty-set bias = 1.2**.

| Decoder | Parameters | Exclusive | Macro F0.5 | Precision | Recall |
|---|---|---|---|---|---|
| threshold | t_top=0.5, t_rest=0.75 | True | 0.9616 | 0.9959 | 0.9040 |
| threshold | t_top=0.5, t_rest=0.75 | False | 0.9616 | 0.9959 | 0.9040 |
| threshold | t_top=0.55, t_rest=0.75 | True | 0.9616 | 0.9960 | 0.9039 |
| expected_f | alpha=1.0, empty_bias=1.2 | False | 0.9617 | 0.9957 | 0.9044 |
| expected_f | alpha=1.0, empty_bias=1.2 | True | 0.9617 | 0.9958 | 0.9043 |
| expected_f | alpha=1.15, empty_bias=1.0 | True | 0.9617 | 0.9961 | 0.9036 |

---

## 5. Results & Error Analysis

* **F0.5 score (macro, out-of-fold, entity-grouped):** **0.9617** (precision 0.9957, recall 0.9044).
* **Stage 1 alone:** 0.9604. **With the collective stage 2:** 0.9617. The submitted model uses stage 2.
* **Singletons:** F0.5 = 0.9820. **Entities with at least one match:** F0.5 = 0.9604.
* **Blocking ceiling on the validation sample** (perfect matcher on our candidates): 0.9749.
* **Test predictions:** 5,651,650 matched pairs, 6.09% of entities predicted as singletons, 3.26 matches per entity on average.

**Breakdown of out-of-fold performance:**

| View | Group | Entities | Macro F0.5 | Precision | Recall |
|---|---|---|---|---|---|
| country | India | 372,512 | 0.9506 | 0.9955 | 0.8809 |
| country | US | 327,488 | 0.9742 | 0.9961 | 0.9310 |
| true_match_count | 4+ | 332,452 | 0.9695 | 0.9982 | 0.9037 |
| true_match_count | 2-3 | 289,464 | 0.9594 | 0.9962 | 0.9061 |
| true_match_count | 0 (singleton) | 39,669 | 0.9820 | — | — |
| true_match_count | 1 | 38,415 | 0.8896 | 0.9892 | 0.8970 |

**False positives (wrong merges), measured on the validation sample.** There were 7,870 false-positive pairs:
* 34.23% have an *identical* core name, i.e. same-brand branches or chains that only the address can tell apart.
* 21.94% carry a numeric conflict (both records have numbers and none agree).
* 13.81% have a missing address on at least one side, which leaves the name as the only evidence.

**False negatives (missed matches), measured on the validation sample.**
* 162,695 of 2,409,025 true pairs never reach the matcher (blocking and pruner losses).
* Among the 68,214 true pairs that are candidates but were rejected:
  * 7.79% mix scripts (one record non-Latin),
  * 63.87% have a missing address,
  * 13.17% have core-name token-set similarity below 60.

  Given the precision-weighted metric, rejecting these low-evidence pairs is a deliberate trade-off.

The highest-confidence false positives and the lowest-confidence false negatives are listed in `work/error_examples.tsv`, which we used for manual review.

---

## 6. Conclusion

Macro F0.5 rewards systems that are selective at the level of whole entities. Our three main levers are complementary blocking channels with IDF weighting, which keep the candidate set small without losing matches; a collective second stage that exploits the structure of the problem; and a decoder that optimises the metric directly and handles singletons and match-set size in one objective. The pipeline is deterministic, runs on commodity CPUs, uses no external data, and regenerates both output files from the raw data with one command.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/src/`. **Entry point:** a single command regenerates both output files, validates them and rebuilds this document from the raw data:

```
pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/src/main.py run --data-dir dataset --out-dir output --work-dir work
```

| Path | Purpose |
|---|---|
| `main.py` | CLI: `run` / `train` / `predict` / `report` / `package` |
| `ber/normalize.py` | Transliteration, abbreviation expansion, legal suffixes, phonetic skeleton, numeric anchors |
| `ber/blocking.py` | Hashed multi-channel keys, inverted index, IDF vote, rerank, competition/group features |
| `ber/pruner.py` | Learned last-stage candidate pruner |
| `ber/features.py` | Vectorised pair features |
| `ber/model.py` | LightGBM with entity-grouped out-of-fold training |
| `ber/context.py` | Stage-2 collective features |
| `ber/decision.py` | Expected-F0.5 and threshold decoders, exclusivity |
| `ber/metrics.py` | Exact macro F0.5, blocking recall and oracle ceiling, reduction ratio |
| `ber/validate.py` | Submission checker mirroring every format rule |
| `ber/pipeline.py` | Orchestration, caching, profiling, error analysis |
| `ber/report.py` | Builds this document from measured metrics |
| `tests/` | Unit and end-to-end tests |

### B. Additional Results

**Feature importance (stage used for submission, share of total gain):**

| Rank | Feature | Gain share |
|---|---|---|
| 1 | `p1` | 86.8% |
| 2 | `p1_g_gap` | 10.4% |
| 3 | `p1_g_sum` | 1.6% |
| 4 | `sup_p` | 0.3% |
| 5 | `p1_t_margin` | 0.3% |
| 6 | `p1_t_n` | 0.1% |
| 7 | `p0` | 0.1% |
| 8 | `p1_g_n` | 0.1% |
| 9 | `blk_g_sum` | 0.1% |
| 10 | `sup_score` | 0.1% |
| 11 | `p1_g_rank` | 0.1% |
| 12 | `n_idf_shared` | 0.1% |
| 13 | `sup_addr` | 0.0% |
| 14 | `key_score` | 0.0% |
| 15 | `blk_g_gap` | 0.0% |

**Runtime profile of the full run:**

| Stage | Wall time | Peak RSS |
|---|---|---|
| load+normalise train | 14 s | 6376 MB |
| profile train | 32 s | 8391 MB |
| blocking train | 2729 s | 22154 MB |
| learned pruner | 201 s | 22154 MB |
| blocking diagnostics | 6 s | 22154 MB |
| training sample | 2 s | 22154 MB |
| features train | 259 s | 22154 MB |
| stage-1 model | 2550 s | 22154 MB |
| stage-2 model | 1604 s | 22154 MB |
| load+normalise test | 8 s | 22154 MB |
| blocking test | 2266 s | 22154 MB |
| features+stage-1 test | 759 s | 22154 MB |
| stage-2 test | 80 s | 22154 MB |
| decode + write | 28 s | 22154 MB |
| validate outputs | 59 s | 22154 MB |
