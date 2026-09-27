# Round 1: baseline cascade

**Setup:**
- multi-channel blocking with key-frequency cap 400, k_wide 40 and automatic K (19 per source);
- learned pruner;
- two-stage LightGBM with 127 leaves on 400K training entities.

**Results:**

| Metric | Value |
|---|---|
| Blocking recall before / after pruning | 87.85% / 87.50% |
| Oracle F0.5 (blocking ceiling) | 0.943 |
| Candidates per entity (train / test) | 4.76 / 5.73 |
| OOF macro F0.5, stage 1 / stage 2 | 0.8577 / 0.8594 |
| OOF precision / recall | 0.993 / 0.721 |
| Public leaderboard | **0.9141** |

**Finding:** precision was excellent, but recall was the bottleneck.
- 344K of 1.38M true validation pairs never reached the matcher.
- India was much weaker than the US (F0.5 0.80 vs 0.92).

**Engineering issue:** the decoder search took 57 minutes per stage, because each of about 550 policies re-matched against 8M ground-truth pairs.

**Next step:** build a diagnostic to learn *why* the missing pairs are lost, before changing any setting.
