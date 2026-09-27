# Round 3: more data, higher capacity, tunable blocking width

**Changes:**
- 1.75× more training entities (700K);
- 255-leaf trees;
- wider blocking (k_wide 120, key-frequency cap 1500);
- blocking width and model capacity exposed as CLI flags.

**Results:**

| Metric | Round 2 | Round 3 |
|---|---|---|
| Blocking recall before / after pruning | 95.35% / 94.97% | **95.97% / 95.59%** |
| Oracle F0.5 | 0.9818 | **0.9842** |
| OOF macro F0.5, stage 1 / stage 2 | 0.9544 / 0.9553 | **0.9604 / 0.9617** |
| Candidates per entity (test) | 6.53 | 6.87 |

The run finished after the submission portal closed, so it has no leaderboard score. Validation has tracked the leaderboard conservatively: it came out 0.055 and 0.008 below it in rounds 1 and 2.

**Remaining error sources (next steps):**
- Records with no address *and* a placeholder or foreign-script name. These need cross-record cluster evidence, i.e. target-to-target expansion.
- Corrupted house numbers.
- Legal-suffix decoys that are identical in every field except the suffix.
