# Changelog

## v3.0.0: Round 3
- 1.75× training sample and 255-leaf trees.
- Blocking width, batch size and tree capacity are now CLI flags.
- Cleaner error-analysis tables in the generated methodology.
- OOF macro F0.5 **0.9617**.

## v2.0.0: Round 2
- Blocking diagnostic tool.
- Address-evidence rerank path, house-number × street keys and vowel-free skeleton keys.
- Region codes (including Indic scripts), ordinals and marker-number normalisation.
- Decoy-aware raw name-variation features.
- `FastScorer`: decoder search from 57 min to seconds.
- Leaderboard **0.9628**, up from 0.9141.

## v1.0.0: Round 1
- Baseline cascade: normalisation → multi-channel blocking → learned pruner → two-stage LightGBM → expected-F0.5 decoder.
- Leaderboard **0.9141**.
