# Round 2: diagnosis-driven blocking and decoy-aware matching

**What the diagnostic showed** (20K sampled entities, 73K true pairs):

| Outcome | Share |
|---|---|
| kept | 88.0% |
| lost after rerank (rank > K) | 4.4% |
| lost in the key vote | 5.3% |
| shared only very common keys | 2.3% |
| no shared key at all | 0.02% |

The lost pairs almost always had an **identical address** but an unusable name: another script (Bengali, Tamil, Devanagari), a placeholder ("Halojax") or a social handle. The name-weighted rerank buried them.

**Error examples showed a second problem: decoys.** "Program" vs "Program **Co**" at the same address is a *different* entity, while "& Co", "[Inc]" and "(Services)" variants are the *same* entity. The normalisation had erased exactly this difference.

**Changes:**
- **Blocking:**
  - an address-evidence rerank path (address + numeric anchors, stronger path wins);
  - house-number × street keys;
  - a vowel-free phonetic skeleton ("iunaited phuds" ≈ "united foods");
  - wider vote (k_wide 80, key-frequency cap 1000).
- **Normalisation:** region/state codes, including state names in Indic scripts, ordinals ("Fifth" → 5), "NULL" handling, splitting "No.504" into a marker and a number, and v/b folding.
- **Features:** 11 raw name-variation features and skeleton similarity.
- **Decoder:** wider search grid; `FastScorer` computes pair correctness once, cutting the search from 57 min to seconds.

**Results:**

| Metric | Round 1 | Round 2 |
|---|---|---|
| Blocking recall (final) | 87.50% | **94.97%** |
| Oracle F0.5 | 0.943 | **0.982** |
| OOF macro F0.5 | 0.8594 | **0.9553** |
| OOF recall | 0.721 | **0.890** |
| India / US F0.5 | 0.80 / 0.92 | **0.94 / 0.97** |
| Public leaderboard | 0.9141 | **0.9628** |
