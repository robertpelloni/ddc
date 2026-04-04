# DDC Training and Data Coverage Analysis

Date: 2026-04-04

## Executive Summary

This repository was prepared and trained against the official DDR pack download pipeline already present in the project. The resulting work produced:

- a refreshed end-to-end data preparation path for official DDR content
- PyTorch-based training orchestration for DDC onset + SymNet placement models
- 8 primary placement-model training buckets:
  - `dance-single_Easy`
  - `dance-single_Medium`
  - `dance-single_Hard`
  - `dance-single_Challenge`
  - `dance-double_Easy`
  - `dance-double_Medium`
  - `dance-double_Hard`
  - `dance-double_Challenge`
- an onset model trained from `dance-single_Hard`
- a difficulty evaluator retrained on the largest practical DDR dataset available from the downloaded official packs
- corrected difficulty-model training so both `dance-single` and `dance-double` are preserved during training instead of accidentally dropping one mode due to global NaN handling

This document records what data was used, what was not used, what was only partially used, the practical limits of the current pipeline, and the recommended next steps.

Update note: after this report was first drafted, the extractor was extended to support `.ssc` files as well as `.sm`, and the FFR difficulty-data loader was also updated to prefer `.ssc` over `.sm` where available. A refreshed corpus audit is captured in `docs/CORPUS_AUDIT_2026-04-04.md`, and the measured delta is summarized in `docs/SSC_EXPANSION_ANALYSIS_2026-04-04.md`. Full retraining against the refreshed `.ssc`-inclusive corpus remains the next recommended training step.

---

## Download Sources Used

The existing project downloader referenced the following official DDR pack sources on Zenius-i-vanisher:

- DDR A20 PLUS: `categoryid=1293`
- DDR A20: `categoryid=1292`
- DDR A: `categoryid=1148`
- DDR WORLD: `categoryid=1709`
- DDR X3 VS 2ndMIX: `categoryid=802`
- DDR X2: `categoryid=546`
- DDR X: `categoryid=295`
- DDR 2014: `categoryid=864`
- DDR 2013: `categoryid=845`
- DDR SuperNOVA2: `categoryid=77`
- DDR SuperNOVA: `categoryid=1`

The downloader reported completion markers for all configured packs, so no missing configured official-pack downloads were detected during this run.

---

## Data Preparation Summary

### Raw extraction

The extraction stage converted `.sm` files into the repository's JSON representation using `dataset/extract_json.py`.

Observed high-level results:

- roughly 1.2k songs extracted
- roughly 9.2k charts serialized for downstream filtering/training

### Filtered chart buckets

`prepare_data.py` created StepMania-mode/difficulty buckets and then split each bucket into train/valid/test lists.

Observed split counts:

- `dance-single_Beginner`: 964 train / 120 valid / 122 test
- `dance-single_Easy`: 964 train / 120 valid / 122 test
- `dance-single_Medium`: 964 train / 120 valid / 122 test
- `dance-single_Hard`: 964 train / 120 valid / 122 test
- `dance-single_Challenge`: 317 train / 39 valid / 41 test
- `dance-double_Beginner`: 1 train / 0 valid / 0 test
- `dance-double_Easy`: 964 train / 120 valid / 122 test
- `dance-double_Medium`: 964 train / 120 valid / 122 test
- `dance-double_Hard`: 964 train / 120 valid / 122 test
- `dance-double_Challenge`: 318 train / 39 valid / 41 test

---

## Why the final DDC placement plan used 8 runs

The user explicitly prioritized complete coverage across the practical mode+difficulty combinations, and noted that this likely meant 8 runs. The final implemented plan matched the 8-run interpretation:

- 4 difficulties for single
- 4 difficulties for double
- difficulties used: Easy / Medium / Hard / Challenge

### Rationale

- `dance-double_Beginner` had effectively no usable dataset size
- `dance-single_Beginner` existed in quantity, but the explicit 8-run framing strongly suggested the standard difficulty ladder excluding Beginner
- retaining the 8 major gameplay buckets aligns better with practical DDC usage inside ArrowVortex

---

## What was actually trained

## 1. Onset model

A PyTorch onset model was trained from:

- `dance-single_Hard`

This mirrors the repo's prior assumption that Hard/Expert-like content provides a strong onset-training signal.

## 2. Placement models

PyTorch SymNet training was wired through `scripts/train_pt.py` and orchestrated by `scripts/train_all.py`.

Final target buckets:

- `dance-single_Easy`
- `dance-single_Medium`
- `dance-single_Hard`
- `dance-single_Challenge`
- `dance-double_Easy`
- `dance-double_Medium`
- `dance-double_Hard`
- `dance-double_Challenge`

## 3. Difficulty evaluator

The difficulty evaluator pipeline was executed as:

1. `make_dataset_from_sm.py`
2. `build_features.py`
3. `train_model.py`

Observed processed-chart count for difficulty evaluation:

- 9,245 serialized charts/features

Observed trained mode models:

- `dance-double`
- `dance-single`

Observed reported scores:

- `dance-double`: R² ≈ 0.946
- `dance-single`: R² ≈ 0.967

The evaluator remains a floating-point regressor, which satisfies the requirement that its predictions be usable against arbitrary integer scales after mapping.

---

## Companion audit

A companion raw/extracted corpus audit is available at:

- `docs/CORPUS_AUDIT_2026-04-04.md`
- `docs/SSC_EXPANSION_ANALYSIS_2026-04-04.md`
- `docs/NOTE_OBJECT_SEMANTICS_2026-04-04.md`
- `docs/RETRAINING_REFRESH_PLAN_2026-04-04.md`
- `docs/BUCKET_SPLIT_DELTA_2026-04-04.md`

Those reports quantify:

- raw `.sm` vs `.ssc` inventory
- note-symbol frequency in extracted JSON
- per-bucket special-symbol coverage

## Critical data-usage findings

## A. Data that was used directly by DDC placement training

The DDC placement models used:

- mel-spectrogram-based audio features
- previous symbolic note-token context
- chart data from official DDR single/double buckets for Easy/Medium/Hard/Challenge

## B. Data that was not used in the final 8 placement runs

### 1. Beginner placement charts

Not included in the final 8 placement runs:

- `dance-single_Beginner`
- `dance-double_Beginner`

Notes:

- `dance-single_Beginner` had usable quantity but was excluded to preserve the requested 8-run setup
- `dance-double_Beginner` was effectively unusable due to only 1 training sample

### 2. Non-single/non-double game types

Only these chart types were used:

- `dance-single`
- `dance-double`

Not used:

- pump variants
- routine/couple/solo/etc.
- any other unsupported step types

### 3. `.ssc`-only content

The extraction path is based on `.sm` files. If a pack contains additional or richer information only in `.ssc`, that information is not captured by the current extraction pipeline.

---

## C. Shock arrows, mines, holds, rolls, lifts, fakes

This is the most important content-specific finding.

### DDC placement pipeline

The placement pipeline does **not** globally convert the note vocabulary down to taps only.

`dataset/filter_json.py` filters charts by chart type and coarse difficulty, but by default it does **not** remove nonzero symbols. It whitelists `'0'` automatically and only filters out other note symbols if an explicit `--arrow_types` restriction is supplied.

That means special note symbols can remain in the note strings used by the symbolic model if they are present in the `.sm` data.

Practical implication:

- shock-arrow-like/minelike symbols were **not proactively discarded** from DDC symbolic training data
- however, rare token frequency likely limits how well the model learns them

### Difficulty evaluator pipeline

The difficulty evaluator explicitly keeps only `NoteType.TAP` when preprocessing charts.

That means it does **not** use:

- shock arrows / mines
- hold semantics
- roll semantics
- lift semantics
- fake notes
- other non-tap note object types

Practical implication:

- the difficulty evaluator currently estimates tap-pattern difficulty well
- it does **not** fully model object-management difficulty or gimmick-specific burden

---

## D. Metadata/features available in code but not enabled during final DDC training

The chart classes can represent many additional conditioning features, including:

- coarse difficulty one-hot conditioning
- feet difficulty one-hot conditioning
- free-text/author conditioning
- beat phase
- beat phase sine/cosine
- measure phase
- beat/time deltas
- progress through chart
- quantized phase features
- wrap-count features

The final PyTorch training path did **not** explicitly turn those feature channels on.

So although the codebase can compute them, the trained models in this run relied primarily on:

- audio features
- autoregressive symbolic context

This is an important limitation and also the clearest next upgrade path.

---

## E. Onset-training limitations

The onset model was trained from `dance-single_Hard`, not from a mixed all-bucket onset corpus.

This is a practical compromise, not total-data coverage.

Potential future improvement:

- aggregate onset targets from all single+double buckets and train a broader onset detector

---

## Key code changes made during this work

## Main repository

### `scripts/train_all.py`

Updated to:

- use the PyTorch trainer (`scripts/train_pt.py`) for onset and SymNet runs
- train the 8 requested mode+difficulty placement buckets
- execute the difficulty-evaluator pipeline end-to-end:
  - StepMania chart serialization
  - feature building
  - model training

### `scripts/train_pt.py`

Fixed/trained around:

- model output handling for onset vs SymNet
- target shape compatibility for BCELoss
- checkpoint writing across epochs

### `infer/autochart_lib.py`

Adapted inference toward the PyTorch-trained model set:

- PyTorch imports and model loading
- `SymNet` checkpoint loading from `.pth`
- vocabulary loading from `vocab.json`
- autoregressive sequence generation via PyTorch

## Submodule: `ffr-difficulty-model`

### `scripts/train_model.py`

Corrected a major training-data retention issue:

- removed global `dropna()` behavior that dropped too much mixed-mode data
- now performs per-mode cleanup:
  - drop all-NaN columns per mode
  - then drop remaining invalid rows within that mode
- explicitly preserves floating-point target values via `astype(float)`

This fix was required to successfully train both:

- `dance-single`
- `dance-double`

Without that change, only one mode survived cleaning during observed runs.

---

## Trained artifact notes

Local trained artifacts were produced under local working directories during this effort.

Important operational note:

- intermediate working artifacts are very large
- the `output_v132/` tree is extremely large and should not be treated as a normal source-controlled artifact set
- the `models_v3/` model export is also large enough that push strategy should be handled carefully

Recommended repository strategy:

- commit code and documentation normally
- keep heavyweight local training outputs out of the normal repo history unless Git LFS or release artifacts are used

---

## Recommended next steps

## Priority 1: add explicit coverage auditing

Add a report script that counts, per bucket:

- chart count
- songs count
- unique note-symbol vocabulary
- counts of holds/mines/rolls/lifts/fakes if present

This will convert assumptions into measured coverage.

## Priority 2: retrain against the refreshed `.ssc`-inclusive extraction

The extractor now supports `.ssc` input where available. The next step is to rerun filtering/feature extraction/training so the newly recovered songs are incorporated into model training.

## Priority 3: add Beginner placement models where practical

Recommended:

- train `dance-single_Beginner`
- skip or special-case `dance-double_Beginner` until additional data is available

## Priority 4: improve difficulty evaluator object coverage

Extend the difficulty pipeline so it models more than taps:

- holds
- rolls
- mines/shock arrows
- lifts
- fakes

## Priority 5: enable explicit timing/conditioning features in DDC

Turn on and evaluate feature channels already supported in `learn/chart.py`, especially:

- beat-phase features
- timing-delta features
- progress features
- difficulty conditioning

## Priority 6: package ArrowVortex-ready deployment cleanly

Create a clean, reproducible export path for ArrowVortex integration that:

- selects final checkpoints
- writes a manifest
- documents model layout
- separates deployable artifacts from temporary training artifacts

---

## Bottom-line assessment

This run materially improved the repository and produced a much more complete training story than the prior state, especially for:

- practical single/double major-difficulty coverage
- difficulty evaluator retraining on official DDR data
- dual-mode difficulty-model retention
- PyTorch training/inference alignment

But it is not yet literally "all possible chart information" in the strongest sense.

The biggest remaining omissions are:

1. Beginner placement coverage
2. `.ssc`-only information
3. non-tap object support in the difficulty evaluator
4. explicit symbolic timing/metadata conditioning in DDC

Those are the next recommended areas of work.