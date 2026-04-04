# Project Dashboard

## Submodules Status

| Submodule | Path | Version (Commit) | URL |
|-----------|------|------------------|-----|
| **ddc_onset** | `ddc_onset/` | tracked via submodule | [https://github.com/robertpelloni/ddc_onset](https://github.com/robertpelloni/ddc_onset) |
| **ffr-difficulty-model** | `ffr-difficulty-model/` | local submodule changes prepared to preserve both single and double mode training | [https://github.com/robertpelloni/ffr-difficulty-model](https://github.com/robertpelloni/ffr-difficulty-model) |

## Project Structure

The project is organized as follows:

- **`autochart.py`**: Main entry point for generating charts from audio files.
- **`dataset/`**: Scripts for processing raw StepMania `.sm` files and JSON datasets.
- **`learn/`**: Core machine learning/training code.
- **`scripts/`**: Training orchestration utilities.
- **`infer/`**: Inference/runtime integration code.
- **`ffr-difficulty-model/`**: Submodule for difficulty rating.
- **`ddc_onset/`**: Submodule for onset detection.
- **`docs/TRAINING_ANALYSIS_2026-04-04.md`**: Detailed report of training coverage, omissions, findings, and recommendations.
- **`docs/CORPUS_AUDIT_2026-04-04.md`**: Detailed report of `.sm`/`.ssc` inventory and note-symbol coverage in the extracted corpus.
- **`docs/SSC_EXPANSION_ANALYSIS_2026-04-04.md`**: Delta report quantifying the exact increase in songs/charts/bucket coverage from `.ssc` support.
- **`docs/NOTE_OBJECT_SEMANTICS_2026-04-04.md`**: Semantic audit documenting which note-object symbols actually appear in the refreshed corpus.
- **`docs/RETRAINING_REFRESH_PLAN_2026-04-04.md`**: Step-by-step refresh plan for `.ssc`-inclusive retraining.
- **`docs/BUCKET_SPLIT_DELTA_2026-04-04.md`**: Exact train/valid/test split deltas for every bucket after refreshed `.ssc`-inclusive preparation.
- **`docs/REPO_HEALTH_AUDIT_2026-04-04.md`**: Remaining repository blocker/hotspot audit for merge conflicts and legacy TensorFlow references.
- **`docs/LEGACY_SUBTREE_QUARANTINE_2026-04-04.md`**: Quarantine decision note for the final legacy-subtree conflict-marker files.

## Build Information

- **Version**: 0.2.13
- **Build Date**: 2026-04-04
- **Environment Notes**: Local training work was adapted to PyTorch for the active environment; legacy TensorFlow-oriented paths remain in the repository.

## Current Training Findings

### DDC placement buckets trained

- `dance-single_Easy`
- `dance-single_Medium`
- `dance-single_Hard`
- `dance-single_Challenge`
- `dance-double_Easy`
- `dance-double_Medium`
- `dance-double_Hard`
- `dance-double_Challenge`

### Extraction status

- `dataset/extract_json.py` now supports `.ssc` as well as `.sm`.
- The FFR difficulty-data loader now prefers `.ssc` over `.sm` within each song directory.
- A refreshed local extraction confirmed corpus growth from 1234 to 1254 songs and from 9241 to 9403 charts.
- A probe run of the refreshed FFR preprocessing path confirmed approximately 1255 simfiles and 9407 serialized charts are reachable.
- Repository-health cleanup reduced unresolved merge-conflict-marker files from 15 to 2 through successive cleanup passes.
- Full retraining against that refreshed corpus is still pending/recommended.

### Additional models

- onset model trained from `dance-single_Hard`
- floating-point difficulty regressors trained for:
  - `dance-single`
  - `dance-double`

## High-Level Limitations

The current practical training pass still leaves room for future expansion:

- Beginner placement buckets are not part of the final 8-bucket export plan
- `.ssc`-only data is not yet included by the extraction path
- the difficulty evaluator currently uses tap-type notes and does not fully model shock arrows/mines/holds/rolls/lifts/fakes
- additional symbolic timing/conditioning features exist in code but were not enabled in the final training run
