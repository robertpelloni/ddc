# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.10] - 2026-04-04

### Added
- Added `scripts/audit_repo_health.py` for repeatable repository-health auditing.
- Added `docs/REPO_HEALTH_AUDIT_2026-04-04.md` documenting unresolved merge conflicts, TensorFlow hotspots, and legacy `.h5` / `train_v2.py` / `models_v2` references.

### Changed
- Documented the remaining blocker surface that should be cleaned up before considering the repo fully normalized after the PyTorch-oriented training refresh work.

## [0.2.9] - 2026-04-04

### Added
- Added `scripts/compare_bucket_counts.py` for reproducible before/after split-count comparison between prepared work directories.
- Added `docs/BUCKET_SPLIT_DELTA_2026-04-04.md` documenting the exact train/valid/test deltas in every bucket after the `.ssc`-inclusive refresh.

### Changed
- Documented the exact downstream DDC training-input expansion for the practical 8-bucket plan.

## [0.2.8] - 2026-04-04

### Added
- Added `docs/RETRAINING_REFRESH_PLAN_2026-04-04.md` capturing the exact next-phase `.ssc`-inclusive retraining workflow.

### Changed
- Cleaned and replaced the conflicted root `README.md` with a concise current-state project overview and documentation index.

## [0.2.7] - 2026-04-04

### Added
- Added `scripts/audit_note_objects.py` for repeatable note-object semantic audits against extracted JSON charts.
- Added `docs/NOTE_OBJECT_SEMANTICS_2026-04-04.md` documenting observed note-object semantics in the refreshed official DDR corpus.

### Changed
- Documented that the refreshed corpus contains taps, hold-heads, tails, and mines, while no roll-head, attack, fake, keysound, or lift symbols were observed in the audited official-pack extraction.
- Documented per-bucket semantic coverage so future difficulty-model work can target hold/mine-aware features more precisely.

## [0.2.6] - 2026-04-04

### Added
- Documented FFR `.ssc`-inclusive loader expansion and validation counts in `docs/SSC_EXPANSION_ANALYSIS_2026-04-04.md`.

### Changed
- Updated the FFR difficulty-data loader to prefer `.ssc` over `.sm` within each song directory.
- Cleaned `learn/extract_feats_v2.py` so the refreshed feature-extraction path is syntactically valid again after prior merge-conflict residue.
- Validated the refreshed FFR preprocessing path at approximately **1255 simfiles** and **9407 serialized charts**.

## [0.2.5] - 2026-04-04

### Added
- Added `docs/SSC_EXPANSION_ANALYSIS_2026-04-04.md` documenting the measured corpus delta unlocked by `.ssc` extraction support.

### Changed
- Fixed `scripts/prepare_data.py` so the downstream data-prep path is syntactically valid again after prior merge-conflict residue.
- Documented the precise `.ssc`-driven corpus expansion:
  - songs: 1234 -> 1254
  - charts: 9241 -> 9403
  - major practical DDC buckets all increased

## [0.2.4] - 2026-04-04

### Added
- Added `.ssc` extraction support to `dataset/extract_json.py` using `simfile` for `.ssc`-only songs.
- Added validation/audit updates confirming the refreshed extracted corpus grows from 1234 to 1254 songs and from 9241 to 9403 charts after including `.ssc` content.

### Changed
- Updated corpus-audit reporting to reflect the new `.ssc`-inclusive extraction state.
- Updated training analysis to distinguish between extractor support being present and full retraining against the refreshed corpus still being pending.

## [0.2.3] - 2026-04-04

### Added
- Added a reproducible corpus-audit script at `scripts/audit_corpus.py`.
- Added `docs/CORPUS_AUDIT_2026-04-04.md` documenting raw `.sm`/`.ssc` inventory, note-symbol distribution, and per-bucket special-symbol coverage.

### Changed
- Documented that 20 `.ssc` files exist in the official DDR corpus but are not yet ingested by the current extractor.
- Documented that the extracted DDC symbolic corpus contains substantial non-binary note symbols (`2`, `3`, `M`) and is therefore not tap-only.

## [0.2.2] - 2026-04-04

### Added
- Added a comprehensive training/data-coverage report in `docs/TRAINING_ANALYSIS_2026-04-04.md`.
- Added local export/documentation flow for the newly trained DDC v2 model set.

### Changed
- Switched the DDC training orchestrator to use the PyTorch trainer for onset and SymNet training.
- Trained the practical 8-bucket placement configuration for ArrowVortex-oriented DDR use:
  - single Easy / Medium / Hard / Challenge
  - double Easy / Medium / Hard / Challenge
- Updated inference scaffolding in `infer/autochart_lib.py` toward PyTorch checkpoint loading.
- Corrected the FFR difficulty-model training path so `dance-single` and `dance-double` both survive data cleaning and train successfully.
- Preserved floating-point difficulty regression behavior for the difficulty evaluator.
- Added ignore rules for large local training artifacts and generated exports.

### Fixed
- Fixed target-shape/model-output issues in the PyTorch DDC training path.
- Fixed mode-specific NaN handling in `ffr-difficulty-model/scripts/train_model.py`.
- Resolved repository documentation inconsistency around current training status and artifact handling.

## [0.2.1] - 2023-10-27

### Added
- **Versioning**: Centralized version number in `VERSION` file.
- **Documentation**: Added `DASHBOARD.md`, `HANDOFF.md`, and LLM instruction files.
- **Submodule**: Ensured `ffr-difficulty-model` is fully integrated.

## [0.2.0] - 2023-10-27

### Added
- **Modernization**: Ported entire codebase from Python 2.7 / TensorFlow 0.12 to Python 3.8+ / TensorFlow 2.x.
- **Git Submodules**: Added `ddc_onset` for beat detection and `ffr-difficulty-model` for difficulty rating.
- **AutoChart CLI**: New `autochart.py` tool for end-to-end chart generation, difficulty rating, and metadata handling.
- **Training Pipeline**: `scripts/train_all.py` allows full retraining of DDC models from raw audio and `.sm` files.
- **Feature Extraction**: Replaced `essentia` dependency with `librosa`.
- **Packaging**: Added `setup.py` for standard pip installation.

### Changed
- **Server**: `infer/ddc_server.py` updated to use the new `AutoChart` library logic.
- **Dependencies**: Updated `requirements.txt` to reflect modern ecosystem (TF 2.x, librosa, simfile).
- **Structure**: Reorganized legacy scripts; removed obsolete bash scripts.

### Removed
- **Legacy Code**: Removed `infer/onset_net.py` and `infer/sym_net.py` (logic moved to `ddc_onset` submodule and `learn/models_v2.py`).
