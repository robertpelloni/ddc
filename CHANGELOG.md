# Changelog

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
