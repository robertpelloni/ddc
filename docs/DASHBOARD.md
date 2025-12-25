# Project Dashboard

## Submodules Status

| Submodule | Path | Version (Commit) | URL |
|-----------|------|------------------|-----|
| **ddc_onset** | `ddc_onset/` | `d02aad3` | [https://github.com/robertpelloni/ddc_onset](https://github.com/robertpelloni/ddc_onset) |
| **ffr-difficulty-model** | `ffr-difficulty-model/` | `54d9247` | [https://github.com/robertpelloni/ffr-difficulty-model](https://github.com/robertpelloni/ffr-difficulty-model) |

## Project Structure

The project is organized as follows:

*   **`autochart.py`**: The main entry point for generating charts from audio files.
*   **`dataset/`**: Scripts for processing raw StepMania (`.sm`) files and JSON datasets.
    *   `extract_json.py`: Extracts data from `.sm` files.
    *   `filter_json.py`: Filters charts based on criteria.
*   **`learn/`**: Core machine learning code.
    *   `models_v2.py`: TensorFlow 2 / Keras model definitions (Onset and SymNet).
    *   `extract_feats_v2.py`: Audio feature extraction using Librosa.
    *   `data_gen.py`: Data generators for training.
*   **`scripts/`**: Shell and Python scripts for orchestration.
    *   `train_v2.py`: Main training script.
    *   `prepare_data.py`: Data preparation pipeline.
    *   `download_data.sh` / `download_data.ps1`: Scripts to download official DDR packs.
*   **`infer/`**: Legacy inference code (mostly superseded by `autochart.py`).
*   **`ffr-difficulty-model/`**: Submodule for calculating difficulty ratings.
*   **`ddc_onset/`**: Submodule for onset detection data/models.

## Build Information

*   **Version**: 2.0.0
*   **Build Date**: 2025-12-25
*   **Environment**: Python 3.x, TensorFlow 2.x, Librosa
