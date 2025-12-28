# Handoff Document

## Project Status
*   **Date**: 2025-12-25
*   **Version**: 2.0.0
*   **State**: Modernized pipeline (TF2/Python3) is fully implemented. Submodules are updated.

## Recent Changes
*   Added `autochart.py` for streamlined inference.
*   Added `download_data.ps1` for Windows support.
*   Integrated `ffr-difficulty-model`.
*   Created `docs/DASHBOARD.md` and `VERSION.md`.

## Active Tasks
*   **Training**: The user needs to run the training pipeline (`scripts/train_all.py`) to generate the models required for `autochart.py`.
*   **Testing**: Verify the full pipeline end-to-end once models are trained.

## Submodules
See `docs/DASHBOARD.md` for details.
