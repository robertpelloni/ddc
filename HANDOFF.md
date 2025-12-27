# Handoff Documentation

**Date:** 2023-10-27
**Agent:** Jules
**Version:** 0.2.1

## Session Summary
This session focused on modernizing the Dance Dance Convolution (DDC) codebase from a legacy Python 2/TensorFlow 0.12 state to a modern Python 3/TensorFlow 2.x architecture. Key external components (`ddc_onset` and `ffr-difficulty-model`) were integrated as git submodules to provide beat detection and difficulty rating capabilities.

## Key Accomplishments
1.  **Modernization**: Codebase is now compatible with Python 3.8+ and TF 2.x. `librosa` replaces `essentia`.
2.  **Architecture**:
    *   **Inference**: `autochart.py` and `infer/autochart_lib.py` drive the end-to-end generation process.
    *   **Training**: `scripts/train_all.py` allows retraining from scratch.
3.  **Integration**:
    *   **DDC Onset**: Used for precise timing/beat detection.
    *   **FFR Difficulty**: Used to rate generated charts on a 0-20 scale.
4.  **Packaging**: `setup.py` added for pip installation.

## Repository State
*   **Branch**: `master` (features merged).
*   **Submodules**:
    *   `ddc_onset`: Checked out and updated.
    *   `ffr-difficulty-model`: Checked out and updated.
*   **Configuration**:
    *   `VERSION` file controls the package version.
    *   `LLM_INSTRUCTIONS.md` guides future agents.

## Known Issues / Future Work
*   **Double Charts**: The current `SymNet` model is trained primarily for Single (4-panel) play. Generating Double (8-panel) charts requires training a new model on appropriate data. The codebase supports the logic, but the model weights are needed.
*   **Integration Tests**: Sandbox environment limitations prevented full integration testing with audio hardware/drivers, so reliance is on unit logic and manual verification.

## Instructions for Next Agent
1.  **Check Submodules**: Always run `git submodule update --init --recursive` upon starting.
2.  **Version Bump**: If you make changes, increment `VERSION` and update `CHANGELOG.md`.
3.  **Deployment**: Use `pip install .` to test installation.
