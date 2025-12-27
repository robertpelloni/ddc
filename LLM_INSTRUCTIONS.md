# LLM Instructions

This repository is maintained with the assistance of Large Language Models (LLMs). This file serves as a central point of truth for context, guidelines, and instructions for any AI agent working on this codebase.

## Core Directives

1.  **Versioning**:
    *   The project version is stored in the `VERSION` file.
    *   **Always** increment this version number for new builds or significant changes.
    *   Update `CHANGELOG.md` with every version bump.
    *   Reference the version in commit messages (e.g., "Release 0.2.1").

2.  **Submodules**:
    *   `ddc_onset`: Located at `ddc_onset/`. Use `git submodule update --remote` to fetch changes.
    *   `ffr-difficulty-model`: Located at `ffr-difficulty-model/`.
    *   Always ensure submodules are synced and committed.

3.  **Code Style**:
    *   Python 3.8+ compatible.
    *   Use `tensorflow` 2.x (Keras API).
    *   Follow PEP 8 where possible, but prioritize consistency with existing modernization patterns.

4.  **Testing**:
    *   Run `autochart.py` manually to verify CLI functionality.
    *   Ensure `import tensorflow` and `import librosa` succeed.

5.  **Documentation**:
    *   Keep `DASHBOARD.md` updated with project structure changes.
    *   Keep `ROADMAP.md` current.

## Project Structure

*   `autochart.py`: Main CLI entry point.
*   `infer/`: Inference logic and library code (`autochart_lib.py`).
*   `learn/`: Training logic (modernized).
*   `scripts/`: Utility scripts (e.g., `train_all.py`).
*   `dataset/`: Data processing utilities.
