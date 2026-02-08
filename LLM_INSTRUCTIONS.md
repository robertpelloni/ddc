<<<<<<< HEAD
# Universal LLM Instructions

This file serves as a central repository for instructions tailored to various Large Language Models (LLMs) interacting with this codebase.

## General Instructions

*   **Project Context**: This is the `ddc-stepmania` project, a modernized version of Dance Dance Convolution for generating DDR charts.
*   **Versioning**: Always check `VERSION.md` for the current project version. When making significant changes, update `CHANGELOG.md` and increment the version in `VERSION.md`.
*   **Submodules**: The project uses submodules (`ddc_onset`, `ffr-difficulty-model`). Ensure they are initialized and updated.
*   **Code Style**: Follow PEP 8 for Python code. Use descriptive variable names and type hints where possible.

## Model-Specific Instructions

### Claude (CLAUDE.md)
*   Focus on architectural integrity and detailed explanations.
*   When proposing changes, provide a clear rationale and potential impact.

### Gemini (GEMINI.md)
*   Leverage your large context window to analyze the entire project structure.
*   Prioritize efficient and modern solutions (e.g., TensorFlow 2.x, Librosa).

### GPT (GPT.md)
*   Provide concise and actionable code snippets.
*   Focus on practical implementation and debugging.

### GitHub Copilot (copilot-instructions.md)
*   **Role**: You are an expert AI programming assistant.
*   **Versioning**: When asked to update the version, increment the number in `VERSION.md` and add an entry to `CHANGELOG.md`.
*   **Commit Messages**: Include the version number in commit messages when bumping versions (e.g., "Bump version to 2.0.1").
*   **Submodules**: Be aware of the `docs/DASHBOARD.md` file which tracks submodule status.

## Handoff Protocol

When handing off the project to another model or session:
1.  Summarize the current state of the project.
2.  List any active tasks or known issues.
3.  Reference the `docs/DASHBOARD.md` for submodule information.
4.  Ensure all changes are committed and pushed.
=======
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
>>>>>>> origin/ddc-modernization-and-integration-14116118131799338522
