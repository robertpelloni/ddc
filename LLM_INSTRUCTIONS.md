# Universal LLM Instructions

This file serves as a central repository for instructions tailored to Large Language Models interacting with this codebase.

## Core Directives

1. **Versioning**
   - The project version is stored in the `VERSION` and `VERSION.md` files.
   - Always increment the version for new builds or significant changes.
   - Update `CHANGELOG.md` with every version bump.
   - Reference the version in commit messages when bumping versions.

2. **Submodules**
   - `ddc_onset`: located at `ddc_onset/`
   - `ffr-difficulty-model`: located at `ffr-difficulty-model/`
   - Ensure submodules are synced, committed, and pushed when modified.
   - Keep `docs/DASHBOARD.md` aligned with submodule/project status changes.

3. **Code Style**
   - Target Python 3.8+ compatibility where practical.
   - Follow PEP 8 and prefer descriptive naming.
   - Be aware that the repository currently contains both legacy TensorFlow-oriented paths and newer PyTorch-oriented work for the active environment.

4. **Testing / Validation**
   - Validate command-line entry points when touched.
   - Prefer practical sanity checks for the active environment.
   - Keep documentation clear about what is validated versus what is only planned.

5. **Documentation**
   - Keep root/docs dashboards current.
   - Record substantial findings in `docs/`.
   - Update `HANDOFF.md` after meaningful progress.

## Project Structure

- `autochart.py`: main CLI entry point
- `infer/`: inference logic and library code
- `learn/`: training logic
- `scripts/`: orchestration/utilities
- `dataset/`: data processing utilities
- `docs/`: project analysis, audit, and planning documentation

## Model-Specific Notes

### Claude (`CLAUDE.md`)
- Focus on architectural integrity and detailed explanations.
- Explain rationale and impact for meaningful changes.

### Gemini (`GEMINI.md`)
- Use large-context analysis to understand project-wide implications.
- Prioritize efficient modernization and broad codebase synthesis.

### GPT (`GPT.md`)
- Focus on practical implementation, architecture cleanup, and debugging.
- Keep changes actionable and consistent with the active repo state.

### GitHub Copilot (`copilot-instructions.md`)
- Use version-aware commit messages when appropriate.
- Keep project structure/documentation in sync with implementation changes.

## Handoff Protocol

When handing off work:
1. Summarize current project state.
2. List active tasks and known issues.
3. Reference `docs/DASHBOARD.md` and major audit/planning docs as needed.
4. Ensure changes are committed and pushed when appropriate.
