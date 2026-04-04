# Project Dashboard

This dashboard provides an overview of the integrated components, submodules, and current versioning of the Dance Dance Convolution project.

## Project Status
**Version:** 0.2.3  
**Build Status:** Passing (manual training/integration validation)  
**Python Runtime Reality:** Current repository training work was adapted to PyTorch for the local environment, while legacy TensorFlow-oriented code paths still exist in the codebase.  

## Current Training State

### DDC placement coverage
The current recommended practical placement-model configuration is:

- `dance-single_Easy`
- `dance-single_Medium`
- `dance-single_Hard`
- `dance-single_Challenge`
- `dance-double_Easy`
- `dance-double_Medium`
- `dance-double_Hard`
- `dance-double_Challenge`

An onset model was also trained from `dance-single_Hard`.

### Difficulty evaluator coverage
The difficulty evaluator was retrained from official DDR chart data and now successfully trains both:

- `dance-single`
- `dance-double`

with floating-point regression outputs suitable for remapping onto arbitrary integer scales.

## Integrated Submodules

| Submodule | Path | Description | Version/Commit |
| :--- | :--- | :--- | :--- |
| **DDC Onset** | `ddc_onset/` | Provides deep learning models for precise onset (beat) detection. Used for aligning steps to audio. | tracked via submodule |
| **FFR Difficulty Model** | `ffr-difficulty-model/` | Difficulty estimator for stepcharts, retrained in this work for both single and double mode preservation. | tracked via submodule |

## Project Structure

### Root Directory
- `autochart.py`: Main CLI tool
- `setup.py`: Packaging script
- `VERSION`: Single source of truth for project version
- `requirements.txt`: Python dependencies
- `LLM_INSTRUCTIONS.md`: Guidelines for AI contributors
- `HANDOFF.md`: Context for handovers

### Components

#### 1. Training Pipeline
**Location:** `scripts/train_all.py`  
**Description:** Orchestrates data prep, feature extraction, DDC training, and difficulty-model retraining.

#### 2. Inference Engine
**Location:** `infer/autochart_lib.py`  
**Description:** Core library for chart generation from arbitrary audio.

#### 3. Server
**Location:** `infer/ddc_server.py`  
**Description:** Flask API layer for external integration such as ArrowVortex-oriented workflows.

#### 4. Data/Training Analysis
**Location:** `docs/TRAINING_ANALYSIS_2026-04-04.md`  
**Description:** Comprehensive audit of what data was used, what was omitted, and recommended next steps.

#### 5. Corpus Audit
**Location:** `docs/CORPUS_AUDIT_2026-04-04.md`  
**Description:** Raw corpus inventory including `.sm` vs `.ssc` file counts, note-symbol distribution, and per-bucket special-symbol coverage.

## Important Operational Notes

- Large local training artifacts should not be committed casually.
- `output_v132/` and local model exports are considered heavyweight generated artifacts.
- If deployable model publication is desired, use a deliberate artifact strategy (Git LFS, release assets, or a dedicated model distribution channel).

## Usage Quickstart

**Retrain from raw packs:**
```bash
python scripts/train_all.py <packs_dir> <work_dir>
```

**Run the server:**
```bash
python infer/ddc_server.py --models_dir <models_dir> --ffr_dir <ffr_model_dir>
```
