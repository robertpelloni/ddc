# Project Dashboard

This dashboard provides an overview of the integrated components, submodules, and current versioning of the Dance Dance Convolution project.

## Project Status
**Version:** 0.2.14  
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

#### 6. SSC Expansion Analysis
**Location:** `docs/SSC_EXPANSION_ANALYSIS_2026-04-04.md`  
**Description:** Delta-focused report quantifying the exact corpus growth unlocked by `.ssc` extraction support.

#### 7. Note Object Semantics Audit
**Location:** `docs/NOTE_OBJECT_SEMANTICS_2026-04-04.md`  
**Description:** Detailed semantic audit of observed chart symbols, including taps, hold-heads, tails, and mines.

#### 8. Retraining Refresh Plan
**Location:** `docs/RETRAINING_REFRESH_PLAN_2026-04-04.md`  
**Description:** Step-by-step plan for the next `.ssc`-inclusive full model refresh.

#### 9. Bucket Split Delta Audit
**Location:** `docs/BUCKET_SPLIT_DELTA_2026-04-04.md`  
**Description:** Exact train/valid/test split delta per bucket after `.ssc`-inclusive refreshed data preparation.

#### 10. Repository Health Audit
**Location:** `docs/REPO_HEALTH_AUDIT_2026-04-04.md`  
**Description:** Audit of remaining unresolved merge conflicts and legacy TensorFlow / `.h5` reference hotspots.

#### 11. Legacy Subtree Quarantine Note
**Location:** `docs/LEGACY_SUBTREE_QUARANTINE_2026-04-04.md`  
**Description:** Rationale for treating the remaining `ddc_stepmania/` conflict-marker files as quarantined legacy-subtree content.

#### 12. SSC Refresh Readiness Audit
**Location:** `docs/SSC_REFRESH_READINESS_2026-04-04.md`  
**Description:** Exact state of the prepared `.ssc`-inclusive refresh work directory and the recommended resume-friendly training command.

## Extraction Status

- `dataset/extract_json.py` now supports both `.sm` and `.ssc` inputs.
- The FFR difficulty-data loader now also prefers `.ssc` over `.sm` where available.
- A refreshed local extraction confirmed growth from **1234** to **1254** extracted songs and from **9241** to **9403** charts when `.ssc`-only songs are included.
- A probe run of the refreshed FFR preprocessing path confirmed approximately **1255 simfiles** and **9407 serialized charts** are now reachable.
- Repository-health cleanup reduced unresolved merge-conflict-marker files from **15** to **2** through successive cleanup passes.
- Full retraining against the refreshed `.ssc`-inclusive extraction is still recommended as the next model-refresh step.

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
