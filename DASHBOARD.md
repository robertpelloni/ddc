# Project Dashboard

This dashboard provides an overview of the integrated components, submodules, and current versioning of the Dance Dance Convolution project.

## Project Status
**Version:** 0.2.28  
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

#### 13. Training Refresh Launch Record
**Location:** `docs/TRAINING_REFRESH_LAUNCH_2026-04-04.md`  
**Description:** Records the actual launch of the `.ssc`-inclusive refresh training run and where to monitor progress.

#### 14. Training Refresh Progress Snapshot
**Location:** `docs/TRAINING_REFRESH_PROGRESS_2026-04-04.md`  
**Description:** In-flight runtime snapshot showing the active refresh progressing and producing initial checkpoints.

#### 15. Training Refresh Progress Snapshot #2
**Location:** `docs/TRAINING_REFRESH_PROGRESS_2_2026-04-04.md`  
**Description:** Later runtime snapshot showing onset checkpoint completion and transition into practical bucketed SymNet training.

#### 16. Training Refresh Progress Snapshot #3
**Location:** `docs/TRAINING_REFRESH_PROGRESS_3_2026-04-04.md`  
**Description:** Later runtime snapshot showing the first practical SymNet bucket checkpoint being produced.

#### 17. Training Refresh Progress Snapshot #4
**Location:** `docs/TRAINING_REFRESH_PROGRESS_4_2026-04-04.md`  
**Description:** Later runtime snapshot showing continued checkpoint accumulation within the first practical bucket.

#### 18. Training Refresh Progress Snapshot #5
**Location:** `docs/TRAINING_REFRESH_PROGRESS_5_2026-04-04.md`  
**Description:** Later runtime snapshot showing the first practical bucket progressing to at least its third checkpoint and later-epoch training output.

#### 19. Training Refresh Progress Snapshot #6
**Location:** `docs/TRAINING_REFRESH_PROGRESS_6_2026-04-04.md`  
**Description:** Later runtime snapshot showing the first practical bucket progressing to at least its fourth checkpoint and later-epoch training output.

#### 20. Training Refresh Progress Snapshot #7
**Location:** `docs/TRAINING_REFRESH_PROGRESS_7_2026-04-04.md`  
**Description:** Later runtime snapshot showing the first practical bucket progressing to at least its fifth checkpoint and later-epoch training output.

#### 21. Training Refresh Progress Snapshot #8
**Location:** `docs/TRAINING_REFRESH_PROGRESS_8_2026-04-04.md`  
**Description:** Major runtime snapshot showing the first practical bucket appearing complete and the second practical bucket already in progress.

#### 22. Training Refresh Progress Snapshot #9
**Location:** `docs/TRAINING_REFRESH_PROGRESS_9_2026-04-04.md`  
**Description:** Follow-up runtime snapshot showing continued in-flight progress inside the second practical bucket after that transition.

#### 23. Training Refresh Progress Snapshot #10
**Location:** `docs/TRAINING_REFRESH_PROGRESS_10_2026-04-04.md`  
**Description:** Follow-up runtime snapshot showing the second practical bucket advancing to at least its seventh checkpoint and entering `Epoch 8/10`.

#### 24. Training Refresh Progress Snapshot #11
**Location:** `docs/TRAINING_REFRESH_PROGRESS_11_2026-04-04.md`  
**Description:** Follow-up runtime snapshot showing the second practical bucket progressing substantially deeper into the late portion of `Epoch 8/10`.

#### 25. Training Refresh Progress Snapshot #12
**Location:** `docs/TRAINING_REFRESH_PROGRESS_12_2026-04-04.md`  
**Description:** Follow-up runtime snapshot showing the second practical bucket advancing to at least its eighth checkpoint and entering `Epoch 9/10`.

#### 26. Training Refresh Progress Snapshot #13
**Location:** `docs/TRAINING_REFRESH_PROGRESS_13_2026-04-04.md`  
**Description:** Follow-up runtime snapshot showing the second practical bucket advancing to at least its ninth checkpoint and entering `Epoch 10/10`.

## Extraction Status

- `dataset/extract_json.py` now supports both `.sm` and `.ssc` inputs.
- The FFR difficulty-data loader now also prefers `.ssc` over `.sm` where available.
- A refreshed local extraction confirmed growth from **1234** to **1254** extracted songs and from **9241** to **9403** charts when `.ssc`-only songs are included.
- A probe run of the refreshed FFR preprocessing path confirmed approximately **1255 simfiles** and **9407 serialized charts** are now reachable.
- Repository-health cleanup reduced unresolved merge-conflict-marker files from **15** to **2** through successive cleanup passes.
- The resume-friendly `.ssc`-inclusive refresh has now been launched against `data/ssc_refresh_work`.
- Active runtime log: `data/ssc_refresh_training.log`
- Observed in-flight milestone: onset training completed its visible 5-checkpoint set under `data/ssc_refresh_work/models/onset/`.
- Observed later in-flight milestone: the refresh transitioned into practical SymNet bucket training and created `data/ssc_refresh_work/models/dance-single_Easy/`.
- Observed next in-flight milestone: `data/ssc_refresh_work/models/dance-single_Easy/model_01.pth` appeared, confirming practical bucket checkpoint production.
- Observed continued in-flight progress: `dance-single_Easy` advanced to at least `model_02.pth`, confirming ongoing checkpoint accumulation within the first practical bucket.
- Observed further in-flight progress: `dance-single_Easy` advanced to at least `model_03.pth`, with later-epoch (`4/10`) log output observed.
- Observed next in-flight progress: `dance-single_Easy` advanced to at least `model_04.pth`, with later-epoch (`5/10`) log output observed.
- Observed further in-flight progress: `dance-single_Easy` advanced to at least `model_05.pth`, with later-epoch (`6/10`) log output observed.
- Observed major in-flight milestone: `dance-single_Easy` reached a full observed 10-checkpoint set and `dance-single_Medium` appeared, advancing to at least `model_06.pth` while later-epoch (`7/10`) log output was observed.
- Observed continued second-bucket progress: the refresh remained alive, log output advanced substantially deeper into `Epoch 7/10`, and artifact recency still pointed to `dance-single_Medium` as the active practical bucket frontier.
- Observed further second-bucket progress: `dance-single_Medium` advanced to at least `model_07.pth`, and the active log progressed into `Epoch 8/10`.
- Observed continued second-bucket execution: `dance-single_Medium` remained the active artifact frontier while the monitored log progressed substantially deeper into the late portion of `Epoch 8/10`.
- Observed next second-bucket milestone: `dance-single_Medium` advanced to at least `model_08.pth`, `Epoch 8/10` completed with validation output, and the active log progressed into `Epoch 9/10`.
- Observed further second-bucket milestone: `dance-single_Medium` advanced to at least `model_09.pth`, the active log progressed into `Epoch 10/10`, and two active Python processes were visible while the run continued without interruption.

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
