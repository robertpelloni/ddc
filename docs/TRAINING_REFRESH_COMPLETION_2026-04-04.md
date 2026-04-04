# SSC-Inclusive Training Refresh Completion Report

Date: 2026-04-04

## Purpose

This document records the completion-state milestone for the active `.ssc`-inclusive refresh run launched against `data/ssc_refresh_work`.

This is the first monitoring snapshot that shows:

- completion across the full practical 8-bucket placement-model plan
- completed onset checkpoints
- refreshed floating-point difficulty-model artifacts for both `dance-single` and `dance-double`

It therefore serves as the primary completion/inventory record for the refresh run.

---

## Completion Summary

The active `.ssc`-inclusive refresh has now produced all major expected artifact categories:

### Placement-model practical buckets

Observed complete checkpoint sets for:

- `dance-single_Easy` → 10 checkpoints
- `dance-single_Medium` → 10 checkpoints
- `dance-single_Hard` → 10 checkpoints
- `dance-single_Challenge` → 10 checkpoints
- `dance-double_Easy` → 10 checkpoints
- `dance-double_Medium` → 10 checkpoints
- `dance-double_Hard` → 10 checkpoints
- `dance-double_Challenge` → 10 checkpoints

### Onset model

Observed complete onset checkpoint set:

- `onset` → 5 checkpoints

### Difficulty evaluator

Observed refreshed floating-point FFR model artifacts:

- `data/ssc_refresh_work/ffr_models/dance-double.p`
- `data/ssc_refresh_work/ffr_models/dance-single.p`

This is the strongest evidence yet that the refresh completed the user-requested broader retraining objective.

---

## Final Observed Artifact Inventory

### Onset

Directory:

- `data/ssc_refresh_work/models/onset/`

Observed:

- `model_01.pth`
- `model_02.pth`
- `model_03.pth`
- `model_04.pth`
- `model_05.pth`

Latest observed mtime:

- `model_05.pth` → `2026-04-04 02:50:21`

### Single-mode placement buckets

#### `dance-single_Easy`
- latest observed checkpoint: `model_10.pth` → `2026-04-04 03:28:13`

#### `dance-single_Medium`
- latest observed checkpoint: `model_10.pth` → `2026-04-04 03:58:00`

#### `dance-single_Hard`
- latest observed checkpoint: `model_10.pth` → `2026-04-04 04:29:43`

#### `dance-single_Challenge`
- latest observed checkpoint: `model_10.pth` → `2026-04-04 04:59:49`

### Double-mode placement buckets

#### `dance-double_Easy`
- latest observed checkpoint: `model_10.pth` → `2026-04-04 05:28:27`

#### `dance-double_Medium`
- latest observed checkpoint: `model_10.pth` → `2026-04-04 05:57:49`

#### `dance-double_Hard`
- latest observed checkpoint: `model_10.pth` → `2026-04-04 06:28:48`

#### `dance-double_Challenge`
- latest observed checkpoint: `model_10.pth` → `2026-04-04 06:57:30`

### Difficulty evaluator artifacts

Directory:

- `data/ssc_refresh_work/ffr_models/`

Observed:

- `dance-double.p` → `2026-04-04 07:01:41`
- `dance-single.p` → `2026-04-04 07:02:00`

---

## Log-State Interpretation

Recent log output was consistent with refresh completion of the difficulty-model phase.

Observed relevant messages included:

- `Processed and serialized 9407 charts from 1255 files.`
- `Successfully built feature dataset with 9407 charts at data/ssc_refresh_work\ffr_data\dataset.csv`
- `Best model for 'dance-double' has R^2 score: 0.948`
- `Saved trained model for 'dance-double' to data/ssc_refresh_work\ffr_models\dance-double.p`
- `Best model for 'dance-single' has R^2 score: 0.967`
- `Saved trained model for 'dance-single' to data/ssc_refresh_work\ffr_models\dance-single.p`

Interpretation:

- the refreshed difficulty-evaluator dataset build completed
- both per-mode floating-point regressors completed successfully
- the refreshed FFR artifacts are now present on disk

---

## Practical Interpretation

This completion state shows that the refresh achieved the user-requested practical retraining plan at a meaningful level.

The observed completed sequence is now:

1. onset checkpoint set completed
2. practical single-mode buckets completed:
   - Easy
   - Medium
   - Hard
   - Challenge
3. practical double-mode buckets completed:
   - Easy
   - Medium
   - Hard
   - Challenge
4. refreshed floating-point difficulty-model artifacts completed:
   - `dance-single.p`
   - `dance-double.p`

That means the `.ssc`-inclusive refresh has visibly traversed the full practical 8-bucket placement-model plan and completed the refreshed two-mode floating difficulty-model training phase.

---

## Important Notes

### 1. Practical-bucket completion is now directly visible

Earlier snapshots inferred likely completion as the run traversed the curriculum. This report is the first one where the filesystem state itself clearly shows the full placement-bucket inventory complete across all practical single/double buckets.

### 2. Difficulty-model completion is now directly visible

Earlier snapshots had no `ffr_models/` directory. This report is the first one where both final `.p` artifacts are present.

### 3. Large artifacts remain local

These generated model artifacts are local working outputs. They should still follow the previously established artifact/publication strategy rather than being casually pushed as heavyweight binaries.

---

## Recommended Next Actions

1. Document/export a clean refreshed deployment-ready model bundle
2. Verify ArrowVortex integration against the refreshed artifacts
3. Record any packaging or deployment mapping needed between:
   - practical placement-model checkpoints
   - onset checkpoints
   - FFR `.p` artifacts
4. Optionally perform a post-refresh validation pass against a few representative songs/audio inputs
5. Optionally later normalize the quarantined legacy files under `ddc_stepmania/`

---

## Bottom Line

The `.ssc`-inclusive refresh now appears complete for the core user-requested retraining objective:

- practical 8-bucket placement retraining completed
- onset retraining completed
- refreshed floating-point difficulty-model retraining completed for both single and double

This is the key completion milestone for the refresh cycle.
