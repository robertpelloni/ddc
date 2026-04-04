# SSC-Inclusive Training Refresh Progress Snapshot #14

Date: 2026-04-04

## Purpose

This document records a major later-stage in-flight milestone for the active `.ssc`-inclusive refresh run.

The key update in this snapshot is that the refresh has clearly moved far beyond the earlier second-bucket monitoring window:

- `dance-single_Medium` now appears complete
- `dance-single_Hard` now appears complete
- `dance-single_Challenge` now appears complete
- `dance-double_Easy` is now the active frontier and has advanced to at least 8 checkpoints

This is the strongest monitoring milestone yet for the practical bucket retraining sequence.

---

## Runtime Snapshot

At this snapshot time:

- the active refresh process was still present
- the main refresh log was still updating
- the refresh had advanced well beyond the single-mode Easy/Medium stages
- the current active log window was consistent with a later bucket remaining in progress
- three active Python processes were visible at monitoring time

Current monitored log:

- `data/ssc_refresh_training.log`

Observed active process state:

- three active Python processes were visible
- no process interruption or cleanup was performed

Interpretation:

- the training environment remains busy
- the user instruction not to kill processes was preserved
- the refresh continued normally during this observation window

---

## Observed progression since prior snapshot

### Onset stage

Observed state remained stable and consistent with a completed onset checkpoint set:

- `data/ssc_refresh_work/models/onset/model_01.pth`
- `data/ssc_refresh_work/models/onset/model_02.pth`
- `data/ssc_refresh_work/models/onset/model_03.pth`
- `data/ssc_refresh_work/models/onset/model_04.pth`
- `data/ssc_refresh_work/models/onset/model_05.pth`

Latest observed onset checkpoint mtime:

- `model_05.pth` → `2026-04-04 02:50:21`

### Single-mode practical buckets

Observed state now showed the following apparent completions:

#### `dance-single_Easy`
- 10 checkpoints
- latest: `model_10.pth` → `2026-04-04 03:28:13`

#### `dance-single_Medium`
- 10 checkpoints
- latest: `model_10.pth` → `2026-04-04 03:58:00`

#### `dance-single_Hard`
- 10 checkpoints
- latest: `model_10.pth` → `2026-04-04 04:29:43`

#### `dance-single_Challenge`
- 10 checkpoints
- latest: `model_10.pth` → `2026-04-04 04:59:49`

Interpretation:

- the full practical single-mode sequence now appears complete across Easy / Medium / Hard / Challenge
- the refresh has therefore progressed through all four practical single-mode buckets
- this is a major milestone in the user-requested practical retraining plan

### Double-mode practical buckets

A double-mode frontier is now clearly active:

#### `dance-double_Easy`
Observed state at this snapshot:

- 8 checkpoints
- latest: `model_08.pth` → `2026-04-04 05:21:49`

Interpretation:

- the refresh has now transitioned into double-mode practical bucket training
- `dance-double_Easy` is the newest artifact frontier observed at this snapshot
- the pipeline is no longer confined to single-mode buckets

---

## Log-state interpretation

Recent log output was consistent with an active later practical bucket still training.

Observed examples included:

- `Val Loss: 2.4303, Val Acc: 0.3408`
- `Epoch 9/10 ...`

Interpretation:

- the active bucket currently represented in the log remains in progress
- given the artifact frontier, the most plausible active stage is `dance-double_Easy`
- the refresh has advanced beyond single-mode practical buckets and is continuing through the double-mode side of the practical plan

This is materially more important than earlier single-bucket monitoring snapshots because the run is now visibly traversing multiple completed practical buckets across the planned curriculum.

---

## Artifact Snapshot

### Models directory

Observed relevant state under `data/ssc_refresh_work/models/`:

- `onset/` contains 5 checkpoints
- `dance-single_Easy/` contains 10 checkpoints
- `dance-single_Medium/` contains 10 checkpoints
- `dance-single_Hard/` contains 10 checkpoints
- `dance-single_Challenge/` contains 10 checkpoints
- `dance-double_Easy/` contains 8 checkpoints

### Difficulty-model directory

Observed relevant state under `data/ssc_refresh_work/ffr_models/`:

- directory still not present at snapshot time
- no refreshed `.p` difficulty-model artifacts yet observed

Interpretation:

- onset stage remains complete and stable
- all four practical single-mode buckets now appear complete
- the refresh has entered double-mode practical training
- the run still has not yet advanced to final difficulty-model artifact production at the moment of this snapshot

---

## Practical interpretation

This is the strongest practical-bucket milestone yet observed during the `.ssc`-inclusive refresh.

The active refresh now visibly supports the following sequence:

1. onset checkpoint set completed
2. `dance-single_Easy` completed
3. `dance-single_Medium` completed
4. `dance-single_Hard` completed
5. `dance-single_Challenge` completed
6. `dance-double_Easy` started and advanced to at least 8 checkpoints

That pattern is strong evidence that the active refresh is successfully traversing the broader 8-bucket practical placement-model retraining plan requested by the user.

---

## Recommended next monitoring actions

1. Continue monitoring `data/ssc_refresh_training.log`
2. Watch for `dance-double_Easy/model_09.pth` and `model_10.pth`
3. Watch for `dance-double_Medium/` to appear after double-Easy completion
4. Watch for later double-mode practical bucket directories (`dance-double_Hard`, `dance-double_Challenge`)
5. Watch for eventual `.p` outputs under `data/ssc_refresh_work/ffr_models/`
6. After full completion, document final artifact inventory and export plan
