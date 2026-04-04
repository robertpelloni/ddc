# SSC-Inclusive Training Refresh Progress Snapshot #2

Date: 2026-04-04

## Purpose

This document records a later in-flight progress snapshot for the active `.ssc`-inclusive refresh training run.

It follows:

- `docs/TRAINING_REFRESH_LAUNCH_2026-04-04.md`
- `docs/TRAINING_REFRESH_PROGRESS_2026-04-04.md`

---

## Runtime Snapshot

At this later snapshot, the refresh run was still active.

Observed process state:

- at least one active Python training process was still present for the refresh run

Current monitored log:

- `data/ssc_refresh_training.log`

---

## Observed progression since prior snapshot

### Onset stage

The onset stage has advanced beyond the earlier mid-run state.

Observed checkpoint state at this snapshot:

- `data/ssc_refresh_work/models/onset/model_01.pth`
- `data/ssc_refresh_work/models/onset/model_02.pth`
- `data/ssc_refresh_work/models/onset/model_03.pth`
- `data/ssc_refresh_work/models/onset/model_04.pth`
- `data/ssc_refresh_work/models/onset/model_05.pth`

This strongly indicates the onset stage has completed its 5-epoch checkpoint production.

### Transition into SymNet bucket training

The active log had moved on from onset-only messages and was now showing output consistent with bucketed SymNet training.

Observed lines included:

- `Using device: cpu`
- `Building vocabulary...`
- `Vocab Size: 79`
- `Loading metadata for 980 charts...`
- `Loaded 980 valid charts.`
- `Loading metadata for 122 charts...`
- `Loaded 122 valid charts.`
- `Epoch 1/10 ...`

This is consistent with the first major bucketed placement-model training stage beginning after onset checkpoint generation.

---

## Artifact Snapshot

### Models directory

Observed under:

- `data/ssc_refresh_work/models/`

State observed:

- `onset/` contains 5 checkpoints
- `dance-single_Easy/` directory exists
- no finished SymNet `.pth` checkpoints were yet observed inside `dance-single_Easy/` at the exact snapshot moment

Interpretation:

- the refresh had progressed into the first practical 8-bucket placement run
- the first placement bucket appeared to be in progress rather than completed at the exact snapshot time

### Difficulty-model directory

Observed under:

- `data/ssc_refresh_work/ffr_models/`

State observed:

- no `.p` files yet

Interpretation:

- the run had not yet reached the final difficulty-model training stage at the exact time of this snapshot

---

## Practical interpretation

This later snapshot shows the refresh is progressing in a healthy staged sequence:

1. onset training launched
2. onset checkpoints accumulated through the expected 5-epoch set
3. the run transitioned into bucketed SymNet training
4. difficulty-model training had not yet started at snapshot time

That is exactly the sort of staged behavior expected from the configured orchestration.

---

## Recommended next monitoring actions

1. Continue monitoring `data/ssc_refresh_training.log`
2. Watch for completed checkpoint sets under:
   - `data/ssc_refresh_work/models/dance-single_Easy/`
   - the remaining practical 8 buckets
3. Watch for eventual `.p` outputs under `data/ssc_refresh_work/ffr_models/`
4. Once the run completes, document final artifact inventory and export recommendations
