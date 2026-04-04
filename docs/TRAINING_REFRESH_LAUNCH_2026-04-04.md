# SSC-Inclusive Training Refresh Launch Record

Date: 2026-04-04

## Purpose

This document records the actual launch of the `.ssc`-inclusive refresh training run after the repository/documentation/corpus-audit preparation work was completed.

---

## Launch Command

The refresh was launched with the resume-friendly command:

```bash
python scripts/train_all.py data/raw/ddr_official data/ssc_refresh_work --jobs 4 --skip_prepare --skip_feature_extraction --skip_existing_models
```

Launched via `nohup` so it can continue independently of the interactive shell.

---

## Runtime Output Location

Current log file:

- `data/ssc_refresh_training.log`

This path is intentionally outside normal source-controlled artifacts and is suitable for ongoing monitoring.

---

## Launch-Time Observations

At launch time, the prepared refresh workspace already contained:

- refreshed `.ssc`-inclusive filtered buckets
- refreshed extracted audio features (`1254` feature files)
- no existing refresh-model checkpoint directories under `data/ssc_refresh_work/models`
- no existing refresh FFR model files under `data/ssc_refresh_work/ffr_models`

That means the refresh run began from a state where:

- data preparation did not need to be repeated
- feature extraction did not need to be repeated
- model training could begin immediately

---

## Early Log Observations

Early runtime output showed the refresh successfully entering onset training, including messages consistent with:

- `Using device: cpu`
- `Loading metadata for 980 charts...`
- `Loaded 980 valid charts.`
- `Loading metadata for 122 charts...`
- `Loaded 122 valid charts.`
- `Epoch 1/5 ...`

This confirms the refresh progressed beyond launch and into actual model training work.

---

## Why this matters

This is the first fully prepared launch of the `.ssc`-inclusive refresh path after:

- `.ssc` extraction support was added
- corpus expansion was measured
- note-object semantics were audited
- bucket split deltas were documented
- the FFR loader was upgraded to prefer `.ssc`
- main-path conflict-marker cleanup reduced blocker count to legacy-subtree-only scope

So this launch is the operational handoff point from analysis/preparation into actual refreshed model production.

---

## Monitoring Guidance

Recommended ways to monitor progress:

- inspect `data/ssc_refresh_training.log`
- verify new checkpoint directories appear under `data/ssc_refresh_work/models/`
- verify refreshed difficulty models appear under `data/ssc_refresh_work/ffr_models/`

---

## Expected Next Outputs

If the run completes successfully, expected outputs include:

- refreshed onset checkpoints
- refreshed 8-bucket placement checkpoints
- refreshed `dance-single` difficulty model
- refreshed `dance-double` difficulty model

Those outputs can then be compared against the previously documented pre-refresh corpus counts and exported into a clean deployment bundle.
