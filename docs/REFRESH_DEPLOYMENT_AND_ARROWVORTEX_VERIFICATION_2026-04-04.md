# SSC Refresh Deployment and ArrowVortex Verification

Date: 2026-04-04

## Purpose

This document records post-refresh deployment analysis after the `.ssc`-inclusive retraining run reached completion-state artifacts.

It focuses on two practical questions:

1. whether the refreshed outputs match the current inference/runtime expectations
2. how to package the refreshed outputs into a cleaner deployment bundle for AutoChart and ArrowVortex-oriented usage

---

## Verification Summary

The refreshed artifacts under `data/ssc_refresh_work/` are structurally compatible with the current runtime path.

### Verified runtime paths

Current refreshed artifact roots:

- placement/onset models: `data/ssc_refresh_work/models`
- floating-point difficulty models: `data/ssc_refresh_work/ffr_models`

### Verified inference expectations

`infer/autochart_lib.py` expects:

- per-placement-bucket directories under `models_dir`
- `model_10.pth` preferred, with fallback to earlier `model_XX.pth`
- `vocab.json` in each practical placement bucket directory
- optional FFR model directory containing `.p` files for the per-mode predictor

Observed refreshed filesystem state matches those expectations:

- practical bucket dirs exist under `data/ssc_refresh_work/models`
- each practical bucket directory contains `vocab.json`
- each practical bucket now contains `model_10.pth`
- `data/ssc_refresh_work/ffr_models` contains:
  - `dance-single.p`
  - `dance-double.p`

### Verified initialization probe

A direct initialization probe succeeded with:

- `models_dir='data/ssc_refresh_work/models'`
- `ffr_model_dir='data/ssc_refresh_work/ffr_models'`

Observed result:

- `AutoChart` initialized successfully
- FFR predictor loaded successfully
- predictor reported 2 models loaded: `dance-double`, `dance-single`

Interpretation:

- the refreshed model layout is compatible with the current AutoChart runtime wiring
- the refreshed FFR artifacts are discoverable by the mode-agnostic difficulty predictor

---

## ArrowVortex-Oriented Usage Mapping

The current `infer/ddc_server.py` accepts:

- `--models_dir`
- `--ffr_dir`

That means the refreshed local artifacts can be used directly with the current server interface via:

```bash
python infer/ddc_server.py --models_dir data/ssc_refresh_work/models --ffr_dir data/ssc_refresh_work/ffr_models --port 8080
```

And the CLI can use the same refreshed outputs via:

```bash
python autochart.py path/to/song.mp3 --models_dir data/ssc_refresh_work/models --ffr_dir data/ssc_refresh_work/ffr_models
```

Interpretation:

- no additional code changes were required just to point ArrowVortex-oriented server usage at the refreshed artifacts
- the main remaining work is packaging/export convenience, not fundamental runtime incompatibility

---

## Bundle Packaging Strategy

To support cleaner deployment without shipping the full work directory, a new helper script was added:

- `scripts/package_refresh_bundle.py`

### What the script does

It packages a completed refresh work directory into a cleaner deployment bundle layout containing:

- `models/onset/`
- all 8 practical placement bucket directories with:
  - `vocab.json`
  - latest checkpoint by default (or all checkpoints if desired)
- `ffr_models/`
  - `dance-single.p`
  - `dance-double.p`
- `MANIFEST.json`
- `README.txt`

### Example usage

Dry-run verification used during this pass:

```bash
python scripts/package_refresh_bundle.py data/ssc_refresh_work C:/Users/hyper/AppData/Local/Temp/ssc_refresh_bundle --latest_only --dry_run
```

Observed dry-run result:

- planned bundle contained the onset checkpoint
- planned bundle contained all 8 practical bucket dirs
- planned bundle contained latest `model_10.pth` for all completed practical buckets
- planned bundle contained both FFR `.p` files
- planned manifest and README generation succeeded in dry-run mode

Interpretation:

- the script provides a straightforward local-only export path for deployment packaging
- it avoids pushing heavyweight generated artifacts while still documenting a reproducible export flow

---

## Recommended Deployment Layout

For a clean deployment-oriented local bundle, the preferred exported layout is:

```text
<bundle>/
  models/
    onset/
    dance-single_Easy/
    dance-single_Medium/
    dance-single_Hard/
    dance-single_Challenge/
    dance-double_Easy/
    dance-double_Medium/
    dance-double_Hard/
    dance-double_Challenge/
  ffr_models/
    dance-single.p
    dance-double.p
  MANIFEST.json
  README.txt
```

This layout aligns directly with the current runtime interface and is cleaner than exposing the entire training work directory.

---

## Practical Conclusions

### 1. Refreshed runtime compatibility looks good

The refreshed outputs are structurally compatible with the current inference/runtime path.

### 2. ArrowVortex-oriented server usage is already path-compatible

The current server can point directly at:

- `data/ssc_refresh_work/models`
- `data/ssc_refresh_work/ffr_models`

### 3. Packaging convenience is the next post-refresh step

The newly added bundle-packaging script now gives a local-only, non-destructive path to exporting a cleaner deployment bundle.

---

## Recommended Next Actions

1. Run `scripts/package_refresh_bundle.py` without `--dry_run` to create a local deployment bundle when ready
2. Run a small end-to-end AutoChart/ArrowVortex-oriented smoke test against the refreshed paths or packaged bundle
3. Decide whether any final artifact publication/distribution strategy is desired for the refreshed models
4. Optionally add an explicit deployment README section once a final artifact location is chosen
