# SSC-Inclusive Retraining Refresh Plan

Date: 2026-04-04

## Purpose

This document records the exact next recommended execution phase after:

- adding `.ssc` extraction support
- expanding the FFR loader to prefer `.ssc`
- auditing corpus growth
- auditing note-object semantics

The goal is to refresh the trained model set against the larger recovered corpus.

---

## Why a refresh is now justified

Measured corpus improvements already documented elsewhere:

- songs: **1234 -> 1254**
- charts: **9241 -> 9403**
- every practical single/double Easy/Medium/Hard/Challenge bucket increased
- additional hold-head / tail / mine-bearing rows were recovered

That means the current trained model artifacts are now behind the best available extracted dataset.

---

## Recommended refresh workflow

## 1. Prepare a fresh work directory

Suggested work directory example:

```bash
python scripts/prepare_data.py data/raw/ddr_official data/ssc_refresh_work
```

## 2. Extract refreshed audio features

```bash
python learn/extract_feats_v2.py data/ssc_refresh_work/all_jsons.txt data/ssc_refresh_work/feats --jobs 4
```

## 3. Retrain DDC onset + placement models

Primary practical target buckets:

- `dance-single_Easy`
- `dance-single_Medium`
- `dance-single_Hard`
- `dance-single_Challenge`
- `dance-double_Easy`
- `dance-double_Medium`
- `dance-double_Hard`
- `dance-double_Challenge`

Onset remains anchored to `dance-single_Hard` unless/until a broader onset strategy is adopted.

## 4. Retrain difficulty evaluator

Use the refreshed FFR loader path so `.ssc`-recoverable songs are included.

## 5. Re-export deployable model bundle

After successful retraining:

- copy final placement/onset checkpoints into a clean export directory
- copy refreshed difficulty models
- write a manifest / version note for ArrowVortex deployment

---

## Recommended validation checks after refresh

- compare bucket counts against the documented `.ssc`-inclusive numbers
- verify both difficulty modes still train:
  - `dance-single`
  - `dance-double`
- verify note-object-bearing rows are still present in the symbolic training corpus
- verify inference can still load the refreshed model layout

---

## Risks / Caveats

- the local environment currently uses a PyTorch-oriented path for the DDC training/inference work done here
- large generated artifacts should remain out of normal git history unless a dedicated artifact strategy is used
- the difficulty evaluator is still tap-centric in feature semantics even though the corpus itself contains holds/tails/mines

---

## Recommended follow-up after retraining

1. add hold/mine-aware difficulty features
2. consider optional Beginner-model refresh for `dance-single_Beginner`
3. package a clean ArrowVortex-ready model release
