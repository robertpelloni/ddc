# SSC Expansion Analysis

Date: 2026-04-04

## Purpose

This document summarizes the practical corpus expansion unlocked by adding `.ssc` extraction support to `dataset/extract_json.py` and fixing the `scripts/prepare_data.py` merge-conflict issue so the downstream data-prep path can be rerun cleanly.

This is a delta-focused companion to:

- `docs/TRAINING_ANALYSIS_2026-04-04.md`
- `docs/CORPUS_AUDIT_2026-04-04.md`

---

## Executive Summary

Adding `.ssc` extraction support increased the locally extracted official DDR corpus from:

- **1234 songs** to **1254 songs** (**+20**)
- **9241 charts** to **9403 charts** (**+162**)

This confirms that all 20 `.ssc`-only song folders identified in the raw corpus can now be recovered by the extractor.

The expansion is not just a song-count improvement. It also increases chart coverage across the main single/double difficulty buckets and slightly increases coverage of non-binary note-object rows.

---

## Corpus-Level Delta

| Metric | Before `.ssc` extraction | After `.ssc` extraction | Delta |
|---|---:|---:|---:|
| Songs audited | 1234 | 1254 | +20 |
| Charts audited | 9241 | 9403 | +162 |
| Charts with non-binary symbols | 8502 | 8663 | +161 |
| Rows with non-binary symbols | 263509 | 270527 | +7018 |
| `0` count | 25,510,544 | 26,202,933 | +692,389 |
| `1` count | 2,793,758 | 2,857,613 | +63,855 |
| `2` count | 159,787 | 164,317 | +4,530 |
| `3` count | 159,787 | 164,317 | +4,530 |
| `M` count | 41,188 | 41,356 | +168 |

---

## Chart-Type Delta

| Chart type | Before | After | Delta |
|---|---:|---:|---:|
| `dance-single` | 5221 | 5312 | +91 |
| `dance-double` | 4017 | 4088 | +71 |
| `dance-couple` | 3 | 3 | 0 |

The `.ssc` additions therefore improve both target gameplay modes that matter for the DDC training/export plan.

---

## Coarse-Difficulty Delta

| Difficulty | Before | After | Delta |
|---|---:|---:|---:|
| `Beginner` | 1207 | 1227 | +20 |
| `Easy` | 2413 | 2453 | +40 |
| `Medium` | 2413 | 2453 | +40 |
| `Hard` | 2413 | 2453 | +40 |
| `Challenge` | 795 | 817 | +22 |

This matters because the `.ssc` expansion improves every major practical difficulty bucket instead of only one isolated area.

---

## Per-Bucket Delta for Main DDC Targets

These are the primary DDC placement targets used by the practical 8-bucket plan.

| Bucket | Before | After | Delta |
|---|---:|---:|---:|
| `dance-single_Easy` | 1206 | 1226 | +20 |
| `dance-single_Medium` | 1206 | 1226 | +20 |
| `dance-single_Hard` | 1206 | 1226 | +20 |
| `dance-single_Challenge` | 397 | 408 | +11 |
| `dance-double_Easy` | 1206 | 1226 | +20 |
| `dance-double_Medium` | 1206 | 1226 | +20 |
| `dance-double_Hard` | 1206 | 1226 | +20 |
| `dance-double_Challenge` | 398 | 409 | +11 |

### Interpretation

This is the strongest practical takeaway:

- every primary Easy/Medium/Hard bucket gains **20 charts**
- every primary Challenge bucket gains **11 charts**
- both single and double training targets benefit

So the `.ssc` support is materially useful for the actual ArrowVortex-oriented model set.

---

## Non-Binary Symbol Delta

The refreshed extraction confirms that the newly recovered `.ssc` content is not merely redundant tap-only material.

Symbol-count deltas:

- `2`: +4,530
- `3`: +4,530
- `M`: +168

This means `.ssc` support also improves the amount of special-object-bearing symbolic data available to the DDC corpus.

---

## Operational Finding: `prepare_data.py`

During follow-on work, `scripts/prepare_data.py` was found to still contain unresolved merge-conflict content. That has now been corrected.

Practical significance:

- the extractor now supports `.ssc`
- the top-level data-prep helper is no longer syntactically broken
- rerunning downstream filtering/splitting on the expanded corpus is now straightforward

---

## What is still pending

Although extraction coverage improved, a full downstream model refresh against the expanded corpus is still the next major step.

That means rerunning:

1. filtered bucket generation
2. feature extraction
3. onset training
4. 8-bucket SymNet training
5. difficulty-evaluator retraining

Until that is done, the current trained model artifacts still reflect the pre-`.ssc` extraction corpus.

---

## Recommended Next Step

The next highest-value action is:

### Full retraining against the `.ssc`-inclusive corpus

Why this is now justified:

- the expansion is real, measured, and nontrivial
- the recovered charts touch every major practical training bucket
- the recovered data includes additional non-binary note-object rows
- both single and double coverage improve

---

## Bottom Line

The `.ssc` extraction work was worth doing.

It recovered:

- **20 additional songs**
- **162 additional charts**
- **7018 additional non-binary-symbol rows**

and it improves every one of the practical single/double Easy/Medium/Hard/Challenge training targets used by the DDC export plan.
