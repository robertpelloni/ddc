# SSC Refresh Readiness Audit

Date: 2026-04-04

## Scope

- Work directory: `data/ssc_refresh_work`

## Current Refresh State

- Filtered bucket directories present: **10**
- Extracted feature files present: **1254**
- Existing PyTorch model checkpoint directories under work dir: **0**
- Existing FFR model files under work dir: **0**

## Bucket Counts

| Bucket | Train | Valid | Test |
|---|---:|---:|---:|
| `dance-double_Beginner` | 1 | 0 | 0 |
| `dance-double_Challenge` | 327 | 40 | 42 |
| `dance-double_Easy` | 980 | 122 | 124 |
| `dance-double_Hard` | 980 | 122 | 124 |
| `dance-double_Medium` | 980 | 122 | 124 |
| `dance-single_Beginner` | 980 | 122 | 124 |
| `dance-single_Challenge` | 326 | 40 | 42 |
| `dance-single_Easy` | 980 | 122 | 124 |
| `dance-single_Hard` | 980 | 122 | 124 |
| `dance-single_Medium` | 980 | 122 | 124 |

## Recommended Resume-Friendly Command

```bash
python scripts/train_all.py data/raw/ddr_official data/ssc_refresh_work --jobs 4 --skip_prepare --skip_feature_extraction --skip_existing_models
```

## Interpretation

- This work directory is prepared specifically for the `.ssc`-inclusive refresh path.
- The recommended command above is resume-friendly and avoids repeating already completed preparation/feature work.
- If model artifacts are already present in the work directory, `--skip_existing_models` prevents redundant retraining for those buckets.
