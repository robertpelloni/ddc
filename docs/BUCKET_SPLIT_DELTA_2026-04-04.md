# Bucket Split Delta Audit

Date: 2026-04-04

## Scope

- Before directory: `data/training_output/json_filtered`
- After directory: `data/ssc_refresh_work/json_filtered`

## Per-Bucket Split Deltas

| Bucket | Train before | Train after | Train delta | Valid before | Valid after | Valid delta | Test before | Test after | Test delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dance-double_Beginner` | 1 | 1 | +0 | 0 | 0 | +0 | 0 | 0 | +0 |
| `dance-double_Challenge` | 318 | 327 | +9 | 39 | 40 | +1 | 41 | 42 | +1 |
| `dance-double_Easy` | 964 | 980 | +16 | 120 | 122 | +2 | 122 | 124 | +2 |
| `dance-double_Hard` | 964 | 980 | +16 | 120 | 122 | +2 | 122 | 124 | +2 |
| `dance-double_Medium` | 964 | 980 | +16 | 120 | 122 | +2 | 122 | 124 | +2 |
| `dance-single_Beginner` | 964 | 980 | +16 | 120 | 122 | +2 | 122 | 124 | +2 |
| `dance-single_Challenge` | 317 | 326 | +9 | 39 | 40 | +1 | 41 | 42 | +1 |
| `dance-single_Easy` | 964 | 980 | +16 | 120 | 122 | +2 | 122 | 124 | +2 |
| `dance-single_Hard` | 964 | 980 | +16 | 120 | 122 | +2 | 122 | 124 | +2 |
| `dance-single_Medium` | 964 | 980 | +16 | 120 | 122 | +2 | 122 | 124 | +2 |

## Practical Highlights

- `dance-single_Easy`: train 964 -> 980 (+16), valid 120 -> 122 (+2), test 122 -> 124 (+2)
- `dance-single_Medium`: train 964 -> 980 (+16), valid 120 -> 122 (+2), test 122 -> 124 (+2)
- `dance-single_Hard`: train 964 -> 980 (+16), valid 120 -> 122 (+2), test 122 -> 124 (+2)
- `dance-single_Challenge`: train 317 -> 326 (+9), valid 39 -> 40 (+1), test 41 -> 42 (+1)
- `dance-double_Easy`: train 964 -> 980 (+16), valid 120 -> 122 (+2), test 122 -> 124 (+2)
- `dance-double_Medium`: train 964 -> 980 (+16), valid 120 -> 122 (+2), test 122 -> 124 (+2)
- `dance-double_Hard`: train 964 -> 980 (+16), valid 120 -> 122 (+2), test 122 -> 124 (+2)
- `dance-double_Challenge`: train 318 -> 327 (+9), valid 39 -> 40 (+1), test 41 -> 42 (+1)

## Interpretation

- This report isolates the downstream training-input delta after `.ssc` support and refreshed bucket preparation.
- It is useful for confirming that the expanded corpus meaningfully increases the exact split files consumed by the practical DDC training plan.
