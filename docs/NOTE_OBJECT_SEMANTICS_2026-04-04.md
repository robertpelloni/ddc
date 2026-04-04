# Note Object Semantics Audit

Date: 2026-04-04

## Scope

- Extracted JSON directory: `data/training_output/json_raw/ddr_official`
- Charts audited: **9403**

## Symbol Mapping

| Symbol | Interpreted semantic | Present in refreshed corpus? | Raw count |
|---|---|---:|---:|
| `1` | `TAP` | Yes | 2857613 |
| `2` | `HOLD_HEAD` | Yes | 164317 |
| `3` | `TAIL` | Yes | 164317 |
| `4` | `ROLL_HEAD` | No | 0 |
| `A` | `ATTACK` | No | 0 |
| `F` | `FAKE` | No | 0 |
| `K` | `KEYSOUND` | No | 0 |
| `L` | `LIFT` | No | 0 |
| `M` | `MINE` | Yes | 41356 |

## Key Findings

- `1` (tap) dominates the corpus, as expected.
- `2` and `3` are both present in substantial quantities, strongly indicating hold-head/tail usage in the official DDR corpus.
- `M` is present and corresponds to mines.
- `4`, `A`, `F`, `K`, and `L` were not observed in the refreshed extracted corpus audit.
- This means the current extracted corpus clearly contains taps, hold-related structure, and mines, but no observed roll-head, attack, fake, keysound, or lift symbols in the audited official-pack extraction.

## Per-Bucket Counts

| Bucket | Chart count with symbol | Symbol totals |
|---|---|---|
| `dance-couple_Easy` | `1` charts=1 | `1`=419 |
| `dance-couple_Hard` | `1` charts=1 | `1`=610 |
| `dance-couple_Medium` | `1` charts=1 | `1`=478 |
| `dance-double_Beginner` | `1` charts=1 | `1`=107 |
| `dance-double_Challenge` | `1` charts=409, `2` charts=396, `3` charts=396, `M` charts=70 | `1`=214498, `2`=11059, `3`=11059, `M`=23232 |
| `dance-double_Easy` | `1` charts=1226, `2` charts=1164, `3` charts=1164 | `1`=235094, `2`=17560, `3`=17560 |
| `dance-double_Hard` | `1` charts=1226, `2` charts=1186, `3` charts=1186, `M` charts=3 | `1`=525596, `2`=24556, `3`=24556, `M`=2220 |
| `dance-double_Medium` | `1` charts=1226, `2` charts=1196, `3` charts=1196, `M` charts=3 | `1`=371006, `2`=23507, `3`=23507, `M`=1660 |
| `dance-single_Beginner` | `1` charts=1226, `2` charts=747, `3` charts=747 | `1`=126476, `2`=6053, `3`=6053 |
| `dance-single_Challenge` | `1` charts=408, `2` charts=397, `3` charts=397, `M` charts=70 | `1`=223185, `2`=12006, `3`=12006, `M`=12336 |
| `dance-single_Easy` | `1` charts=1226, `2` charts=1173, `3` charts=1173 | `1`=237431, `2`=18588, `3`=18588 |
| `dance-single_Hard` | `1` charts=1226, `2` charts=1192, `3` charts=1192, `M` charts=3 | `1`=542410, `2`=26678, `3`=26678, `M`=1080 |
| `dance-single_Medium` | `1` charts=1226, `2` charts=1201, `3` charts=1201, `M` charts=3 | `1`=380303, `2`=24310, `3`=24310, `M`=828 |

## Example Rows by Symbol

### `2` / `HOLD_HEAD`

- `0002`: 13643
- `2000`: 13575
- `0020`: 11904
- `0200`: 11415
- `00002000`: 9482
- `00020000`: 9219
- `2002`: 5199
- `00200000`: 4394
- `00000200`: 4324
- `02000000`: 4161
- `00000020`: 4083
- `00000002`: 3477
- `20000000`: 3476
- `00022000`: 2642
- `0220`: 1963

### `3` / `TAIL`

- `0003`: 6095
- `3000`: 5882
- `3003`: 5821
- `0030`: 5779
- `0300`: 5642
- `00003000`: 3870
- `00030000`: 3654
- `3001`: 3258
- `0130`: 3212
- `1003`: 3197
- `0310`: 3159
- `00033000`: 3054
- `0103`: 2635
- `3100`: 2520
- `00013000`: 2487

### `M` / `MINE`

- `MMMM`: 3561
- `MMMMMMMM`: 3006
- `MMMM1000`: 131
- `0001MMMM`: 128
- `MMMM0000`: 105
- `0000MMMM`: 102
- `MMMM0100`: 40
- `0010MMMM`: 37
- `0100MMMM`: 36
- `MMMM0010`: 29
- `MMMM0001`: 20
- `1000MMMM`: 18
- `0003MMMM`: 15
- `MMMM3000`: 14
- `0011MMMM`: 11

## Practical Implications

- The DDC symbolic training corpus is not tap-only; it includes hold-heads, tails, and mines.
- The difficulty evaluator still currently reduces charts to tap notes only, so these semantics are not yet fully represented there.
- The next useful improvement is to extend difficulty-feature extraction to account for hold and mine burden explicitly.
