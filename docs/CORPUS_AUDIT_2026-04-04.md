# DDR Corpus Audit

Date: 2026-04-04

## Scope

- Raw directory: `data/raw/ddr_official`
- Extracted JSON directory: `data/training_output/json_raw/ddr_official`

## Raw File Inventory

- `.sm` files found: **1235**
- `.ssc` files found: **20**

### `.ssc` files present in the raw corpus

- `data/raw/ddr_official\ACE FOR ACES\ACE FOR ACES.ssc`
- `data/raw/ddr_official\Burstix Comet\Burstix Comet.ssc`
- `data/raw/ddr_official\CHAOS Terror-Tech Mix\CHAOS Terror-Tech Mix.ssc`
- `data/raw/ddr_official\Casino fire Kotomi-chan (cover)\Casino fire Kotomi-chan (cover).ssc`
- `data/raw/ddr_official\Dopamine (STARDOM Remix)\Dopamine (STARDOM Remix).ssc`
- `data/raw/ddr_official\EYE OF THE HEAVEN\EYE OF THE HEAVEN.ssc`
- `data/raw/ddr_official\Florence\Florence.ssc`
- `data/raw/ddr_official\GO!\GO!.ssc`
- `data/raw/ddr_official\Hou (Five Flares Mix)\Hou (Five Flares Mix).ssc`
- `data/raw/ddr_official\Meikyuu no rondo\Meikyuu no rondo.ssc`
- `data/raw/ddr_official\Monsters Den\Monsters Den.ssc`
- `data/raw/ddr_official\Rabbit hole\Rabbit hole.ssc`
- `data/raw/ddr_official\SCREW owo SCREW\SCREW owo SCREW.ssc`
- `data/raw/ddr_official\Shippuujinrai\Shippuujinrai.ssc`
- `data/raw/ddr_official\Shukusei!! Loli-kami requiem\Shukusei!! Loli-kami requiem.ssc`
- `data/raw/ddr_official\Stand Alone Beat Masta\Stand Alone Beat Masta.ssc`
- `data/raw/ddr_official\Steps for Victory\Steps for Victory.ssc`
- `data/raw/ddr_official\Superior MAXXX\Superior MAXXX.ssc`
- `data/raw/ddr_official\Yuusha\Yuusha.ssc`
- `data/raw/ddr_official\ZENDEGI DANCE\ZENDEGI DANCE.ssc`

## Extracted JSON Audit

- Songs audited: **1254**
- Charts audited: **9403**
- Charts containing any non-binary note symbol (`not just 0/1`): **8663**
- Rows containing any non-binary note symbol: **270527**

### Chart types

- `dance-couple`: 3
- `dance-double`: 4088
- `dance-single`: 5312

### Coarse difficulties

- `Beginner`: 1227
- `Challenge`: 817
- `Easy`: 2453
- `Hard`: 2453
- `Medium`: 2453

### Note-character frequency

- `0`: 26202933
- `1`: 2857613
- `2`: 164317
- `3`: 164317
- `M`: 41356

### Most common special rows

- `0002`: 13643
- `2000`: 13575
- `0020`: 11904
- `0200`: 11415
- `00002000`: 9482
- `00020000`: 9219
- `0003`: 6095
- `3000`: 5882
- `3003`: 5821
- `0030`: 5779
- `0300`: 5642
- `2002`: 5199
- `00200000`: 4394
- `00000200`: 4324
- `02000000`: 4161
- `00000020`: 4083
- `00003000`: 3870
- `00030000`: 3654
- `MMMM`: 3561
- `00000002`: 3477
- `20000000`: 3476
- `3001`: 3258
- `0130`: 3212
- `1003`: 3197
- `0310`: 3159
- `00033000`: 3054
- `MMMMMMMM`: 3006
- `00022000`: 2642
- `0103`: 2635
- `3100`: 2520
- `00013000`: 2487
- `00031000`: 2469
- `0330`: 2298
- `3010`: 2165
- `0013`: 2008
- `0220`: 1963
- `00000300`: 1891
- `0301`: 1864
- `03000000`: 1856
- `00300000`: 1854

## Per-bucket summary

| Bucket | Charts | Charts with special symbols | Character counts |
|---|---:|---:|---|
| `dance-couple / Easy` | 1 | 0 | `0`=1373, `1`=419 |
| `dance-couple / Hard` | 1 | 0 | `0`=4254, `1`=610 |
| `dance-couple / Medium` | 1 | 0 | `0`=2210, `1`=478 |
| `dance-double / Beginner` | 1 | 0 | `0`=1525, `1`=107 |
| `dance-double / Challenge` | 409 | 401 | `0`=2695720, `1`=214498, `2`=11059, `3`=11059, `M`=23232 |
| `dance-double / Easy` | 1226 | 1164 | `0`=3011082, `1`=235094, `2`=17560, `3`=17560 |
| `dance-double / Hard` | 1226 | 1186 | `0`=6567552, `1`=525596, `2`=24556, `3`=24556, `M`=2220 |
| `dance-double / Medium` | 1226 | 1196 | `0`=4676560, `1`=371006, `2`=23507, `3`=23507, `M`=1660 |
| `dance-single / Beginner` | 1226 | 747 | `0`=1352170, `1`=126476, `2`=6053, `3`=6053 |
| `dance-single / Challenge` | 408 | 403 | `0`=1275947, `1`=223185, `2`=12006, `3`=12006, `M`=12336 |
| `dance-single / Easy` | 1226 | 1173 | `0`=1379273, `1`=237431, `2`=18588, `3`=18588 |
| `dance-single / Hard` | 1226 | 1192 | `0`=3066378, `1`=542410, `2`=26678, `3`=26678, `M`=1080 |
| `dance-single / Medium` | 1226 | 1201 | `0`=2168889, `1`=380303, `2`=24310, `3`=24310, `M`=828 |

## Interpretation

- The extractor now supports `.ssc` in addition to `.sm`; this audit reflects a refreshed JSON extraction that includes those files.
- The extracted corpus clearly contains non-binary symbols beyond `0` and `1`, notably `2`, `3`, and `M`.
- This means the DDC symbolic corpus is not tap-only.
- However, the difficulty evaluator currently reduces charts to tap notes only, so non-tap object semantics are not fully modeled there.

## Recommended next actions

1. Re-run downstream filtering/training so the newly recovered `.ssc` songs are included in model training.
2. Add a note-object semantic audit mapping for `2`, `3`, `M`, and any additional symbols encountered in future packs.
3. Extend the difficulty evaluator so it captures object types beyond taps.
