# DDR Corpus Audit

Date: 2026-04-04

## Scope

- Raw directory: `data/raw/ddr_official`
- Extracted JSON directory: `data/training_output/json_raw/ddr_official`

## Raw File Inventory

- `.sm` files found: **1235**
- `.ssc` files found: **20**

### `.ssc` files present but not currently used by extraction

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

- Songs audited: **1234**
- Charts audited: **9241**
- Charts containing any non-binary note symbol (`not just 0/1`): **8502**
- Rows containing any non-binary note symbol: **263509**

### Chart types

- `dance-couple`: 3
- `dance-double`: 4017
- `dance-single`: 5221

### Coarse difficulties

- `Beginner`: 1207
- `Challenge`: 795
- `Easy`: 2413
- `Hard`: 2413
- `Medium`: 2413

### Note-character frequency

- `0`: 25510544
- `1`: 2793758
- `2`: 159787
- `3`: 159787
- `M`: 41188

### Most common special rows

- `0002`: 13293
- `2000`: 13233
- `0020`: 11641
- `0200`: 11149
- `00002000`: 9220
- `00020000`: 8960
- `0003`: 5883
- `3000`: 5681
- `3003`: 5643
- `0030`: 5624
- `0300`: 5472
- `2002`: 5034
- `00200000`: 4295
- `00000200`: 4225
- `02000000`: 4068
- `00000020`: 3997
- `00003000`: 3729
- `MMMM`: 3555
- `00030000`: 3519
- `00000002`: 3379
- `20000000`: 3364
- `3001`: 3177
- `0130`: 3150
- `1003`: 3122
- `0310`: 3115
- `MMMMMMMM`: 2997
- `00033000`: 2973
- `0103`: 2604
- `00022000`: 2568
- `3100`: 2492
- `00013000`: 2418
- `00031000`: 2412
- `0330`: 2251
- `3010`: 2129
- `0013`: 1976
- `0220`: 1921
- `0301`: 1835
- `00000300`: 1820
- `1030`: 1809
- `03000000`: 1801

## Per-bucket summary

| Bucket | Charts | Charts with special symbols | Character counts |
|---|---:|---:|---|
| `dance-couple / Easy` | 1 | 0 | `0`=1373, `1`=419 |
| `dance-couple / Hard` | 1 | 0 | `0`=4254, `1`=610 |
| `dance-couple / Medium` | 1 | 0 | `0`=2210, `1`=478 |
| `dance-double / Beginner` | 1 | 0 | `0`=1525, `1`=107 |
| `dance-double / Challenge` | 398 | 390 | `0`=2575057, `1`=207277, `2`=10545, `3`=10545, `M`=23088 |
| `dance-double / Easy` | 1206 | 1144 | `0`=2951618, `1`=230884, `2`=17173, `3`=17173 |
| `dance-double / Hard` | 1206 | 1166 | `0`=6407330, `1`=514072, `2`=23917, `3`=23917, `M`=2220 |
| `dance-double / Medium` | 1206 | 1176 | `0`=4568037, `1`=363665, `2`=22863, `3`=22863, `M`=1660 |
| `dance-single / Beginner` | 1206 | 728 | `0`=1325189, `1`=124551, `2`=5922, `3`=5922 |
| `dance-single / Challenge` | 397 | 392 | `0`=1213190, `1`=215576, `2`=11441, `3`=11441, `M`=12312 |
| `dance-single / Easy` | 1206 | 1153 | `0`=1352105, `1`=233065, `2`=18203, `3`=18203 |
| `dance-single / Hard` | 1206 | 1172 | `0`=2991152, `1`=530448, `2`=26032, `3`=26032, `M`=1080 |
| `dance-single / Medium` | 1206 | 1181 | `0`=2117504, `1`=372606, `2`=23691, `3`=23691, `M`=828 |

## Interpretation

- The current extractor/train path uses `.sm` data and ignores `.ssc` files.
- The extracted corpus clearly contains non-binary symbols beyond `0` and `1`, notably `2`, `3`, and `M`.
- This means the DDC symbolic corpus is not tap-only.
- However, the difficulty evaluator currently reduces charts to tap notes only, so non-tap object semantics are not fully modeled there.

## Recommended next actions

1. Add `.ssc` ingestion to extraction so the remaining 20 official-pack files are included.
2. Add a note-object semantic audit mapping for `2`, `3`, `M`, and any additional symbols encountered in future packs.
3. Extend the difficulty evaluator so it captures object types beyond taps.
