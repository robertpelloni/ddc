# Dance Dance Convolution

Modernized pipeline for DDR chart generation and analysis.

## Overview

This repository contains a modernized DDC workflow for:

- official DDR corpus download/extraction
- dataset preparation and filtering
- audio feature extraction
- DDC onset + placement-model training
- difficulty-evaluator training
- inference / AutoChart generation
- ArrowVortex-oriented integration work

Recent work in this repository also added:

- `.ssc` extraction support in addition to `.sm`
- corpus auditing for note-symbol coverage
- note-object semantic auditing
- refreshed PyTorch-oriented training/inference wiring for the active environment

## Installation

1. Clone with submodules:
   ```bash
   git clone --recursive <repo_url>
   cd ddc
   ```
2. Install dependencies as appropriate for your environment.
3. Install the package if desired:
   ```bash
   pip install .
   ```

## Usage

### AutoChart CLI

Generate stepcharts for a song or directory of songs:

```bash
python autochart.py path/to/song.mp3 --models_dir path/to/models --ffr_dir path/to/ffr_models
```

For the refreshed local `.ssc`-inclusive run, the current direct-use paths are:

```bash
python autochart.py path/to/song.mp3 --models_dir data/ssc_refresh_work/models --ffr_dir data/ssc_refresh_work/ffr_models
```

Or for a directory:

```bash
python autochart.py path/to/songs/ --models_dir path/to/models
```

Options include:
- `--out_dir`: output directory
- `--ffr_dir`: path to trained difficulty models
- `--google_key`: optional Google API key for image lookup
- `--cx`: optional Google Custom Search ID

### Library Usage

```python
from infer.autochart_lib import AutoChart

ac = AutoChart(models_dir='models_dir', ffr_model_dir='ffr_models')
ac.process_song('mysong.mp3', 'output_dir')
```

## Training

To run the end-to-end training pipeline:

```bash
python scripts/train_all.py /path/to/sm_packs /path/to/work_dir
```

## Current Documentation

Key project analysis/report documents:

- `docs/TRAINING_ANALYSIS_2026-04-04.md`
- `docs/CORPUS_AUDIT_2026-04-04.md`
- `docs/SSC_EXPANSION_ANALYSIS_2026-04-04.md`
- `docs/NOTE_OBJECT_SEMANTICS_2026-04-04.md`
- `docs/TRAINING_REFRESH_COMPLETION_2026-04-04.md`
- `docs/REFRESH_DEPLOYMENT_AND_ARROWVORTEX_VERIFICATION_2026-04-04.md`

## Features

- modernized Python 3 workflow
- official DDR pack download support
- `.sm` and `.ssc` extraction support
- DDC onset detection and step generation pipeline
- floating-point difficulty regression support
- ArrowVortex-oriented server/inference path

## Credits

- Original DDC: Chris Donahue et al.
- Modernization / integration work in this repo: subsequent contributors
- DDC onset port: Robert Pelloni
- FFR difficulty model integration: Robert Pelloni
