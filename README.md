# Dance Dance Convolution

Modernized pipeline for DDR chart generation (Python 3 / TensorFlow 2).

## Installation

1. Clone repository with submodules:
   ```bash
   git clone --recursive <repo_url>
   cd ddc
   ```
2. Install the package:
   ```bash
   pip install .
   ```

## Usage

### AutoChart CLI

Generate stepcharts for a song or directory of songs:

```bash
python3 autochart.py path/to/song.mp3 --models_dir path/to/models --ffr_dir ffr-difficulty-model/
```

Or for a directory:
```bash
python3 autochart.py path/to/songs/ --models_dir path/to/models
```

Options:
- `--out_dir`: Output directory (default: `output`)
- `--ffr_dir`: Path to trained FFR models (optional, for difficulty rating)
- `--google_key`: Google API Key (optional, for image search)
- `--cx`: Google Custom Search ID (optional, for image search)

### Library Usage (ArrowVortex Integration)

You can import `AutoChart` as a library:

```python
from infer.autochart_lib import AutoChart

ac = AutoChart(models_dir='models_v2')
ac.process_song('mysong.mp3', 'output_dir')
```

## Training

To retrain all models (Onset, SymNet, FFR):

```bash
python3 scripts/train_all.py /path/to/sm_packs /path/to/work_dir
```

## Features
- **Modernized:** Python 3 & TensorFlow 2.x support.
- **Onset Detection:** Uses `ddc_onset` (PyTorch) for high-quality onsets, falling back to Librosa.
- **Step Generation:** Generates Single and Double charts for 5 difficulties.
- **Difficulty Rating:** Integrates `ffr-difficulty-model` to rate generated charts.
- **Metadata:** Extracts album art from MP3s or fetches via Google API.

## Credits
- Original DDC: Chris Donahue et al.
- DDC Onset Port: Robert Pelloni
- FFR Difficulty Model: Robert Pelloni
