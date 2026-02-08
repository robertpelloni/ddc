# Dance Dance Convolution

<<<<<<< HEAD
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
=======
Dance Dance Convolution is an automatic choreography system for Dance Dance Revolution (DDR), converting raw audio into playable dances.

<p align="center">
    <img src="docs/fig1.png" width="650px"/>
</p>
>>>>>>> origin/master_v2

This repository contains the code used to produce the dataset and results in the [Dance Dance Convolution paper](https://arxiv.org/abs/1703.06891). You can find a live demo of our system [here](http://deepx.ucsd.edu/ddc) as well as an example [video](https://www.youtube.com/watch?v=yUc3O237p9M).

<<<<<<< HEAD
### AutoChart CLI

Generate stepcharts for a song or directory of songs:

```bash
python3 autochart.py path/to/song.mp3 --models_dir path/to/models --ffr_dir ffr-difficulty-model/
```

Or for a directory:
```bash
python3 autochart.py path/to/songs/ --models_dir path/to/models
```
=======
The `Fraxtil` and `In The Groove` datasets from the paper are amalgamations of three and two StepMania "packs" respectively. Instructions for downloading these packs and building the datasets can be found below.

This is a streamlined version of the legacy code used to produce our paper (which uses outdated libraries). The legacy code is available at `master_v1` for reproducability.

Please email me with any issues: cdonahue \[@at@\] ucsd \(.dot.\) edu
>>>>>>> origin/master_v2

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

<<<<<<< HEAD
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
=======
# Requirements

* tensorflow >1.0
* numpy
* tqdm
* scipy

# Directory description

* `ddc/`: Core library with dataset extraction and training code
* `scripts/`: shell scripts to build the dataset (`smd_*`) and train (`sml_*`)

# Building dataset

1. `$ git clone git@github.com:chrisdonahue/ddc.git`
1. `cd ddc`
1. `$ sudo pip install -e .` (installs as editable library)
1. `$ export SM_DATA_DIR=~/ddc/data` (or another directory of your choosing)
1. `$ mkdir $SM_DATA_DIR`
1. `$ cd $SM_DATA_DIR`
1. Download game data
    * [(Fraxtil) Tsunamix III](https://fra.xtil.net/simfiles/data/tsunamix/III/Tsunamix%20III%20[SM5].zip)
    * [(Fraxtil) Fraxtil's Arrow Arrangements](https://fra.xtil.net/simfiles/data/arrowarrangements/Fraxtil's%20Arrow%20Arrangements%20[SM5].zip)
    * [(Fraxtil) Fraxtil's Beast Beats](https://fra.xtil.net/simfiles/data/beastbeats/Fraxtil's%20Beast%20Beats%20[SM5].zip)
    * [(ITG) In The Groove](http://stepmaniaonline.net/downloads/packs/In%20The%20Groove%201.zip)
    * [(ITG) In The Groove 2](http://stepmaniaonline.net/downloads/packs/In%20The%20Groove%202.zip)
1. `cd ~/ddc/scripts`
1. `./smd.sh` (extracts dataset)
>>>>>>> origin/master_v2
