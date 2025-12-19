# Dance Dance Convolution

Modernized pipeline for DDR chart generation.

## Installation
1. Clone repository with submodules:
   ```bash
   git clone --recursive <repo_url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Features
- Onset Detection
- Symbol Generation (Steps) using LSTM
- Difficulty Rating (FFR Model)
- Automatic Metadata & Image Fetching

## Usage
```bash
python autochart.py song.mp3 --models_dir models/ --ffr_dir ffr-difficulty-model/
```

## Submodules
This repository uses `ffr-difficulty-model` as a submodule.
