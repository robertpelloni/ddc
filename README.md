# Dance Dance Convolution

Modernized pipeline for DDR chart generation.

## Features
- Onset Detection
- Symbol Generation (Steps) using LSTM
- Difficulty Rating (FFR Model)
- Automatic Metadata & Image Fetching

## Usage
```bash
python autochart.py song.mp3 --models_dir models/ --ffr_dir ffr-difficulty-model/
```
