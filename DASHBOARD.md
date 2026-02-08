# Project Dashboard

This dashboard provides an overview of the integrated components, submodules, and current versioning of the Dance Dance Convolution project.

## Project Status
**Version:** 0.2.1
**Build Status:** Passing (Manual Verification)
**Python Version:** 3.8+

## Integrated Submodules

The project integrates two key external repositories as git submodules to enhance functionality.

| Submodule | Path | Description | Version/Commit |
| :--- | :--- | :--- | :--- |
| **DDC Onset** | `ddc_onset/` | Provides deep learning models for precise onset (beat) detection. Used for aligning steps to audio. | `main` branch |
| **FFR Difficulty Model** | `ffr-difficulty-model/` | A neural network model for estimating the difficulty of stepcharts (scale 0-20+). | `main` branch |

## Project Structure

### Root Directory
*   `autochart.py`: **Main CLI Tool**. Run this to generate charts.
*   `setup.py`: Packaging script.
*   `VERSION`: Single source of truth for project version.
*   `requirements.txt`: Python dependencies.
*   `LLM_INSTRUCTIONS.md`: Guidelines for AI contributors.
*   `HANDOFF.md`: Context for handovers.

### Components

#### 1. Training Pipeline
**Location:** `scripts/train_all.py`
**Description:** Automates the retraining of DDC step placement models.
*   **Data Prep:** Buckets songs by difficulty (Beginner -> Challenge).
*   **Feature Extraction:** Uses `librosa` to generate mel-spectrograms.
*   **Training:** Retrains `OnsetNet` (if needed) and `SymNet` (step placement) using TensorFlow 2.x.

#### 2. Inference Engine (AutoChart)
**Location:** `infer/autochart_lib.py`
**Description:** The core tool for generating stepfiles from arbitrary audio.
*   **Input:** MP3/WAV/OGG audio files.
*   **Processing:**
    *   Beat Detection (`ddc_onset`)
    *   Step Generation (DDC `SymNet` Model)
    *   Difficulty Rating (`ffr-difficulty-model`)
    *   Metadata fetching (Google Custom Search API for images)
*   **Output:** Complete `.sm` file + Audio + Album Art in a structured folder.

#### 3. Server
**Location:** `infer/ddc_server.py`
**Description:** A Flask-based server that exposes the AutoChart functionality via an API, suitable for integration into ArrowVortex or other tools.

#### 4. Submodules
*   `ddc_onset/`: Contains separate `setup.py` and logic for onset detection.
*   `ffr-difficulty-model/`: Contains model definitions for difficulty scoring.

## Usage Quickstart

**Generate a chart for a song:**
```bash
python autochart.py --audio path/to/song.mp3 --output output_dir/
```

**Retrain models:**
```bash
python scripts/train_all.py --data_dir path/to/sm_files/ --output_dir data_out/
```
