# Project Analysis and Feature Roadmap

This document provides a comprehensive analysis of the `ddc-stepmania` project, its current state, and a detailed roadmap for future development.

## Current State of the Project

The project is currently in a foundational state. The core architecture is in place, the necessary data has been downloaded, and the codebase has been significantly refactored and modernized. However, the advanced features requested by the user are not yet implemented.

### Key Accomplishments:

*   **Data Acquisition**: All 12 official DDR data packs have been successfully downloaded and extracted into the `data/raw/` directory.
*   **Project Structure**: The project has been organized into a proper Python package, `ddc_stepmania`, with a clear separation of concerns between data processing, model training, and inference.
*   **Code Modernization**: The codebase has been migrated from Python 2/TensorFlow 1.x to Python 3/TensorFlow 2.x (using the `compat.v1` module).
*   **Dependency Resolution**: Numerous bugs related to file paths, data loading, and missing libraries (`pydub`, `ffmpeg`) have been resolved.

### Existing Components:

*   **`ddc_stepmania` package**:
    *   `dataset`: Scripts for processing raw `.sm` files into JSON and then pickled Python objects.
    *   `learn`: The core machine learning package, containing the `OnsetNet` and `SymNet` models, as well as the training scripts (`onset_train.py`, `sym_train.py`).
    *   `infer`: Scripts and a server for using the trained models to generate new stepcharts.
*   **`ffr-difficulty-model`**: A pre-existing, integrated model for calculating the difficulty of a given stepchart.
*   **`scripts/`**: A collection of shell scripts that orchestrate the data processing and training pipelines.

## Feature Roadmap

The following is a detailed roadmap for the implementation of the remaining features, in order of priority.

### 1. Advanced Music Analysis

This is the most critical set of features, as it forms the foundation for generating high-quality stepcharts.

*   **BPM and Tempo Change Detection**:
    *   **Goal**: Accurately determine the BPM of a song, including any variations or tempo changes.
    *   **Implementation**: Utilize the `librosa` library, specifically the `librosa.beat.beat_track` function, to perform robust beat tracking and tempo estimation.
    *   **Integration**: The detected BPM and beat information will be stored in the processed data files and used to inform the `Chart` and `BeatCalc` classes.

*   **First and Last Beat Identification**:
    *   **Goal**: Pinpoint the exact timestamps of the first and last significant beats in the audio.
    *   **Implementation**: This information will be derived directly from the output of the beat tracking algorithm.
    *   **Integration**: This will be used to define the time window within which arrows can be placed, preventing them from appearing in silent sections of the song.

*   **Automatic Music Sectioning**:
    *   **Goal**: Segment a song into meaningful sections (e.g., intro, verse, chorus, bridge, outro).
    *   **Implementation**: Use a structural segmentation algorithm, such as `librosa.segment.agglomerative`, to identify the major sections of the song.
    *   **Integration**: The section boundaries will be used to create more musically interesting and varied stepcharts, with different patterns and densities for each section.

### 2. Refined Arrow Placement Logic

This set of features will focus on improving the quality and playability of the generated stepcharts.

*   **Beat-Aligned Arrow Placement**:
    *   **Goal**: Ensure that all generated arrows are perfectly aligned with the beat grid of the song.
    *   **Implementation**: Modify the `OnsetNet` to predict onsets that are quantized to the beat grid. This will prevent the creation of off-beat or awkwardly timed arrows.
    *   **Integration**: The `gen_labels.py` script will be updated to generate beat-aligned labels, and the `OnsetNet` will be retrained on this new data.

*   **Pattern-Based Arrow Selection**:
    *   **Goal**: Generate arrow patterns that are idiomatic to DDR and avoid random or unplayable sequences.
    *   **Implementation**: The `SymNet` model will be enhanced to learn common DDR patterns (e.g., crossovers, jacks, streams). This could involve using a more sophisticated model architecture or incorporating a pattern library.
    *   **Integration**: The `sym_train.py` script will be updated to train the new `SymNet` model, and the inference scripts will be modified to use it.

*   **Playability Enforcement**:
    *   **Goal**: Implement a post-processing step to clean up the generated charts and enforce playability rules.
    *   **Implementation**: This will involve creating a set of rules to prevent things like double-steps (placing arrows on both feet at the same time), overly dense patterns, and other unplayable sequences.
    *   **Integration**: A new script will be added to the `infer` package to perform this post-processing step.

### 3. Difficulty Generation and Rating

This set of features will focus on the automatic generation of charts for all difficulty levels and the assignment of accurate difficulty ratings.

*   **Multi-Difficulty Generation**:
    *   **Goal**: Automatically generate charts for all standard DDR difficulties (Beginner, Basic, Difficult, Expert, Challenge).
    *   **Implementation**: This will likely involve training separate models for each difficulty level, or creating a single, conditional model that takes the desired difficulty as an input.
    *   **Integration**: The training and inference pipelines will be updated to support the generation of multiple difficulty levels.

*   **Difficulty Rating Integration**:
    *   **Goal**: Use the `ffr-difficulty-model` to analyze the generated charts and assign them an accurate difficulty rating.
    *   **Implementation**: The API of the `ffr-difficulty-model` will be used to analyze the generated `.sm` files.
    *   **Integration**: The inference pipeline will be updated to include a final step where the difficulty rating is calculated and added to the generated chart's metadata.

### 4. Project Finalization

*   **Packaging**: The entire project will be structured into a clean, distributable Python package, so that it can be easily installed and used by others.
*   **Documentation**: Comprehensive documentation will be created, including an updated `README.md`, docstrings for all key functions and classes, and a `CONTRIBUTING.md` file.
*   **Testing**: A suite of unit and integration tests will be created to ensure the correctness and robustness of the entire system.
