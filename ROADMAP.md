# Roadmap

## Implemented Features
- [x] **Modernization**: Ported to Python 3 and TensorFlow 2.x.
- [x] **Submodule Integration**: `ddc_onset` and `ffr-difficulty-model`.
- [x] **Inference CLI**: `autochart.py` for end-to-end generation.
- [x] **Training Pipeline**: `scripts/train_all.py` for retraining.
- [x] **Metadata Handling**: Automatic album art fetch and `.sm` file tagging.

## Pending / Future Features
- [ ] **Double Chart Support**: Train a new `SymNet` model specifically for 8-panel (Double) charts. The current logic supports the *structure* of Double charts, but needs a trained weight file.
- [ ] **Advanced Difficulty Models**: Integrate more granular difficulty models beyond FFR (e.g., specific stamina/tech metrics).
- [ ] **Real-time Inference**: Optimize the pipeline for near real-time generation suitable for streaming applications.
- [ ] **GUI**: Create a desktop GUI wrapper around `autochart.py`.
- [ ] **Integration Tests**: Set up a CI/CD pipeline with synthetic audio generation to test the full stack without copyright issues.
