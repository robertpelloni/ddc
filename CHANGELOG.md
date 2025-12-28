# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-12-25

### Added
- **Modernized Pipeline**: Introduced `autochart.py` for streamlined chart generation using TensorFlow 2 and Librosa.
- **PowerShell Support**: Added `download_data.ps1` for Windows users to download DDR packs.
- **FFR Integration**: Integrated `ffr-difficulty-model` for automatic difficulty rating of generated charts.
- **Documentation**: Added `docs/DASHBOARD.md` and updated LLM instruction files.
- **Versioning**: Added `VERSION.md` for centralized version management.

### Changed
- **Architecture**: Migrated from legacy Python 2/TF1 code to Python 3/TF2.
- **Training**: Updated training scripts (`train_v2.py`) to support the new architecture.
- **Structure**: Reorganized project structure for better modularity.

### Removed
- Legacy client/server scripts (`ddc_client.py`, `ddc_server.sh`) were removed in previous updates.
