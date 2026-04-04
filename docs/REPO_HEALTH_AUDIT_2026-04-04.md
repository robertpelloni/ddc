# Repository Health Audit

Date: 2026-04-04

## Scope

- Repository root: `.`

## Summary

- Files containing unresolved merge-conflict markers: **9**
- Files containing TensorFlow references: **48**
- Files containing legacy model-path / legacy-training references: **11**

## Unresolved Merge-Conflict Markers

- `autochart.py`
- `ddc_stepmania\learn\beatcalc.py`
- `ddc_stepmania\learn\chart.py`
- `learn\beatcalc.py`
- `learn\data_gen.py`
- `learn\models_v2.py`
- `scripts\smd_1_extract.sh`
- `scripts\smd_4_analyze.sh`
- `scripts\train_v2.py`

## TensorFlow Reference Hotspots

### `CHANGELOG.md`

- 12: - Added `docs/REPO_HEALTH_AUDIT_2026-04-04.md` documenting unresolved merge conflicts, TensorFlow hotspots, and legacy `.h5` / `train_v2.py` / `models_v2` references.
- 117: - **Modernization**: Ported entire codebase from Python 2.7 / TensorFlow 0.12 to Python 3.8+ / TensorFlow 2.x.

### `DASHBOARD.md`

- 8: **Python Runtime Reality:** Current repository training work was adapted to PyTorch for the local environment, while legacy TensorFlow-oriented code paths still exist in the codebase.
- 91: **Description:** Audit of remaining unresolved merge conflicts and legacy TensorFlow / `.h5` reference hotspots.

### `LLM_INSTRUCTIONS.md`

- 22: - Be aware that the repository currently contains both legacy TensorFlow-oriented paths and newer PyTorch-oriented work for the active environment.

### `PROJECT_ANALYSIS.md`

- 13: *   **Code Modernization**: The codebase has been migrated from Python 2/TensorFlow 1.x to Python 3/TensorFlow 2.x (using the `compat.v1` module).

### `ROADMAP.md`

- 4: - [x] **Modernization**: Ported to Python 3 and TensorFlow 2.x.

### `build\lib\infer\autochart_lib.py`

- 8: import tensorflow as tf

### `build\lib\learn\data_gen.py`

- 5: import tensorflow as tf

### `build\lib\learn\models_v2.py`

- 1: import tensorflow as tf
- 2: from tensorflow.keras import layers, models, Input

### `build\lib\learn\onset_extract.py`

- 5: import tensorflow as tf

### `build\lib\learn\onset_net.py`

- 3: import tensorflow as tf

### `build\lib\learn\onset_train.py`

- 7: import tensorflow as tf

### `build\lib\learn\sym_net.py`

- 4: import tensorflow as tf
- 12: # https://github.com/sherjilozair/char-rnn-tensorflow/blob/master/model.py

### `build\lib\learn\sym_train.py`

- 7: import tensorflow as tf

### `build\lib\scripts\train_v2.py`

- 5: import tensorflow as tf

### `ddc.egg-info\requires.txt`

- 1: tensorflow>=2.0

### `ddc_onset\README.md`

- 39: This model was [originally implemented and trained in TensorFlow](https://github.com/chrisdonahue/ddc/blob/master/learn/onset_net.py). I ported that ugly implementation to a clean PyTorch one a few years back as part of the process of building [Beat Sage](https://beatsage.com).

### `ddc_stepmania\infer\ddc_server.py`

- 12: import tensorflow as tf

### `ddc_stepmania\infer\onset_net.py`

- 3: import tensorflow as tf

### `ddc_stepmania\infer\sym_net.py`

- 4: import tensorflow as tf
- 12: # https://github.com/sherjilozair/char-rnn-tensorflow/blob/master/model.py

### `ddc_stepmania\learn\models\placement.py`

- 1: import tensorflow as tf

### `ddc_stepmania\learn\models\placement_train.py`

- 1: import tensorflow as tf
- 34: # https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/training/input.py

### `ddc_stepmania\learn\onset_extract.py`

- 5: import tensorflow as tf

### `ddc_stepmania\learn\onset_net.py`

- 4: import tensorflow.compat.v1 as tf

### `ddc_stepmania\learn\onset_train.py`

- 9: import tensorflow.compat.v1 as tf

### `ddc_stepmania\learn\sym_net.py`

- 4: import tensorflow as tf
- 12: # https://github.com/sherjilozair/char-rnn-tensorflow/blob/master/model.py

### `ddc_stepmania\learn\sym_train.py`

- 7: import tensorflow as tf

### `ddc_stepmania\learn\tf_feats.py`

- 3: import tensorflow as tf

### `ddc_stepmania\learn\util.py`

- 11: import tensorflow.compat.v1 as tf

### `docs\DASHBOARD.md`

- 27: - **`docs/REPO_HEALTH_AUDIT_2026-04-04.md`**: Remaining repository blocker/hotspot audit for merge conflicts and legacy TensorFlow references.
- 33: - **Environment Notes**: Local training work was adapted to PyTorch for the active environment; legacy TensorFlow-oriented paths remain in the repository.

### `docs\REPO_HEALTH_AUDIT_2026-04-04.md`

- 12: - Files containing TensorFlow references: **47**
- 33: ## TensorFlow Reference Hotspots
- 37: - 108: - **Modernization**: Ported entire codebase from Python 2.7 / TensorFlow 0.12 to Python 3.8+ / TensorFlow 2.x.
- 41: - 8: **Python Runtime Reality:** Current repository training work was adapted to PyTorch for the local environment, while legacy TensorFlow-oriented code paths still exist in the codebase.
- 45: - 21: *   Prioritize efficient and modern solutions (e.g., TensorFlow 2.x, Librosa).
- 46: - 60: *   Use `tensorflow` 2.x (Keras API).
- 47: - 65: *   Ensure `import tensorflow` and `import librosa` succeed.
- 51: - 13: *   **Code Modernization**: The codebase has been migrated from Python 2/TensorFlow 1.x to Python 3/TensorFlow 2.x (using the `compat.v1` module).
- 55: - 4: - [x] **Modernization**: Ported to Python 3 and TensorFlow 2.x.
- 59: - 8: import tensorflow as tf
- 63: - 5: import tensorflow as tf
- 67: - 1: import tensorflow as tf
- 68: - 2: from tensorflow.keras import layers, models, Input
- 72: - 5: import tensorflow as tf
- 76: - 3: import tensorflow as tf
- 80: - 7: import tensorflow as tf
- 84: - 4: import tensorflow as tf
- 85: - 12: # https://github.com/sherjilozair/char-rnn-tensorflow/blob/master/model.py
- 89: - 7: import tensorflow as tf
- 93: - 5: import tensorflow as tf
- ... 61 more

### `infer-tf1\ddc_server.py`

- 8: import tensorflow as tf

### `infer-tf1\ddc_server_old.py`

- 8: import tensorflow as tf

### `infer-tf1\onset_net.py`

- 3: import tensorflow as tf

### `infer-tf1\sym_net.py`

- 4: import tensorflow as tf
- 12: # https://github.com/sherjilozair/char-rnn-tensorflow/blob/master/model.py

### `infer-tf1\tf_upgrade.py`

- 1: # Copyright 2016 The TensorFlow Authors. All Rights Reserved.
- 15: """Upgrader for Python scripts from pre-1.0 TensorFlow to 1.0 TensorFlow."""
- 302: class TensorFlowCallVisitor(ast.NodeVisitor):
- 303: """AST Visitor that finds TensorFlow Function calls.
- 502: class TensorFlowCodeUpgrader(object):
- 503: """Class that handles upgrading a set of Python files to TensorFlow 1.0."""
- 557: visitor = TensorFlowCallVisitor(in_filename, lines)
- 626: description="""Convert a TensorFlow Python file to 1.0
- 660: upgrade = TensorFlowCodeUpgrader()
- 675: print("TensorFlow 1.0 Upgrade Script")

### `learn\data_gen.py`

- 8: import tensorflow as tf
- 15: """Dummy class for when TensorFlow is not available."""
- 28: import tensorflow as tf

### `learn\models_v2.py`

- 3: import tensorflow as tf
- 8: from tensorflow.keras import layers, models, Input
- 20: import tensorflow as tf
- 21: from tensorflow.keras import layers, models, Input

### `learn\onset_extract.py`

- 5: import tensorflow as tf

### `learn\onset_net.py`

- 3: import tensorflow as tf

### `learn\onset_train.py`

- 7: import tensorflow as tf

### `learn\sym_net.py`

- 4: import tensorflow as tf
- 12: # https://github.com/sherjilozair/char-rnn-tensorflow/blob/master/model.py

### `learn\sym_train.py`

- 7: import tensorflow as tf

### `learn\train_v2.py`

- 7: import tensorflow as tf

### `requirements.txt`

- 1: tensorflow>=2.0

### `scripts\audit_repo_health.py`

- 27: if 'tensorflow' in lower or 'model.h5' in lower or 'train_v2.py' in lower or 'models_v2' in lower:
- 29: if 'tensorflow' in lower:
- 50: f'- Files containing TensorFlow references: **{len(tf_refs)}**',
- 63: lines += ['', '## TensorFlow Reference Hotspots', '']
- 87: '- Legacy TensorFlow-oriented code paths and `.h5` references remain in parallel with the newer PyTorch-oriented work.',
- 93: '2. Decide which TensorFlow-era paths remain intentionally supported versus deprecated.',

### `scripts\train_v2.py`

- 5: import tensorflow as tf

### `setup.py`

- 16: "tensorflow>=2.0",

### `tests\test_utils.py`

- 4: import tensorflow as tf
- 5: from tensorflow.keras import layers, models


## Legacy Model / Training Reference Hotspots

### `CHANGELOG.md`

- 12: - Added `docs/REPO_HEALTH_AUDIT_2026-04-04.md` documenting unresolved merge conflicts, TensorFlow hotspots, and legacy `.h5` / `train_v2.py` / `models_v2` references.
- 130: - **Legacy Code**: Removed `infer/onset_net.py` and `infer/sym_net.py` (logic moved to `ddc_onset` submodule and `learn/models_v2.py`).

### `build\lib\infer\autochart_lib.py`

- 32: from learn.models_v2 import create_sym_model
- 128: onset_model_path = os.path.join(self.models_dir, 'onset', 'model.h5')
- 149: model_path = os.path.join(self.models_dir, model_name, 'model.h5')
- 202: from learn.models_v2 import SymNetV2

### `build\lib\scripts\train_all.py`

- 34: run_cmd(f"python3 scripts/train_v2.py --dataset_dir {onset_data_dir} --feats_dir {feats_dir} --out_dir {os.path.join(models_dir, 'onset')} --model_type onset --epochs 5")
- 51: run_cmd(f"python3 scripts/train_v2.py --dataset_dir {bucket_dir} --feats_dir {feats_dir} --out_dir {out_dir} --model_type sym --epochs 10")

### `build\lib\scripts\train_v2.py`

- 6: from learn.models_v2 import create_onset_model, create_sym_model

### `ddc.egg-info\SOURCES.txt`

- 34: learn/models_v2.py
- 45: scripts/train_v2.py

### `docs\REPO_HEALTH_AUDIT_2026-04-04.md`

- 27: - `learn\models_v2.py`
- 30: - `scripts\train_v2.py`
- 65: ### `build\lib\learn\models_v2.py`
- 91: ### `build\lib\scripts\train_v2.py`
- 194: ### `learn\models_v2.py`
- 222: ### `learn\train_v2.py`
- 232: - 27: if 'tensorflow' in lower or 'model.h5' in lower or 'train_v2.py' in lower or 'models_v2' in lower:
- 239: ### `scripts\train_v2.py`
- 257: - 121: - **Legacy Code**: Removed `infer/onset_net.py` and `infer/sym_net.py` (logic moved to `ddc_onset` submodule and `learn/models_v2.py`).
- 261: - 32: from learn.models_v2 import create_sym_model
- 262: - 128: onset_model_path = os.path.join(self.models_dir, 'onset', 'model.h5')
- 263: - 149: model_path = os.path.join(self.models_dir, model_name, 'model.h5')
- 264: - 202: from learn.models_v2 import SymNetV2
- 268: - 34: run_cmd(f"python3 scripts/train_v2.py --dataset_dir {onset_data_dir} --feats_dir {feats_dir} --out_dir {os.path.join(models_dir, 'onset')} --model_type onset --epochs 5")
- 269: - 51: run_cmd(f"python3 scripts/train_v2.py --dataset_dir {bucket_dir} --feats_dir {feats_dir} --out_dir {out_dir} --model_type sym --epochs 10")
- 271: ### `build\lib\scripts\train_v2.py`
- 273: - 6: from learn.models_v2 import create_onset_model, create_sym_model
- 277: - 34: learn/models_v2.py
- 278: - 45: scripts/train_v2.py
- 282: - 136: onset_model_path = os.path.join(self.models_dir, "onset", "model.h5")
- ... 9 more

### `infer\autochart_lib.py`

- 136: onset_model_path = os.path.join(self.models_dir, "onset", "model.h5")

### `learn\train_v2.py`

- 11: from learn.models_v2 import create_onset_model, create_sym_model

### `scripts\audit_repo_health.py`

- 27: if 'tensorflow' in lower or 'model.h5' in lower or 'train_v2.py' in lower or 'models_v2' in lower:
- 31: if 'model.h5' in lower or 'train_v2.py' in lower or 'models_v2' in lower:
- 94: '3. Replace or clearly quarantine stale `.h5`/`models_v2`/`train_v2.py` references where the PyTorch path is now canonical for the active environment.',

### `scripts\train_v2.py`

- 6: from learn.models_v2 import create_onset_model, create_sym_model

### `tests\test_integration.py`

- 38: model_path = os.path.join(model_dir, 'model.h5')

## Interpretation

- The repository has advanced substantially, but it still contains unresolved merge-conflict markers in several files.
- Legacy TensorFlow-oriented code paths and `.h5` references remain in parallel with the newer PyTorch-oriented work.
- This does not block documentation work, corpus auditing, or targeted pipeline modernization, but it does mean a future cleanup pass is still warranted before calling the codebase fully normalized.

## Recommended Next Cleanup Actions

1. Resolve remaining merge-conflict markers in root metadata/docs and training-related modules.
2. Decide which TensorFlow-era paths remain intentionally supported versus deprecated.
3. Replace or clearly quarantine stale `.h5`/`models_v2`/`train_v2.py` references where the PyTorch path is now canonical for the active environment.
4. After cleanup, rerun the repo-health audit to verify the blocker count drops toward zero.
