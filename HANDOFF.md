# DDC Training / Integration Handoff

Date: 2026-04-04

## Status

Core training/integration work has been advanced substantially.

### Completed in this pass

- Confirmed official DDR pack download sources already configured in the repo and verified completed download markers.
- Prepared filtered JSON buckets for single/double DDR chart training.
- Switched the top-level training orchestration to use the PyTorch trainer in the current environment.
- Trained the practical 8-bucket placement layout:
  - single Easy / Medium / Hard / Challenge
  - double Easy / Medium / Hard / Challenge
- Trained an onset model from `dance-single_Hard`.
- Rebuilt the difficulty-evaluator dataset and trained both:
  - `dance-single`
  - `dance-double`
- Fixed the submodule difficulty-training script so global NaN cleanup does not silently eliminate one mode.
- Added detailed documentation of findings in `docs/TRAINING_ANALYSIS_2026-04-04.md`.
- Added corpus audit tooling/report in `scripts/audit_corpus.py` and `docs/CORPUS_AUDIT_2026-04-04.md`.
- Extended `dataset/extract_json.py` to support `.ssc` files via `simfile`.
- Validated that refreshed local extraction increases coverage from 1234 to 1254 songs and from 9241 to 9403 charts.
- Documented the exact `.ssc`-driven corpus delta in `docs/SSC_EXPANSION_ANALYSIS_2026-04-04.md`.
- Updated the FFR difficulty-data loader so it also prefers `.ssc` over `.sm` where available.
- Validated the refreshed FFR preprocessing path at roughly 1255 simfiles / 9407 charts.
- Added note-object semantic audit tooling/report in `scripts/audit_note_objects.py` and `docs/NOTE_OBJECT_SEMANTICS_2026-04-04.md`.
- Added `docs/RETRAINING_REFRESH_PLAN_2026-04-04.md` describing the exact next-phase retraining workflow.
- Added `scripts/compare_bucket_counts.py` and `docs/BUCKET_SPLIT_DELTA_2026-04-04.md` to quantify exact downstream split-file deltas after `.ssc`-inclusive preparation.
- Added `scripts/audit_repo_health.py` and `docs/REPO_HEALTH_AUDIT_2026-04-04.md` to quantify remaining normalization blockers in the repository.
- Resolved top-level conflict-marker files in `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `GPT.md`, `LLM_INSTRUCTIONS.md`, and `setup.py`.
- Resolved conflict-marker files in `autochart.py`, `learn/beatcalc.py`, `learn/data_gen.py`, `learn/models_v2.py`, `scripts/train_v2.py`, `scripts/smd_1_extract.sh`, and `scripts/smd_4_analyze.sh`.
- Refreshed the repo-health audit and reduced unresolved merge-conflict-marker files from 15 to 2.
- Added `docs/LEGACY_SUBTREE_QUARANTINE_2026-04-04.md` documenting the decision to treat the final `ddc_stepmania/` conflict-marker files as quarantined legacy-subtree content.
- Added `scripts/audit_refresh_readiness.py` and `docs/SSC_REFRESH_READINESS_2026-04-04.md` to record the exact state of the prepared `.ssc`-inclusive refresh work directory.
- Added resume-friendly skip flags to `scripts/train_all.py` so the refresh run can be restarted safely without repeating completed work.
- Launched the actual `.ssc`-inclusive refresh run and documented it in `docs/TRAINING_REFRESH_LAUNCH_2026-04-04.md`.
- Captured an in-flight runtime progress snapshot in `docs/TRAINING_REFRESH_PROGRESS_2026-04-04.md`, including initial onset checkpoint production.
- Captured a later progress snapshot in `docs/TRAINING_REFRESH_PROGRESS_2_2026-04-04.md`, showing onset checkpoint-set completion and transition into the first practical SymNet bucket stage.
- Replaced the conflicted root `README.md` with a clean current-state overview.
- Updated versioning/documentation files to `0.2.17`.

## Key Findings

- Beginner placement was not part of the final 8-run practical export plan.
- `dance-double_Beginner` has effectively no usable dataset size in the observed official-pack corpus.
- The DDC symbolic training path does not automatically collapse chart content to tap-only tokens.
- The difficulty evaluator *does* currently reduce charts to tap notes only, so shock arrows/mines/holds/rolls/lifts/fakes are not fully represented there.
- `.ssc` support now exists in extraction, but the full downstream filtering/training refresh against the expanded corpus is still pending.
- Corpus audit confirmed 20 `.ssc` files exist in the raw official DDR corpus and are now recoverable through the refreshed extractor.
- Corpus audit confirmed extracted note vocabulary contains substantial non-binary symbols: `2`, `3`, and `M`.
- Note-object semantic audit now documents the strongest supported interpretation of observed symbols:
  - `1` = tap
  - `2` = hold head
  - `3` = tail
  - `M` = mine
  - no observed `4`, `A`, `F`, `K`, or `L` in the refreshed official-pack extraction

## Important Repository Notes

- Large generated artifacts should not be pushed casually.
- `output_v132/` is heavyweight local training output and should remain local.
- local model exports are also large and should use a deliberate artifact/publication strategy.

## Recommended Next Steps

1. Continue monitoring the active refresh log at `data/ssc_refresh_training.log`.
2. Verify completed checkpoint output appears for `dance-single_Easy` and subsequent practical 8-bucket directories under `data/ssc_refresh_work/models/`.
3. Verify refreshed difficulty-model output eventually appears under `data/ssc_refresh_work/ffr_models/`.
4. After training completes, document final artifact inventory and export a clean deployment-ready model bundle.
5. Add `dance-single_Beginner` placement training as an optional extension if still desired.
6. Extend the difficulty evaluator to include non-tap object semantics.
7. Optionally normalize the quarantined `ddc_stepmania/` conflict-marker files in a dedicated legacy-maintenance pass later.
