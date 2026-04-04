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
- Updated versioning/documentation files to `0.2.6`.

## Key Findings

- Beginner placement was not part of the final 8-run practical export plan.
- `dance-double_Beginner` has effectively no usable dataset size in the observed official-pack corpus.
- The DDC symbolic training path does not automatically collapse chart content to tap-only tokens.
- The difficulty evaluator *does* currently reduce charts to tap notes only, so shock arrows/mines/holds/rolls/lifts/fakes are not fully represented there.
- `.ssc` support now exists in extraction, but the full downstream filtering/training refresh against the expanded corpus is still pending.
- Corpus audit confirmed 20 `.ssc` files exist in the raw official DDR corpus and are now recoverable through the refreshed extractor.
- Corpus audit confirmed extracted note vocabulary contains substantial non-binary symbols: `2`, `3`, and `M`.

## Important Repository Notes

- Large generated artifacts should not be pushed casually.
- `output_v132/` is heavyweight local training output and should remain local.
- local model exports are also large and should use a deliberate artifact/publication strategy.

## Recommended Next Steps

1. Rerun full downstream filtering/feature extraction/training against the expanded `.ssc`-inclusive corpus.
2. Retrain the difficulty evaluator against the refreshed `.ssc`-inclusive loader path.
3. Audit actual note-object semantics for symbols such as `2`, `3`, and `M`.
4. Add `dance-single_Beginner` placement training as an optional extension.
5. Extend the difficulty evaluator to include non-tap object semantics.
6. Create a clean ArrowVortex deploy/export path for final checkpoints.
