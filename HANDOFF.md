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
- Updated versioning/documentation files to `0.2.2`.

## Key Findings

- Beginner placement was not part of the final 8-run practical export plan.
- `dance-double_Beginner` has effectively no usable dataset size in the observed official-pack corpus.
- The DDC symbolic training path does not automatically collapse chart content to tap-only tokens.
- The difficulty evaluator *does* currently reduce charts to tap notes only, so shock arrows/mines/holds/rolls/lifts/fakes are not fully represented there.
- `.ssc`-only information is still outside the current extraction path.

## Important Repository Notes

- Large generated artifacts should not be pushed casually.
- `output_v132/` is heavyweight local training output and should remain local.
- local model exports are also large and should use a deliberate artifact/publication strategy.

## Recommended Next Steps

1. Audit actual note-symbol coverage in the official DDR corpus.
2. Add `.ssc` ingestion.
3. Add `dance-single_Beginner` placement training as an optional extension.
4. Extend the difficulty evaluator to include non-tap object semantics.
5. Create a clean ArrowVortex deploy/export path for final checkpoints.
6. Push code/documentation/submodule changes cleanly without pushing massive generated artifacts.
