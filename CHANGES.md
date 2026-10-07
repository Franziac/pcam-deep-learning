# Stage 2 repository changes

The repository was simplified around the Stage 2 grading requirements.

- Replaced the ImageNet DenseNet comparison with a clearly separate **MLP** method.
- Kept the original small **CNN** architecture as Method 1.
- Added one shared training protocol for both methods.
- Pinned the PCam dataset revision.
- Training code loads only train and validation data.
- Validation comparison selects the final method by validation BCE only.
- Final evaluation loads the test split only after model selection.
- Final evaluation reports both binary cross-entropy and 0/1 error.
- Both methods use the same optimizer, learning rate, batch size, epoch budget, preprocessing, and threshold.
- Only one best checkpoint per run is retained instead of every epoch checkpoint.
- Removed the nightly/release-candidate dependency mix and pinned a stable TensorFlow version.
- Added dataset verification so the report can use measured class counts.
- Added tests for metric calculation and validation-only selection.
- Added report-mapping notes under `docs/`.
- Large historical checkpoint files are not included in the final repository package.

- Added learning-curve generation for report-ready BCE and 0/1-error plots.
- Added automatic report-ready visualizations: combined learning curves, validation comparison, final-test confusion/ROC/PR diagnostics, training examples with the 32 x 32 label region, and confident test errors.
- Added compressed final-test predictions so figures can be regenerated without rerunning inference.
- Added a visualization manifest that records which data split each figure uses.
- Added `docs/visualizations.md` with guidance on which figures belong in the four-page report versus the appendix.

