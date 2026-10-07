# Results in this source package

`reference_cnn_2026-10-07/` preserves the earlier CNN numeric result from the original repository.

It is retained only for provenance. Do not compare this historical test result against a newly trained MLP to select the final method. Run `run_stage2.py` so the CNN and MLP are trained and selected under the same protocol. The final test split is then evaluated only for the validation-selected winner.

No MLP metrics are fabricated in this package. Large model checkpoints are intentionally excluded and ignored by Git.

A fresh `run_stage2.py` run also creates `figures/` with report-ready plots. These figures are generated from the new common-protocol run and should be preferred over any historical visualization.
