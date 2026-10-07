# PCam Stage 2 machine-learning project

This repository implements the Stage 2 comparison for the same ML problem used in Stage 1: binary classification of 96 x 96 RGB PatchCamelyon (PCam) histopathology patches as tumour-positive or tumour-negative.

## Methods

The primary comparison uses two distinct method families from the course:

- **CNN**: three convolution/max-pooling blocks, global average pooling, a 64-unit ReLU layer, dropout, and one sigmoid output.
- **MLP**: the same RGB pixels are rescaled, flattened, passed through a 64-unit ReLU hidden layer with dropout, and mapped to one sigmoid output.

The MLP deliberately has no convolution or ImageNet pretraining. This makes the comparison about the value of the CNN's image-specific spatial inductive bias, instead of comparing two CNN variants.

Both methods use the same PCam revision, preprocessing, optimizer, learning rate, batch size, epoch budget, binary cross-entropy loss, and validation protocol.

## Leakage-safe experiment protocol

1. Train both methods only on the official `train` split.
2. Use the official `valid` split to select the best epoch for each method.
3. Compare the two methods using validation data only.
4. Select the final method by the lower validation binary cross-entropy. Use validation 0/1 error only as a tie-breaker.
5. Load the official `test` split only after the final method is selected.
6. Report final test binary cross-entropy and 0/1 error (`1 - accuracy`). Precision, recall, F1, specificity, and confusion counts are supporting metrics.

`run_stage2.py` enforces this order.

## Environment

Standard TensorFlow environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

WSL/Linux with a supported NVIDIA GPU:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-wsl-gpu.txt
```

## Verify the exact dataset

The repository pins the Hugging Face dataset revision in `pcam_config.py`. Before writing the report, measure the split sizes and class counts from that exact revision:

```bash
python check_data.py --output artifacts/dataset_summary.json
```

Use the measured counts in the report. Do not copy an assumed exact class balance if the local data does not match it.

## Run the complete Stage 2 experiment

```bash
python run_stage2.py
```

This creates one timestamped experiment under `artifacts/` with:

- `runs/cnn/.../history.csv` and `validation_summary.json`;
- `runs/mlp/.../history.csv` and `validation_summary.json`;
- `comparison/validation_comparison.csv`;
- `comparison/selection.json`;
- `final_test/final_test_metrics.json` and `.csv`;
- `final_test/test_predictions.npz` so test figures can be regenerated without rerunning inference;
- `figures/` with report-ready plots and a `manifest.json`;
- one saved `best.weights.h5` per model run.

Only the final validation-selected method is evaluated on the test split. Test-based figures are created only after that selection and evaluation have completed.

## Run the steps manually

Train the two methods:

```bash
python train.py cnn
python train.py mlp
```

Then compare the completed run directories:

```bash
python compare_models.py \
  --cnn-run artifacts/cnn/<cnn-run> \
  --mlp-run artifacts/mlp/<mlp-run> \
  --output-dir artifacts/comparison
```

Finally evaluate only the selected method:

```bash
python evaluate_final.py \
  --selection artifacts/comparison/selection.json \
  --output-dir artifacts/final_test
```

## Report visualizations

A complete `run_stage2.py` run generates five figures automatically under `artifacts/<experiment>/figures/`:

- `learning_curves.png`: CNN and MLP training/validation BCE and 0/1 error across epochs;
- `validation_comparison.png`: training-versus-validation metrics at each method's selected epoch;
- `final_test_diagnostics.png`: confusion matrix, ROC curve, and precision-recall curve for the final method;
- `dataset_examples.png`: training patches from both classes with the center 32 x 32 label region marked;
- `final_test_errors.png`: the most confident false positives and false negatives.

For the four-page report, the first three are the most useful. Put the dataset and error-example figures in the appendix unless space remains. See `docs/visualizations.md` for the rationale and report placement.

Regenerate all figures for a completed experiment without retraining:

```bash
python plot_report_figures.py artifacts/stage2-YYYYMMDD-HHMMSS
```

To generate only the training-example figure, without loading the test split:

```bash
python plot_dataset_examples.py --output dataset_examples.png
```

The older one-run helper remains available:

```bash
python plot_learning_curves.py <run-directory>
```

## Code organization

- `pcam_config.py`: shared experiment constants and pinned dataset revision.
- `pcam_data.py`: split loading. Training code can load only train and validation; final evaluation loads test.
- `pcam_models.py`: CNN and MLP hypothesis spaces.
- `train.py`: common training and validation procedure.
- `compare_models.py`: validation-only method comparison and final selection.
- `evaluate_final.py`: final test evaluation after selection.
- `metrics.py`: BCE, 0/1 error, and classification metrics.
- `check_data.py`: exact split and label-count verification.
- `visualizations.py`: report-ready dataset, learning-curve, comparison, and final-test figures.
- `plot_report_figures.py`: regenerate all report figures for a completed experiment.
- `plot_dataset_examples.py`: training-only PCam example figure.
- `plot_learning_curves.py`: legacy one-run learning-curve helper.
- `docs/visualizations.md`: visualization rationale, leakage rules, and report placement.
- `docs/stage2_report_mapping.md`: mapping from generated outputs to the Stage 2 report sections.

Large checkpoints and generated `artifacts/` are ignored by Git.

## Existing CNN reference result

`results/reference_cnn_2026-10-07/` preserves the numeric result from the previous repository state for provenance. The old CNN selected epoch 7 with validation BCE 0.357594 and validation accuracy 86.09%. Its earlier test accuracy was 82.99%, which corresponds to a 17.01% 0/1 error.

That historical test result must **not** be used to choose CNN versus MLP. The final comparison must come from a fresh common-protocol run of both methods.

## Current package status

The source code and validation/test protocol are complete. No MLP result is invented in this package. Full CNN and MLP training must be run in a TensorFlow environment before the Stage 2 report is finalized. The current execution environment used to prepare this source package did not contain TensorFlow or the PCam dataset, so only code-level tests were run here.
