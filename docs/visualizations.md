# Visualization plan for the Stage 2 report

The report has a four-page main-text limit, so each figure should answer a specific grading question. The project generates five figures, but only three are recommended for the main report.

## Recommended figures in the main report

### 1. `learning_curves.png` — training and validation behavior

**Question answered:** Do the methods overfit, and how does generalization change with epoch?

The figure shows training and validation binary cross-entropy (BCE) and 0/1 error for both the CNN and MLP. A vertical line marks the validation-selected epoch. This is the most useful figure for discussing the training-validation gap and whether additional training improves or harms validation performance.

Data used: train and validation histories only. No test data is used.

### 2. `validation_comparison.png` — final method selection

**Question answered:** Which method is selected and why?

The figure compares training and validation BCE and 0/1 error at each method's selected epoch. It makes the validation-only selection decision visible and also exposes the generalization gap. The report should still include the exact values in a compact table because the grading criteria explicitly ask for errors/metrics.

Data used: validation-only model-selection outputs.

### 3. `final_test_diagnostics.png` — final model behavior

**Question answered:** How does the final selected method behave on unseen test data?

The three-panel figure contains:

- a confusion matrix with counts and row percentages;
- an ROC curve with area under the curve (AUC);
- a precision-recall curve with average precision (AP).

The report's required test result remains BCE and 0/1 error. ROC/PR are supporting diagnostics, not replacements for the required test error. ROC shows the sensitivity/specificity tradeoff over thresholds. Precision-recall shows the precision/recall tradeoff over thresholds.

Data used: the official test split, only after validation-only model selection has finished.

## Useful appendix figures

### `dataset_examples.png`

Shows four negative and four positive training patches. The white 32 x 32 box marks the only region that determines the PCam label. This is useful for explaining the data point and label definition without using test examples.

### `final_test_errors.png`

Shows the most confident false positives and false negatives for the final selected model. The same center-region box is drawn on each patch. This figure supports qualitative error analysis and the limitations/future-work discussion. Do not infer a pathology explanation from these images unless it is visually supported and you can justify it.

## Why these visualizations were selected

The Stage 2 rubric emphasizes three tasks: compare training and validation errors for all methods, choose a final method from validation performance, and report/analyze test performance for the final method. The selected figures map directly to those tasks.

Standard binary-classification diagnostics also support this design. Scikit-learn documents confusion matrices for thresholded prediction errors, ROC curves for true-positive versus false-positive rate across thresholds, and precision-recall curves for the precision/recall tradeoff across thresholds.

PCam labels depend only on whether the center 32 x 32 region contains tumor tissue. The dataset and error-example figures therefore draw the center box explicitly.

## Generation

A complete run now generates the figures automatically:

```bash
python run_stage2.py
```

For an already completed experiment:

```bash
python plot_report_figures.py artifacts/stage2-YYYYMMDD-HHMMSS
```

The generated `figures/manifest.json` states which split each figure uses and recommends which figures belong in the main report versus the appendix.

## Reference documentation used for the design

- Scikit-learn classification metrics overview: https://scikit-learn.org/stable/modules/model_evaluation.html
- Scikit-learn ROC curve: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
- Scikit-learn precision-recall curve: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
- PCam dataset card and label definition: https://huggingface.co/datasets/1aurent/PatchCamelyon
