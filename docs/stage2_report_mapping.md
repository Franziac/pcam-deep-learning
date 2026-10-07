# Stage 2 report mapping

This file maps repository outputs to the current report outline and peer-grading questions. It is a writing aid, not a finished report.

## Introduction

State the application domain: binary classification of metastatic tissue in H&E-stained lymph-node image patches. Add one short report overview sentence that identifies the Problem Formulation, Methods, Results, and Conclusion sections.

## Problem formulation

One data point is one 96 x 96 RGB PCam patch. The features are the 27,648 pixel-channel intensity values. The binary label is positive when the center 32 x 32 region contains at least one annotated tumor pixel.

Use `check_data.py` to report the measured split sizes and class counts from the pinned dataset revision.

## Methods

### Feature selection and preprocessing

No explicit feature-selection algorithm is applied. The original data point is a 96 x 96 RGB image.

The CNN keeps the original 96 x 96 resolution and rescales pixel values from [0, 255] to [0, 1].

The MLP rescales pixel values to [-1, 1], resizes each image to 48 x 48, and then flattens the image before the fully connected layers. The resizing reduces the dimensionality of the MLP input and keeps the fully connected model computationally practical.

The two methods therefore use the same underlying RGB image information and the same dataset splits, but they do not use identical preprocessing.

### Method 1: CNN

Explain that convolutional filters use local receptive fields and shared weights, so the model is suited to image texture and spatial patterns. State the three convolution/max-pooling blocks, global average pooling, 64-unit ReLU layer, dropout, and sigmoid output.

### Method 2: MLP

The second method is a multi-layer perceptron. The 48 x 48 RGB input is flattened and passed through fully connected layers with 128 and 32 ReLU units, followed by dropout and one sigmoid output.

The MLP can learn nonlinear relationships between pixel values, but it does not encode image locality or share parameters between different image locations. It therefore has a different hypothesis space from the CNN.

This makes it a useful baseline for the PCam task. If the CNN achieves lower validation error, this supports the idea that convolutional locality and weight sharing are useful for histopathology image classification rather than the improvement coming only from using a nonlinear neural network.

### Loss

Both methods optimize binary cross-entropy because the task has binary labels and sigmoid probabilities. For validation and final reporting, also report 0/1 error as `1 - accuracy`.

### Validation

State the official split sizes. Training fits parameters on `train`. The `valid` split selects the best epoch and the final method. Each method keeps the earliest epoch with the lowest validation BCE. The final method is chosen by lowest validation BCE, with validation 0/1 error used only as a tie-breaker. The `test` split is loaded only after the final method is selected.

## Results

Use `artifacts/.../comparison/validation_comparison.csv` as the main comparison table. It contains, for both methods:

- training BCE;
- validation BCE;
- training 0/1 error;
- validation 0/1 error;
- accuracy, precision, and recall; and
- selected epoch and parameter count.

Discuss the train-validation gap, not only the winner. A larger gap is evidence of stronger overfitting. Choose the final method from validation results only.

Use `artifacts/.../figures/learning_curves.png` to show how train/validation BCE and 0/1 error evolve for both methods. Use `validation_comparison.png` to make the validation-only final-method decision visible. Keep the exact numeric values in a compact table as well.

Use `artifacts/.../final_test/final_test_metrics.json` for the final method. Report at minimum the test BCE and the mean 0/1 test error. Accuracy, precision, recall, F1, specificity, and confusion counts can support the interpretation. `figures/final_test_diagnostics.png` combines the confusion matrix with ROC and precision-recall curves. These curves are supporting diagnostics; they do not replace the required test error.

Because the main report is limited to four pages, put `dataset_examples.png` and `final_test_errors.png` in the appendix unless they are needed to support a specific discussion. See `docs/visualizations.md` for the full figure plan.

## Conclusion

Summarize which model generalized better and whether the performance is satisfactory. Discuss overfitting and remaining error. Plausible future work includes controlled image augmentation, stain normalization, stronger regularization, or a more specialized CNN. Keep these as future work unless they were actually evaluated.

## AI use

State exactly which project tasks used generative AI. Do not claim that AI performed experiments that were not run or verified by the project authors.
