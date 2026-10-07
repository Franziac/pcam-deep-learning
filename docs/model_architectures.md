# Model architectures

## CNN

```text
96x96x3 RGB
  -> Rescale 1/255
  -> Conv2D(32, 3x3, ReLU) -> MaxPool(2x2)
  -> Conv2D(64, 3x3, ReLU) -> MaxPool(2x2)
  -> Conv2D(128, 3x3, ReLU) -> MaxPool(2x2)
  -> GlobalAveragePooling2D
  -> Dense(64, ReLU)
  -> Dropout(0.35)
  -> Dense(1, sigmoid)
```

## MLP

```text
96x96x3 RGB
  -> Rescale 1/255
  -> Flatten
  -> Dense(64, ReLU)
  -> Dropout(0.35)
  -> Dense(1, sigmoid)
```

The MLP deliberately does not use convolution, pooling, or ImageNet pretraining. This keeps the second method distinct and avoids the transfer-learning confound in the previous DenseNet comparison.
