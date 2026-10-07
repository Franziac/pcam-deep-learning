# Model architectures

## CNN

```text
96 x 96 x 3 RGB
  -> Rescale from [0, 255] to [0, 1]
  -> Conv2D(32, 3 x 3, ReLU) -> MaxPool(2 x 2)
  -> Conv2D(64, 3 x 3, ReLU) -> MaxPool(2 x 2)
  -> Conv2D(128, 3 x 3, ReLU) -> MaxPool(2 x 2)
  -> GlobalAveragePooling2D
  -> Dense(64, ReLU)
  -> Dropout(0.35)
  -> Dense(1, sigmoid)
```

The CNN keeps the original 96 x 96 image resolution. Its convolutional layers use local receptive fields and shared weights, which preserve and exploit spatial image structure.

## MLP

```text
96 x 96 x 3 RGB
  -> Rescale from [0, 255] to [-1, 1]
  -> Resize to 48 x 48
  -> Flatten
  -> Dense(128, ReLU)
  -> Dense(32, ReLU)
  -> Dropout(0.10)
  -> Dense(1, sigmoid)
```

The MLP is a nonlinear fully connected neural network. It uses the RGB pixel values as input but does not use convolution or pooling. The image is first resized to 48 x 48 to reduce the size of the flattened input and keep the number of fully connected parameters manageable.

After flattening, the model no longer has an explicit representation of two-dimensional image locality. A pixel at one location has a separate connection from a pixel at another location. The model also does not share weights between image locations.

This makes the MLP a useful comparison method for the CNN. Both methods can learn nonlinear decision functions, but only the CNN has an image-specific inductive bias based on local receptive fields and shared convolutional filters.

The MLP does not use ImageNet pretraining or another pretrained feature extractor.
