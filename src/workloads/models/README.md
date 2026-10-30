# Models

Each file loads one model the experiments measure.

---

## Image classification

Input `(B, 3, 224, 224)` images, normalized with the ImageNet mean and standard deviation; output `(B, 1000)` logits, one per ImageNet class.

### `resnet.py`

ResNet-18 with torchvision's pretrained ImageNet weights (`IMAGENET1K_V1`, 11.7M parameters, about 69.8% top-1 on ImageNet). A residual CNN: 3×3 convolutions, each followed by BatchNorm and ReLU, with a skip connection around every pair, then global average pooling and one Linear layer. Kernels Reef needs: Conv2D, ReLU, max pool, average pool, residual add, Linear. BatchNorm folds into the convolution before it.
