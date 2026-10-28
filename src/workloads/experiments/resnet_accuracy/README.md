# resnet_accuracy

How much top-1 accuracy does ResNet-18 keep when its weights (and optionally its activations) are rounded to int8 or int4? ResNet-18 is the demo's image classifier, and Reef's quantization choice depends on whether a CNN survives int4.

## Method

- **Model:** torchvision's pretrained ResNet-18 (`IMAGENET1K_V1`), with every BatchNorm folded into the convolution before it. Folding must leave fp32 accuracy unchanged; Reef stores the folded weights, so those are what get rounded.
- **Data:** Imagenette's validation split (3,925 images, 10 ImageNet classes), preprocessed exactly as the weights expect. Each Imagenette folder is scored against its ImageNet class index.
- **Precisions** (`../quantization.py`), each starting from the same folded fp32 weights:

  | precision | weights | Conv2d/Linear inputs |
  |---|---|---|
  | `fp32` | fp32 | fp32 |
  | `int8` | int8, one scale per output channel | fp32 |
  | `int4` | int4, one scale per 32 inputs of each output channel | fp32 |
  | `w8a8` | int8, as `int8` | int8 |
  | `w4a8` | int4, as `int4` | int8 |

- **Metric:** top-1 accuracy, the model's highest score among all 1000 ImageNet classes. With 3,925 images at about 67% accuracy, one standard error is about ±0.75 points, so differences under about 1.5 points are within noise.

## Caveat

The model picks among all 1000 ImageNet classes, and some are near-duplicates of Imagenette's: in fp32, 138 churches are called "monastery", 53 garbage trucks "moving van" and 43 cassette players "tape player". That is why fp32 scores 66.8% (close to ResNet-18's ~70% on full ImageNet); restricted to the 10 Imagenette classes it scores 97.6%. What matters here is the drop from fp32, since every precision is scored the same way.

## Running

From the repo root, after downloading Imagenette into `data/` (see the top-level README):

```sh
uv run python src/workloads/experiments/resnet_accuracy/main.py
uv run python src/workloads/experiments/resnet_accuracy/main.py --precision fp32 --precision int4
```

## Output

`results/resnet_accuracy/resnet18.csv`, one row per precision:

| column | meaning |
|---|---|
| `precision` | one of the precisions above |
| `top1` | fraction of validation images classified correctly |

## Findings

Post-training quantization (round the trained weights, no retraining), all 3,925 validation images:

| precision | top-1 | vs fp32 |
|---|---|---|
| `fp32` | 66.8% | — |
| `int8` | 67.3% | +0.5 |
| `w8a8` | 67.2% | +0.4 |
| `int4` | 50.8% | −16.1 |
| `w4a8` | 50.8% | −16.1 |

- Folding BatchNorm left fp32 unchanged (66.83% before and after).
- int8 costs nothing, weights and activations alike (+0.5 is within noise).
- int4 weights lose about 16 points with plain rounding, even with one scale per 32 weights.
- The loss comes from the weights, not the activations: making activations int8 changes nothing (`int4` vs `w4a8`, `int8` vs `w8a8`). Int4 activations were not tested.
- Next: keep the first conv and the final Linear in int8, find which layers lose the most when rounded to int4 on their own, try smaller groups, and try quantization-aware training (QAT: fine-tuning with the rounding in place so the weights adapt to it).
