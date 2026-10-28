# reef-workloads

Benchmark AI models and inference workloads for Reef. Each experiment asks how a model Reef might run behaves at the precisions Reef might run it at (fp32, int8, int4), starting with accuracy. Int8 and int4 are simulated: values are rounded onto the integer grid and the arithmetic runs in fp32, so the numbers measure what rounding does to the model, not integer speed.

```
src/workloads/
  models/       the models under test (pretrained CNNs, detectors, diffusion)
  experiments/  one folder per experiment, each with a runnable main.py,
                plus helpers the experiments share (quantization.py, imagenette.py)
tests/          mirrors src/workloads/
results/        experiment outputs, one folder per experiment
data/           downloaded datasets (gitignored)
```

## Install

You need [uv](https://docs.astral.sh/uv/getting-started/installation/). uv installs Python 3.12 itself.

```sh
git clone https://github.com/LonghornSilicon/reef-workloads.git
cd reef-workloads
uv sync
```

## Data

Experiments that measure image classification read [Imagenette](https://github.com/fastai/imagenette), a 10-class subset of ImageNet, from `data/`:

```sh
mkdir -p data && cd data
curl -LO https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
tar -xzf imagenette2-160.tgz
```

## Test

```sh
uv run pytest -m "not slow"   # fast, no network
uv run pytest                 # adds tests that download weights or read data/
```

## Experiments

Run experiments from the repo root. Each one takes `--help`.

| experiment | question | command | output |
|---|---|---|---|
| `resnet_accuracy` | How much top-1 accuracy does ResNet-18 keep at int8 and int4? | `uv run python src/workloads/experiments/resnet_accuracy/main.py` | `results/resnet_accuracy/resnet18.csv` |

Each experiment has its own README covering its method, findings and tests: [resnet_accuracy](src/workloads/experiments/resnet_accuracy/README.md).

## Contributing

- **Core code** (models) lives in `src/workloads/` and never imports from `experiments/`.
- **An experiment** is a folder `src/workloads/experiments/<name>/` with a `main.py` that:
  - runs on its own as `uv run python src/workloads/experiments/<name>/main.py`, and parses its arguments with `argparse` so that `--help` works;
  - writes its outputs to `results/<name>/`;
  - has a `README.md` giving the question it answers, its method, how to run it, its outputs and what they show;
  - uses absolute imports (`from workloads.models.resnet import resnet18`).

  Code that several experiments share goes directly in `src/workloads/experiments/`.
- **Tests** mirror `src/workloads/` under `tests/`. Each experiment has a `tests/experiments/test_<name>.py` that, at minimum, checks that `--help` runs. Mark tests that download files or read `data/` with `@pytest.mark.slow`.
- **Results:** commit small text outputs (CSV). Plots (PNG) are gitignored.
- **Lint** before pushing. ruff requires a docstring on every public module, class and function.

  ```sh
  uv run ruff format
  uv run ruff check --fix
  ```
