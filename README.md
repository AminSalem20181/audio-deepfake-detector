# SDE-Seg: Stable-Evidence-Guided Audio Deepfake Detection

This repository contains a reproducible implementation of SDE-Seg, a lightweight and explainable audio deepfake detector with a 30,434-parameter CNN and Stable Evidence Guidance Module (SEGM).

Important: this repository contains code and metadata only. Licensed datasets, checkpoints, caches, and measured results are intentionally excluded. No metric is claimed unless it was produced by a recorded run.

## Setup

```bash
python3.10 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Run the synthetic/unit-test suite:

```bash
PYTHONPATH=. python -m pytest -q
```

Place the licensed, official downloads under `data/raw/`:

```text
data/raw/LA/ASVspoof2019_LA_train/flac
data/raw/LA/ASVspoof2019_LA_dev/flac
data/raw/LA/ASVspoof2019_LA_eval/flac
data/raw/LA/ASVspoof2019_LA_cm_protocols
data/raw/in_the_wild/release_in_the_wild
data/raw/ml_itw
```

The LA archive is available from https://datashare.ed.ac.uk/handle/10283/3336. In-the-Wild is available from https://huggingface.co/datasets/mueller91/In-The-Wild and ML-ITW from https://huggingface.co/datasets/chlo1/ML-ITW. The code never downloads data automatically.

Run `python scripts/check_data.py` to inspect counts, duration, metadata columns and labels. Run `bash scripts/run_all.sh` after the data check. All measured metrics are written to `results/logs/` and generated tables to `results/tables/`.

## Training and evaluation

The training entry point is resumable and saves an epoch checkpoint plus a `*_latest.pt` checkpoint:

```bash
PYTHONPATH=. python -m sdeseg.train --data-root data/raw/LA --variant full --seed 0 --checkpoint-dir results/checkpoints
```

Variants are `full`, `no_segm`, and `segm_no_stab`. Use `--smoke --epochs 1` for a bounded pipeline check. Evaluate a real checkpoint only after model selection on dev:

```bash
PYTHONPATH=. python -m sdeseg.evaluate --checkpoint results/checkpoints/full_seed0_latest.pt --data-root data/raw/LA --split eval
```

`REPORT.md` records unavailable datasets, deviations, hardware, and any values that remain `N/A`.

For Google Colab, upload the project archive (or clone this repository), extract it as `/content/sde-seg`, select a T4 runtime, and run `colab_run.ipynb` in order. Store the raw LA archive in Drive and extract it into the local Colab disk; the notebook stores checkpoints and logs in `MyDrive/sde-seg-run` so a disconnected session can resume.

## Dataset citations

- ASVspoof 2019 LA: https://datashare.ed.ac.uk/handle/10283/3336
- In-The-Wild: https://huggingface.co/datasets/mueller91/In-The-Wild
- ML-ITW: https://huggingface.co/datasets/chlo1/ML-ITW

Respect the individual dataset licenses and do not commit or redistribute raw audio.

## Reproducibility

Seeds, configuration and software versions are logged for every training invocation. The model and comparison twin have parameter-count tests. Synthetic tensors are used only by tests; they are never reported as dataset results.
