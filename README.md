# SDE-Seg

Lightweight audio deepfake detection with SEGM stability training and spoof-logit Grad-CAM. Positive label: **spoof=1**; score: **P(spoof)**. The user-requested compact pilot completed on a Colab Tesla T4. All 27 final measurements and their checkpoint identities were verified locally. The earlier full-data training was stopped with checkpoints preserved. See REPORT.md for verified progress and limitations.

## Presentation, outputs and trained models

- [Persian professor briefing, written in first person](docs/PROFESSOR_BRIEF_FA.md): the project from beginning to end, why each step was taken, limitations and likely questions.
- [Pilot metrics and figures](results/pilot-summary/README_FA.md): Accuracy, Balanced Accuracy, EER, AUC, F1 and confusion matrices from the saved predictions.
- [Nine published trained checkpoints](pretrained/README.md), with hashes in [manifest.json](pretrained/manifest.json).

Reproduce the summary without training or rescoring audio:

```bash
python scripts/summarize_pilot.py
```

The summary uses a fixed score threshold of 0.5, chosen as a reporting convention, not fitted to test labels. Accuracy is not `100 − EER`. The majority-only baseline is included because class imbalance can make Accuracy misleading. ROC plots show all three full-model seeds; the illustrative confusion matrices use seed zero, not a test-selected winner. PNG and PDF figures are provided in `results/pilot-summary/figures/`.

Load a published best-dev model without retraining:

```python
from sdeseg.evaluate import load_model
model, training_state, device = load_model('pretrained/full_seed0.pt')
```

Canonical final metrics, saved predictions and selected training histories are published under `results/quick-pilot/` for traceability. Raw audio, caches, optimizer-resume copies, temporary browser-control notebooks and large archives are excluded from Git.

## Compact run (current user-requested scope)

`configs/pilot.yaml` limits training to 3,000 real LA utterances, dev to 1,000, each test dataset to 1,000, each run to eight epochs with patience three, and explanation analysis to 30 items. All four lambda candidates and all three variants × three seeds remain included. Fixed stratified sampling approximately preserves class proportions. Selection uses dev only; final test subsets are opened after all checkpoints are frozen.

```bash
python -m sdeseg.pilot \
  --data-root data/raw/LA --subset-root data/pilot-la \
  --results results/quick-pilot \
  --itw-manifest data/manifests/itw.csv \
  --mlitw-manifest data/manifests/mlitw.csv
```

Tables explicitly identify `pilot_subset`. These are short-run subset results, not full-dataset paper benchmarks. Source and evaluation manifests record actual counts and sampling seeds. Old full-data outputs are preserved separately. In the active Colab session, pilot artifacts are under `MyDrive/sde-seg-run/research-v2/quick-pilot`.

## Verified compact-pilot results

The measured pilot execution time was 1,496.44 seconds (about 25 minutes), excluding implementation and prior dataset preparation. The selected lambda was 0.01. Centre-crop EER (%), mean ± sample standard deviation across seeds 0/1/2:

| Variant | LA19 subset | In-The-Wild subset | ML-ITW subset |
| --- | --- | --- | --- |
| full | 10.063029 ± 2.238533 | 53.327551 ± 14.990101 | 47.987606 ± 0.286559 |
| no_segm | 10.055453 ± 3.106781 | 59.639180 ± 4.090201 | 50.356140 ± 2.826827 |
| segm_no_stab | 10.137351 ± 2.551943 | 53.439833 ± 13.758934 | 48.598628 ± 0.604774 |

Each test subset contains 1,000 utterances. These are **pilot_subset** results, not full-dataset benchmarks. The short run did not demonstrate reliable cross-domain detection; the report includes this negative finding and the small ablation differences. No optional baseline EER was fabricated.

Measured tables: [Table 4](results/quick-pilot/tables/table4_eer.csv), [window scores](results/quick-pilot/tables/windowed_eer.csv), [stability](results/quick-pilot/tables/stability.csv), [deletion](results/quick-pilot/tables/deletion.csv). See [REPORT.md](REPORT.md) for provenance, hardware, limitations and artifact verification. The final [Colab notebook](https://colab.research.google.com/drive/10u4HBNGurhci0aDYdQq4rQSaHSKjpg7Z) has five useful code cells. The complete results archive is `MyDrive/sde-seg-run/research-v2/SDE-Seg-pilot-results.zip`.

## Local setup

Use Python 3.10+ and a matching PyTorch/torchaudio installation suitable for your hardware. `requirements-lock-macos.txt` records the executed local Python 3.12 environment; Colab uses its matching GPU packages through `requirements.txt`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD"
python -m pytest -q
```

## Colab and your existing Drive files

Build the source bundle:

```bash
python scripts/package_colab.py
```

Use `SDE-Seg-final.ipynb` for the clean saved notebook with pinned source installation, real data preparation, training, evaluation and paper tables. It is also saved in the Drive run directory. The older `colab_run.ipynb` supports uploading a newly packaged source archive. It reads **MyDrive/LA.zip**, extracts audio to local Colab disk, and writes corrected experiments to **MyDrive/sde-seg-run/research-v2**. Change `LA_ZIP` if your archive is in another folder. Old `MyDrive/sde-seg-run/checkpoints` files are preserved. They cannot supply a best-dev checkpoint because the old trainer did not validate on dev; corrected paper experiments start separately. The compact run is complete. The clean notebook checks for an active duplicate pilot before changing runtime source files.

## Data

The LA archive is supplied by the user. Cross-domain preparation downloads pinned public sources, verifies their SHA256s, and records actual labels and audio statistics:

- [ASVspoof 2019 LA](https://datashare.ed.ac.uk/handle/10283/3336), DOI 10.7488/ds/298; LA.zip only, ODC-By.
- [In-The-Wild](https://huggingface.co/datasets/mueller91/In-The-Wild).
- [ML-ITW](https://huggingface.co/datasets/chlo1/ML-ITW).

Respect each dataset's license; do not redistribute raw audio. Extract LA.zip below `data/raw/LA/`; nested `LA/` directories are supported. The root must contain exactly one official protocol and audio folder per split:

```text
data/raw/LA/[optional LA/]
  ASVspoof2019_LA_train/flac/*.flac
  ASVspoof2019_LA_dev/flac/*.flac
  ASVspoof2019_LA_eval/flac/*.flac
  ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.*.txt
```

Inspect train/dev only before model selection:

```bash
python scripts/check_data.py --data-root data/raw/LA
```

This fails on missing protocol entries or unreadable audio and records actual class counts and hours. Use `--splits train dev eval` only for the final data audit.

For the inspected pinned public cross-domain releases:

```bash
pip install -r requirements-data.txt
python scripts/audit_sources.py --results results
python scripts/prepare_cross.py --data-root data/raw --results results
```

ML-ITW uses the public `nccm2p2/ML-ITW` copy only when all four shard hashes equal the original author’s published hashes. Actual downloaded bytes are checked before export; unknown labels or formats stop preparation. Inspection details are recorded before scoring. To prepare a different local CSV release, first inspect its actual dataset card, file tree and CSV:

```bash
python scripts/prepare_manifest.py --metadata /path/meta.csv --audio-root /path/audio --inspect
```

Then create an explicit manifest using the actual columns and label strings. Example **only if inspection confirms these names**:

```bash
python scripts/prepare_manifest.py \
  --metadata /path/meta.csv --audio-root /path/audio \
  --path-column file --label-column label \
  --label-map '{"bona-fide":0,"spoof":1}' \
  --mapping-note 'Mapping confirmed from dataset metadata and card' \
  --output data/manifests/itw.csv
```

For ML-ITW use its observed columns/mapping, add `--language-column` when available, and save `data/manifests/mlitw.csv`. The companion `.inspection.json` records counts, labels, languages and hours. Document the actual mapping and dataset card/license decision in REPORT.md before using ML-ITW. A supplied manifest must contain `path,label`, one audio file per row, numeric labels 0/1; relative paths resolve against the manifest directory. Audio archives/parquet formats must first be exported to audio files and a reviewed manifest; automatic ML-ITW format guessing is deliberately absent.

## Reproduce experiments

```bash
bash scripts/run_all.sh \
  --data-root data/raw/LA \
  --itw-manifest data/manifests/itw.csv \
  --mlitw-manifest data/manifests/mlitw.csv
```

The command runs tests, checks train/dev audio, performs a real 500-item smoke run, tunes lambda on seed-0 dev EER over `{0, .01, .1, 1}`, trains all three variants with seeds 0/1/2, freezes model selection, then evaluates each selected checkpoint on each test set. It also writes prediction-level provenance, stability/deletion results, figures and efficiency logs, then generates paper CSVs. Ties in lambda selection use grid order. Each run saves best/latest checkpoints, optimizer, cosine schedule, random states, config and epoch timings. Early stopping uses dev EER with patience 10. Repeating the command resumes matching training configurations and reuses identical final measurements.

For the real-data smoke only:

```bash
bash scripts/run_all.sh --data-root data/raw/LA --smoke-only
```

For an explicitly incomplete LA-only experiment add `--la-only` instead of cross-domain manifests. Cross-domain table entries remain N/A. Cross-domain data are never required for the train/dev smoke.

Final evaluation writes a `.started.json` journal. If it crashes, inspect that journal and the error before deliberately recovering; automatic retesting is refused. Test EERs must never be used to change model selection. A new output directory is not permission to tune against test results.

## Methods and artifacts

Preprocessing: mono, 16 kHz, peak normalisation, four-second random training/centre test crops, repeat padding; 512 FFT, 400-sample Hann window, 160 hop; first 400 frames; 64 log-Mel bands and utterance mean/std normalisation. Dev/test centre features are cached as float16. SpecAugment runs only in training; noisy stability inputs receive Gaussian noise after the training input is augmented and receive no additional masks. Both passes receive gradients. Class balancing uses BCE `pos_weight=N_bonafide/N_spoof` with ordinary shuffled batches; no weighted sampler. AMP is disabled.

EER linearly interpolates the ROC segment where FPR crosses FNR and is reported in percent. Table 4 uses centre crops and mean ± **sample** standard deviation across all three seeds; incomplete groups remain N/A. Additional cross-domain scores average probabilities over complete non-overlapping four-second windows, dropping a trailing partial window; clips shorter than four seconds use repeat padding.

Stability uses the same deterministic random subset of up to 500 utterances per dataset. Constant CAM rank correlations are undefined, recorded as null and excluded from the correlation average with valid counts retained. Deletion uses spoof samples from that subset, input-resolution Grad-CAM cells, utterance-mean replacement and matched random cell counts. Figures show up to six bona fide and six spoof examples plus SEGM clean/noisy attention.

Efficiency uses synthetic input solely to measure network cost: fvcore FLOPs (one multiply-add counted as one operation, unsupported operations logged), and batch-one latency excluding preprocessing, median/IQR of 200 runs after 20 warm-ups at one/default CPU thread counts and CUDA if available. Hardware and individual timings are saved.

```bash
python -m sdeseg.efficiency
python scripts/make_tables.py --results results
```

`results/logs/parameters.json` supports Table 3. `results/logs/final/*.json` plus prediction files support Table 4, per-run EER/AUC, stability and deletion CSVs. `eer_runs.csv` records checkpoint hashes and source logs. No metric is filled from a paper reference. Optional AASIST-L retraining and official min t-DCF remain N/A.

## Clean final notebook

`SDE-Seg-final.ipynb` contains six functional code cells for Drive/configuration, source installation, LA preparation, dev-only model training, final scoring/explanations, and table generation. Temporary browser diagnostics and editor tests are excluded. Its source archive must exist at the recorded Drive artifact path. Dataset results remain unavailable until those runs finish.

To prepare an existing LA archive locally:

```bash
PYTHONPATH=. python scripts/prepare_la.py --archive /path/LA.zip --data-root data/raw/LA --results results
```

To train and freeze model selection before opening final-test datasets:

```bash
bash scripts/run_all.sh --data-root data/raw/LA --train-only
```

`PYTHONPATH=. python scripts/train_job.py --data-root data/raw/LA --results results` additionally records a compact progress snapshot and process status for Colab monitoring. It uses the same train-only experiment; it does not score final-test datasets.

To reproduce the measured AASIST-L parameter count from an official checkout:

```bash
git clone https://github.com/clovaai/aasist.git /tmp/aasist
git -C /tmp/aasist checkout a04c9863f63d44471dde8a6abcb3b082b07cd1d1
python scripts/count_aasist.py --repository /tmp/aasist
python scripts/make_tables.py --results results
```

This instantiates the official model on CPU; it does not train the optional baseline or produce baseline EERs.
