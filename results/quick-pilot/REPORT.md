# SDE-Seg compact pilot report

**Scope: pilot_subset. These measurements are not full-dataset benchmark results.**

User-requested short run: up to 3000 training items, 1000 dev items, 1000 test items per dataset, 8 epochs per run, patience 3, and 30 explanation items per dataset/checkpoint.

Sampling approximately preserves original class proportions and uses fixed seeds. All variants share the same subsets. Model selection uses subset dev only. Subset manifests and original/evaluated counts are recorded in `logs/training_subsets.json` and `logs/evaluation_subsets.json`. These limits supersede the original full-experiment protocol at the user’s request.

Status: all 27 required final evaluations recorded. 27 final measurement records are available.

## Model selection

Selected lambda: 0.01. Selection uses LA train/dev only. Each run uses its own best-dev checkpoint. Final test measurements do not select the model.

## Primary centre-crop EER (%)

| variant | LA19_eval | In-The-Wild | ML-ITW | protocol |
| --- | --- | --- | --- | --- |
| full | 10.063029 ± 2.238533 | 53.327551 ± 14.990101 | 47.987606 ± 0.286559 | pilot_subset |
| no_segm | 10.055453 ± 3.106781 | 59.639180 ± 4.090201 | 50.356140 ± 2.826827 | pilot_subset |
| segm_no_stab | 10.137351 ± 2.551943 | 53.439833 ± 13.758934 | 48.598628 ± 0.604774 | pilot_subset |
| AASIST-L | N/A | N/A | N/A | pilot_subset |

Source: `tables/table4_eer.csv`.

Mean and sample standard deviation are reported only for all three seeds. Incomplete groups remain N/A. Small differences alone do not establish statistical significance.

## Cross-domain window-average EER (%)

| variant | LA19_eval | In-The-Wild | ML-ITW | protocol |
| --- | --- | --- | --- | --- |
| full | N/A | 52.648791 ± 14.569044 | 48.588033 ± 0.992132 | pilot_subset |
| no_segm | N/A | 59.045043 ± 4.411597 | 49.983761 ± 1.574472 | pilot_subset |
| segm_no_stab | N/A | 53.078556 ± 13.049879 | 48.313090 ± 1.369621 | pilot_subset |
| AASIST-L | N/A | N/A | N/A | pilot_subset |

Source: `tables/windowed_eer.csv`.

Window averaging uses complete non-overlapping four-second windows; short recordings use repeat padding. Centre crops remain the primary metric.

## Descriptive ablation findings

- LA19_eval: full mean EER is higher than no_segm by 0.007576 percentage points.
- LA19_eval: full mean EER is lower than segm_no_stab by 0.074322 percentage points.
- LA19_eval, noise sigma=0.05: full mean CAM difference is lower than segm_no_stab (0.02663916 versus 0.02735406).
- LA19_eval, noise sigma=0.1: full mean CAM difference is lower than segm_no_stab (0.05551024 versus 0.05701345).
- LA19_eval, noise sigma=0.2: full mean CAM difference is lower than segm_no_stab (0.10313599 versus 0.10610568).
- In-The-Wild: full mean EER is lower than no_segm by 6.311629 percentage points.
- In-The-Wild: full mean EER is lower than segm_no_stab by 0.112282 percentage points.
- In-The-Wild, noise sigma=0.05: full mean CAM difference is lower than segm_no_stab (0.02115405 versus 0.02187282).
- In-The-Wild, noise sigma=0.1: full mean CAM difference is lower than segm_no_stab (0.04312501 versus 0.04368545).
- In-The-Wild, noise sigma=0.2: full mean CAM difference is lower than segm_no_stab (0.08442196 versus 0.08561181).
- ML-ITW: full mean EER is lower than no_segm by 2.368534 percentage points.
- ML-ITW: full mean EER is lower than segm_no_stab by 0.611022 percentage points.
- ML-ITW, noise sigma=0.05: full mean CAM difference is lower than segm_no_stab (0.01686221 versus 0.01732737).
- ML-ITW, noise sigma=0.1: full mean CAM difference is lower than segm_no_stab (0.03222688 versus 0.03332768).
- ML-ITW, noise sigma=0.2: full mean CAM difference is lower than segm_no_stab (0.06327942 versus 0.06526851).

Higher EER or CAM difference is a negative result for the corresponding objective. These comparisons are descriptive and do not establish statistical significance.

## Executed parameter counts

| model | trainable_parameters | reduction_vs_standard_percent | reduction_vs_aasist_percent | source |
| --- | --- | --- | --- | --- |
| SDE-Seg | 30434 | 87.59628630350258 | 64.32372869434741 | logs/parameters.json |
| standard_conv_twin | 245362 | 0.0 | -187.62572386467542 | logs/parameters.json |
| AASIST-L | 85306 | 65.23259510437639 | 0 | logs/aasist_parameters.json |

Source: `tables/table3_parameters.csv`.

## Measured inference latency

| device | threads | median_ms | iqr_ms | source |
| --- | --- | --- | --- | --- |
| cpu | 1 | 14.32641799965495 | 0.6459422488660493 | logs/efficiency.json |
| cpu | 1 | 14.39943749937811 | 0.6448660005844431 | logs/efficiency.json |
| cuda | 1 | 0.968383999861544 | 0.08187950061255833 | logs/efficiency.json |

Source: `tables/latency.csv`.

## Counted network operations

| model | flops | tool | source |
| --- | --- | --- | --- |
| SDE-Seg | 60313728 | fvcore; one multiply-add counts as one FLOP; unsupported operators excluded | logs/efficiency.json |

Source: `tables/flops.csv`.

## Explanations and provenance

Per-seed AUC, dataset counts, class counts, duration and checkpoint SHA256 are in `logs/final/*.json` and `tables/eer_runs.csv`. Prediction files retain paths, labels and scores. Grad-CAM stability and matched-random deletion measurements are in `tables/stability.csv` and `tables/deletion.csv`; qualitative examples are in `figures/`.

Undefined constant-map rank correlations are recorded as null with valid counts retained. Optional AASIST-L retraining and official minimum t-DCF remain N/A. Efficiency excludes preprocessing; unsupported FLOP operators and hardware are recorded in `logs/efficiency.json`.

## Limits

Split checks reject repeated resolved paths; they do not detect independently copied or re-encoded recordings. ML-ITW source revision, shard checksums and observed label mapping are recorded separately before scoring. Results describe these datasets and training seeds; they do not establish performance for all unseen generators or languages.


## Method choices and deviations

- Class balancing: ordinary shuffled batches with BCE `pos_weight=N_bonafide/N_spoof`, computed from training rows only. No weighted sampler is used.
- Lambda: minimum seed-0 dev EER over the specified grid; grid order resolves ties. All seed runs use that selected value, even if zero wins. Each checkpoint uses its own best dev epoch.
- Precision: float32 training; AMP disabled. Dev/test centre features are cached as float16 and converted to float32 on both first and subsequent reads. No effect on real-data EER has yet been measured.
- Training augmentation is applied before the shared clean/noisy input pair; the noisy pass adds Gaussian noise and receives no additional SpecAugment masks.
- EER uses linear interpolation of the ROC crossing, positive class spoof. Table 4 uses centre crops and mean ± sample standard deviation (ddof=1) only when all three seeds are present.
- Window scores average probabilities of complete non-overlapping four-second windows, discard any trailing partial window, and repeat-pad clips shorter than four seconds. Centre scores remain primary.
- Explanation indices are selected uniformly without replacement with RNG seed 0, up to 500 utterances per dataset. Deletion is restricted to spoof examples among that subset; counts are logged. Constant CAMs have undefined Spearman correlation, stored as null with valid-correlation counts. MAD still includes those samples.
- Qualitative examples use the first up to six examples of each class from the manifest; figures use full seed 0 after model selection. Selection does not depend on test scores.
- Latency excludes preprocessing and uses an untrained network with synthetic input solely for timing. FLOPs follow fvcore's one-operation-per-multiply-add convention, with unsupported operations explicitly listed.
- Full workflow requires both cross-domain manifests. `--la-only` is an explicit partial experiment; unavailable cross-domain rows stay N/A.
- Optional AASIST-L retraining and official min t-DCF are not implemented as experiments and remain N/A. No reference EER is presented as a local result.
- Split isolation resolves paths, rejects repeated files and checks train/dev/test/cross-domain overlap. It does not detect separately copied or re-encoded duplicate recordings; data provenance remains necessary.
- Resume is at epoch boundaries. CUDA/CPU-library differences can affect numerical reproducibility; CUDA determinism settings and seeds are recorded, but bitwise equivalence across hardware is not claimed.



## Executed environment

```json
{
  "cpu": "Intel(R) Xeon(R) CPU @ 2.00GHz",
  "gpu": "Tesla T4",
  "platform": "Linux-6.6.159+-x86_64-with-glibc2.39",
  "versions": {
    "python": "3.13.16",
    "pytorch": "2.11.0+cu130",
    "cuda": "13.0",
    "cudnn": 92700,
    "platform": "Linux-6.6.159+-x86_64-with-glibc2.39",
    "packages": {
      "torchaudio": "2.11.0+cu130",
      "numpy": "2.1.3",
      "scipy": "1.16.3",
      "scikit-learn": "1.6.1",
      "soundfile": "0.14.0",
      "pandas": "2.2.3",
      "fvcore": "0.1.5.post20221221"
    }
  }
}
```


## Recorded data inspection before scoring

# Full-source dataset inspection

These are original source-dataset counts, not pilot evaluation counts. Pilot limits and sampled paths are recorded in logs/training_subsets.json and logs/evaluation_subsets.json.


## Cross-domain data inspection

{
  "dataset": "In-The-Wild",
  "repo": "mueller91/In-The-Wild",
  "revision": "eee168f92c367f8c82ff2cf42b6f61e362fd6211",
  "archive_sha256": "46665f30a6758b45642c659c7e0e80da2ba3d121b064716824a56a010ddf9e1a",
  "columns": [
    "file",
    "speaker",
    "label"
  ],
  "duration_hours": 37.85055604166685,
  "license": "cc-by-sa-4.0",
  "files": 31779,
  "labels": {
    "1": 11816,
    "0": 19963
  },
  "label_mapping": {
    "bona-fide": 0,
    "spoof": 1
  },
  "mapping_decision": "Observed bona-fide maps to 0 and spoof maps to 1; no other labels accepted."
}

## Cross-domain data inspection

{
  "dataset": "ML-ITW",
  "repo": "nccm2p2/ML-ITW",
  "revision": "c4a69001e61acee4ea77404ee3d7406c9a377c5f",
  "original_source": "chlo1/ML-ITW",
  "original_commit": "13037020c09d93814ba13715b1ca085f9ba25c9e",
  "original_hash_match": true,
  "parquet_files": {
    "data/eval-00000-of-00004.parquet": "a138b6a97597c260b18bfc8c223cbc0fe65ff34f422e52527fc1772a28c91efb",
    "data/eval-00001-of-00004.parquet": "838422c0d9489d96a166609e79e47dc24d57531a4584704074ff65b042eede65",
    "data/eval-00002-of-00004.parquet": "d3ad4e19d777366007a092f848c31bb339712181d774e13be3b812b62b06b58a",
    "data/eval-00003-of-00004.parquet": "b338416488dcb8edeaddef9f0d904ea087de6b755c01fd9bb8e8d4735f053f74"
  },
  "schemas": [
    "file: struct<bytes: binary, path: string>\n  child 0, bytes: binary\n  child 1, path: string\nspeaker: string\nlanguage: string\nplatform: string\nlabel: string\n-- schema metadata --\nhuggingface: '{\"info\": {\"features\": {\"file\": {\"_type\": \"Audio\"}, \"speaker' + 193",
    "file: struct<bytes: binary, path: string>\n  child 0, bytes: binary\n  child 1, path: string\nspeaker: string\nlanguage: string\nplatform: string\nlabel: string\n-- schema metadata --\nhuggingface: '{\"info\": {\"features\": {\"file\": {\"_type\": \"Audio\"}, \"speaker' + 193",
    "file: struct<bytes: binary, path: string>\n  child 0, bytes: binary\n  child 1, path: string\nspeaker: string\nlanguage: string\nplatform: string\nlabel: string\n-- schema metadata --\nhuggingface: '{\"info\": {\"features\": {\"file\": {\"_type\": \"Audio\"}, \"speaker' + 193",
    "file: struct<bytes: binary, path: string>\n  child 0, bytes: binary\n  child 1, path: string\nspeaker: string\nlanguage: string\nplatform: string\nlabel: string\n-- schema metadata --\nhuggingface: '{\"info\": {\"features\": {\"file\": {\"_type\": \"Audio\"}, \"speaker' + 193"
  ],
  "observed_labels": {
    "spoof": 6361,
    "bona-fide": 18168
  },
  "languages": {
    "english": 8529,
    "chinese": 8473,
    "japanese": 1569,
    "german": 1060,
    "hindi": 650,
    "portuguese": 1019,
    "korean": 928,
    "russian": 373,
    "spanish": 646,
    "hungarian": 299,
    "french": 256,
    "turkish": 588,
    "ukrainian": 90,
    "hebrew": 49
  },
  "duration_hours": 28.385892100694427,
  "license": "not declared in dataset card",
  "files": 24529,
  "labels": {
    "1": 6361,
    "0": 18168
  },
  "label_mapping": {
    "bona-fide": 0,
    "spoof": 1
  },
  "mapping_decision": "Observed bona-fide maps to 0 and spoof maps to 1; no other labels accepted."
}


## Post-run artifact verification

The complete 33,408,177-byte results archive downloaded from Colab matched SHA256 `dbde630831a5a4d6c41aee3240a587d579077f8e50bc2a43c6ba9ac71630dce1`. All 27 recorded centre EERs and AUCs, and all cross-domain window EERs, were recomputed from saved predictions without rescoring audio. All 27 evaluation-to-checkpoint SHA256 mappings were verified (nine distinct final checkpoints). Best-dev checkpoint status, exact architecture parameter counts, pilot configuration, evaluated sample counts and three-seed table aggregation were verified locally. Source: `results/quick-pilot/logs/local_artifact_verification.json`.

The full local suite passed **18 tests in 22.17 seconds**, including the complete synthetic pilot with frozen selection before test access. Test-fixture metrics are not research results. All five code cells in the clean final notebook compile and contain no saved debugging outputs. All 39 generated PNGs exist (12 Grad-CAM examples and one attention comparison for each dataset); representative CAM and clean/noisy attention figures were visually inspected.

The pilot process recorded 1,496.4397464429994 seconds from preparation through completed reports, on a Tesla T4. This is measured pilot execution time, excluding earlier implementation, source-data preparation and subsequent artifact review.

## Interpretation of the compact experiment

The full model's mean centre-crop EER was 10.063029% on the LA subset, 53.327551% on In-The-Wild and 47.987606% on ML-ITW. The cross-domain results do not demonstrate reliable generalisation in this short pilot. The full model's LA mean EER was slightly higher than no-SEGM (by 0.007576 percentage points); its differences from SEGM without stability were small relative to observed seed variation. Mean CAM differences were slightly lower with stability regularisation in the recorded comparisons, but the small explanation sample does not establish a robust improvement. These are descriptive findings; no claim of statistical significance is made. Scores were not inverted or selected using test EER.

The optional AASIST-L was instantiated only for its measured 85,306-parameter count; it was not trained or tested. Official minimum t-DCF remains N/A. Full-dataset results from the interrupted earlier workflow are not substituted for any pilot result.


Artifact paths such as `logs/final/` in this report refer to `results/quick-pilot/` locally or `MyDrive/sde-seg-run/research-v2/quick-pilot` in Drive. The conventional local `results/tables/` directory and its `results/logs/` sources are synchronized copies of the same completed pilot measurements.


## Presentation and GitHub supplement (2026-10-30)

Accuracy, Balanced Accuracy, spoof precision/recall/F1 and confusion matrices were computed from the same 27 saved prediction files with a fixed 0.5 threshold. No audio was rescored, no model was retrained, and no threshold was selected using final-test outcomes. See `results/pilot-summary/README_FA.md`, its CSVs and `summary_method.json`. Accuracy is not 100 minus EER. The full-model LA Accuracy was 89.30% on average, versus a majority-only Accuracy of 89.7%; its Balanced Accuracy was 89.31%, compared with 50% for the majority-only rule. Cross-domain Balanced Accuracy remained around 50%.

The first-person Persian presentation guide is `docs/PROFESSOR_BRIEF_FA.md`. Nine small original best-dev checkpoints are published in `pretrained/`, and their hashes match the final-evaluation identities. Canonical final predictions, measurements and selected training histories are included for reproducibility. Raw dataset audio is excluded. The expanded local suite passed **20 tests in 22.44 seconds**, including checks that perfect ROC ranking can coexist with poor fixed-threshold Accuracy and that majority-only Accuracy can hide balanced detection failure.
