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
