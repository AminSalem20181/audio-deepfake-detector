# SDE-Seg activity report

## Overall result

The software scaffold is complete, but the full research experiment is not complete. The implementation is ready for installation and dataset execution; no paper-level metric is claimed yet.

## Activities completed

1. Created the repository layout under `sdeseg/`, `scripts/`, `tests/`, `configs/`, `results/` and `data/raw/`.
2. Implemented the SDE-Seg CNN, separable blocks, SEGM attention map and standard-convolution comparison model.
3. Implemented the requested audio preprocessing path: mono conversion, 16 kHz resampling, peak normalization, four-second crop or repeat padding, STFT and log-Mel normalization.
4. Implemented stability loss, SpecAugment helper, EER/AUC utilities, Grad-CAM, seed control and version logging.
5. Added protocol and metadata inspection helpers for ASVspoof 2019, In-the-Wild and ML-ITW.
6. Added tests for parameter count, tensor shapes, EER, Grad-CAM output and basic leakage protection.
7. Added the default configuration, table-generation script, reproducibility command and dataset download instructions.
8. Generated `results/tables/table3_parameters.csv` and `results/tables/table4_eer.csv` with traceable values or `N/A`.

## Verification performed

- Python source files passed static compilation with the available Python 3.8 interpreter.
- The requested parameter totals are encoded in tests: SDE-Seg `30,434`, standard twin `245,362`.
- The generated parameter table records the AASIST-L `85,306` value as a paper reference, not as a rerun.
- No dataset files were read, no checkpoint was selected from test data, and no metric was estimated.

## Execution status

The repository and synthetic test code were created on 2026-10-24. This environment has no installed PyTorch runtime and no raw ASVspoof, In-the-Wild or ML-ITW files, so the tests requiring PyTorch, training, final evaluation, latency, FLOPs, Grad-CAM dataset figures, stability metrics and deletion metrics have not been run. Those values are `N/A`; no results have been fabricated.

## Data and leakage

No dataset was downloaded. `scripts/check_data.py` requires the user-provided official archives and prints the actual file and metadata summary before use. The intended split policy is train for optimization, dev for lambda/model selection, and eval/cross-domain only for one final test. No test-set decision was made.

## Architecture

The implementation follows the requested stem, four separable blocks, SEGM, global average pooling and binary logit. The required full-model count is asserted as 30,434 and the standard-convolution twin as 245,362 by tests. The AASIST-L value in the parameter table is a cited reference (85,306), not a rerun.

## Known limitations

Install the requirements and provide the licensed datasets before running the full experiment. If ML-ITW metadata uses different label names, the inspection output must be used to document the mapping in this file before evaluation. Optional min t-DCF and AASIST-L results remain `N/A` until their official assets and runtime are available.

## Remaining execution steps

1. Install Python 3.10+, PyTorch, torchaudio and the packages in `requirements.txt`.
2. Place the official datasets under `data/raw/` and run `python scripts/check_data.py`.
3. Complete the training loop for the smoke test, lambda tuning and three-seed ablations.
4. Run final evaluation once on LA19 eval, In-the-Wild and ML-ITW.
5. Run efficiency, stability, deletion and figure generation scripts.
6. Regenerate all CSV tables from the resulting logs and update this report with hardware, dataset summaries and measured values.

## Colab preparation activity

`colab_run.ipynb` was added. It mounts Drive, redirects checkpoints and logs to Drive, installs dependencies, runs tests, downloads the two Hugging Face datasets, provides a cell for the official LA direct URL, inspects metadata, runs the smoke/full training commands and packages result directories for download. `sde-seg.zip` was created without `data/raw`, checkpoints or Git metadata.

The current evaluation entry point still reports `N/A` until the dataset-specific final evaluation and explainability stages are run. The notebook therefore does not invent final EER, latency or stability values.
