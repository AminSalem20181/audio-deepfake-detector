#!/usr/bin/env bash
set -euo pipefail
python scripts/check_data.py
python -m pytest -q
python scripts/make_tables.py
if [ "${SMOKE_ONLY:-0}" = 1 ]; then
  python -m sdeseg.train --config configs/default.yaml --variant full --seed 0 --smoke --epochs 1
else
  for variant in full no_segm segm_no_stab; do
    for seed in 0 1 2; do
      python -m sdeseg.train --config configs/default.yaml --variant "$variant" --seed "$seed"
    done
  done
fi
python scripts/make_figures.py
