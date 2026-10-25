"""Create explainability figures when a checkpoint and labeled data are available."""
from pathlib import Path
import argparse

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--output', default='results/figures'); args=ap.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    print('Figure generation requires a trained checkpoint and dataset-specific file list; no figures fabricated.')

if __name__ == '__main__': main()
