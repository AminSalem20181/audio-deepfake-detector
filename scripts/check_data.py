"""Strict protocol/manifest inspection. Audio errors are fatal, never skipped."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import soundfile as sf
from sdeseg.data.dataset import ProtocolDataset, manifest_dataset, assert_disjoint
from sdeseg.utils import save_json


def describe(dataset):
    total = 0.0
    for path, _ in dataset.rows:
        total += sf.info(path).duration
    return dict(files=len(dataset), labels=dict(Counter(y for _,y in dataset.rows)),
                duration_hours=total/3600)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',default='data/raw/LA')
    ap.add_argument('--splits',nargs='+',choices=['train','dev','eval'],default=['train','dev'])
    ap.add_argument('--itw-manifest');ap.add_argument('--mlitw-manifest')
    ap.add_argument('--output',default='results/logs/data_check.json')
    a=ap.parse_args()
    datasets={split:ProtocolDataset(a.data_root,split) for split in a.splits}
    for name,path in [('In-The-Wild',a.itw_manifest),('ML-ITW',a.mlitw_manifest)]:
        if path: datasets[name]=manifest_dataset(path)
    assert_disjoint(*datasets.values())
    result={name:describe(ds) for name,ds in datasets.items()}
    save_json(a.output,result);print(result)


if __name__ == '__main__':
    main()
