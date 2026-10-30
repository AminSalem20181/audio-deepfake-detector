"""Ordered experiment: smoke, dev-only lambda search, seeds, frozen final tests."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import yaml
from .train import train_run
from .data.dataset import ProtocolDataset, manifest_dataset, assert_disjoint
from .evaluate import evaluate_run
from .utils import save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='data/raw/LA')
    ap.add_argument('--results', default='results')
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--itw-manifest')
    ap.add_argument('--mlitw-manifest')
    ap.add_argument('--la-only', action='store_true', help='Explicitly leave cross-domain results N/A')
    ap.add_argument('--smoke-only', action='store_true')
    ap.add_argument('--train-only', action='store_true', help='Freeze dev-selected checkpoints without opening final-test data')
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    if not a.la_only and not a.smoke_only and not a.train_only and (not a.itw_manifest or not a.mlitw_manifest):
        ap.error('Provide both inspected cross-domain manifests, or explicitly use --la-only')
    cfg = yaml.safe_load(Path(a.config).read_text())
    out = Path(a.results)
    from collections import Counter
    import soundfile as sf
    checked = {}
    preflight = [ProtocolDataset(a.data_root, split) for split in ('train', 'dev')]
    assert_disjoint(*preflight)
    for split, dataset in zip(('train', 'dev'), preflight):
        checked[split] = dict(files=len(dataset), labels=dict(Counter(y for _, y in dataset.rows)),
                              duration_hours=sum(sf.info(p).duration for p, _ in dataset.rows)/3600)
    save_json(out/'logs/data_check.json', checked)
    train_run(cfg, a.data_root, out/'checkpoints/smoke', smoke=True, workers=a.workers)
    if a.smoke_only:
        return
    tuning = []
    for lam in cfg['lambda_grid']:
        print(f'Tuning lambda={lam:g}, seed=0', flush=True)
        checkpoint, value = train_run(cfg, a.data_root, out/f'checkpoints/tune_lambda{lam:g}',
                                      lam=lam, workers=a.workers)
        tuning.append(dict(lambda_value=lam, best_dev_eer=value, checkpoint=str(checkpoint)))
    selected = min(tuning, key=lambda x: x['best_dev_eer'])['lambda_value']
    save_json(out/'logs/lambda_selection.json', dict(candidates=tuning, selected_lambda=selected,
              policy='minimum seed-0 dev EER; grid order breaks ties'))
    checkpoints = []
    for variant in ('full', 'no_segm', 'segm_no_stab'):
        for seed in cfg['seeds']:
            print(f'Training variant={variant}, seed={seed}, selected_lambda={selected:g}', flush=True)
            path, _ = train_run(cfg, a.data_root, out/f'checkpoints/{variant}_seed{seed}',
                                variant, seed, selected, workers=a.workers)
            checkpoints.append(path)
    selection = dict(lambda_value=selected, checkpoints=[str(p) for p in checkpoints],
                     config=cfg, policy='best dev EER per run; centre 4 s primary test metric')
    frozen = out/'logs/frozen_selection.json'
    if frozen.exists() and json.loads(frozen.read_text()) != selection:
        raise ValueError('Selection has changed after being frozen; use a separate experiment directory')
    save_json(frozen, selection)
    if a.train_only:
        print(f'Training complete; frozen selection: {frozen}', flush=True)
        return
    # Test data are opened only after all model-selection decisions are frozen.
    cache = Path(a.data_root).resolve()/'.sdeseg_cache/test'
    datasets = {'LA19_eval': ProtocolDataset(a.data_root, 'eval', cache=cache)}
    if a.itw_manifest:
        datasets['In-The-Wild'] = manifest_dataset(a.itw_manifest, cache)
    if a.mlitw_manifest:
        datasets['ML-ITW'] = manifest_dataset(a.mlitw_manifest, cache)
    assert_disjoint(ProtocolDataset(a.data_root, 'train'), ProtocolDataset(a.data_root, 'dev'), *datasets.values())
    for path in checkpoints:
        for name, ds in datasets.items():
            evaluate_run(path, ds, out/f'logs/final/{path.parent.name}_{name}.json', name,
                         cfg['batch_size'], a.workers, windowed=name != 'LA19_eval')
    from .figures import create_figures
    for name, ds in datasets.items():
        create_figures(checkpoints[0], ds, out/'figures'/name)
    from .efficiency import benchmark
    benchmark(out/'logs/efficiency.json')
    from .tables import make_tables
    make_tables(out)
    from .report import write_report
    write_report(out)


if __name__ == '__main__':
    main()
