"""Compact, explicitly labelled subset experiment with dev-only selection."""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import soundfile as sf
import yaml

from .data.asvspoof19 import find_protocol
from .data.dataset import AudioDataset, ProtocolDataset, assert_disjoint, manifest_dataset
from .train import train_run
from .utils import save_json, source_fingerprint


def stratified_rows(rows: list, limit: int, seed: int) -> list:
    """Sample without replacement, approximately preserving both class priors."""
    if limit < 2:
        raise ValueError('A binary evaluation subset needs at least two items')
    groups = [[i for i, (_, label) in enumerate(rows) if label == y] for y in (0, 1)]
    if not all(groups):
        raise ValueError('Both classes must exist before subset sampling')
    if len(rows) <= limit:
        return list(rows)
    n0 = max(1, min(len(groups[0]), limit - 1, round(limit * len(groups[0]) / len(rows))))
    counts = (n0, limit - n0)
    rng = np.random.default_rng(seed)
    selected = sorted(int(i) for group, count in zip(groups, counts)
                      for i in rng.choice(group, count, replace=False))
    return [rows[i] for i in selected]


def subset_record(rows: list, original_count: int, seed: int) -> dict:
    return dict(files=len(rows), original_files=original_count, subset_seed=seed,
                labels=dict(Counter(y for _, y in rows)),
                duration_hours=sum(sf.info(p).duration for p, _ in rows) / 3600,
                manifest_sha256=hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
                rows=rows)


def prepare_training(source: Path, target: Path, cfg: dict, results: Path) -> None:
    """Keep original protocol lines and symlink selected real train/dev audio."""
    records = {}
    for offset, split in enumerate(('train', 'dev')):
        original = ProtocolDataset(source, split)
        seed = cfg['subset_seed'] + offset
        rows = stratified_rows(original.rows, cfg[f'{split}_samples'], seed)
        audio = target / f'ASVspoof2019_LA_{split}/flac'
        audio.mkdir(parents=True, exist_ok=True)
        selected_ids = {Path(p).stem for p, _ in rows}
        protocol = find_protocol(source, split)
        lines = [line for line in protocol.read_text().splitlines()
                 if line.strip() and line.split()[1] in selected_ids]
        if len(lines) != len(rows):
            raise ValueError('Selected IDs did not match the original protocol')
        for path, _ in rows:
            link = audio / Path(path).name
            if link.exists():
                if link.resolve() != Path(path).resolve():
                    raise ValueError(f'Existing subset link differs: {link}')
            else:
                link.symlink_to(Path(path).resolve())
        destination = target / protocol.name
        content = '\n'.join(lines) + '\n'
        if destination.exists() and destination.read_text() != content:
            raise ValueError('Subset changed; use a new pilot directory')
        destination.write_text(content)
        records[split] = subset_record(rows, len(original), seed)
    assert_disjoint(ProtocolDataset(target, 'train'), ProtocolDataset(target, 'dev'))
    save_json(results/'logs/training_subsets.json', records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--subset-root', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--itw-manifest', type=Path, required=True)
    parser.add_argument('--mlitw-manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/pilot.yaml'))
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if cfg.get('experiment_kind') != 'pilot_subset':
        raise ValueError('Pilot runs require explicit subset labelling in the config')
    out = args.results
    config_path = out/'logs/pilot_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise ValueError('Pilot configuration changed; use a new output directory')
    save_json(config_path, cfg)
    save_json(out/'logs/experiment_profile.json', dict(kind='pilot_subset', config=cfg,
        limitation='Small stratified subsets and short training; not full-dataset paper benchmarks.',
        policy='Subset selection fixed before test scoring; original class proportions approximately preserved.'))
    started = time.monotonic()
    job = dict(pid=os.getpid(), hostname=platform.node(),
               started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
               implementation_sha256=source_fingerprint(('pilot.py', 'train.py', 'evaluate.py',
                   'data/*.py', 'models/*.py', 'explain/*.py', 'losses.py', 'metrics.py')))

    def progress(stage: str, **details) -> None:
        job.update(stage=stage, updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                   elapsed_seconds=time.monotonic()-started, **details)
        save_json(out/'logs/pilot_job.json', job)
        print(json.dumps(job), flush=True)

    try:
        progress('prepare_train_dev')
        prepare_training(args.data_root, args.subset_root, cfg, out)
        progress('smoke')
        train_run(cfg, args.subset_root, out/'checkpoints/smoke', smoke=True, workers=args.workers)
        tuning = []
        for lam in cfg['lambda_grid']:
            progress('tuning', lambda_value=lam)
            path, value = train_run(cfg, args.subset_root, out/f'checkpoints/tune_lambda{lam:g}',
                                    lam=lam, workers=args.workers)
            tuning.append(dict(lambda_value=lam, best_dev_eer=value, checkpoint=str(path)))
        selected = min(tuning, key=lambda r: r['best_dev_eer'])['lambda_value']
        save_json(out/'logs/lambda_selection.json', dict(candidates=tuning, selected_lambda=selected,
            policy='Minimum seed-0 subset-dev EER; grid order breaks ties; no final-test selection.'))
        checkpoints = []
        for variant in ('full', 'no_segm', 'segm_no_stab'):
            for seed in cfg['seeds']:
                progress('training', variant=variant, seed=seed, lambda_value=selected)
                path, _ = train_run(cfg, args.subset_root, out/f'checkpoints/{variant}_seed{seed}',
                                    variant, seed, selected, workers=args.workers)
                checkpoints.append(path)
        selection = dict(lambda_value=selected, checkpoints=[str(p) for p in checkpoints],
                         config=cfg, policy='pilot_subset; best subset-dev checkpoint per run')
        frozen = out/'logs/frozen_selection.json'
        if frozen.exists() and json.loads(frozen.read_text()) != selection:
            raise ValueError('Previously frozen pilot selection differs')
        save_json(frozen, selection)
        # Final datasets are first opened after every model choice is frozen.
        progress('prepare_final_subsets')
        originals = {'LA19_eval': ProtocolDataset(args.data_root, 'eval'),
                     'In-The-Wild': manifest_dataset(args.itw_manifest),
                     'ML-ITW': manifest_dataset(args.mlitw_manifest)}
        datasets, inspected = {}, {}
        for offset, (name, dataset) in enumerate(originals.items(), 10):
            seed = cfg['subset_seed'] + offset
            rows = stratified_rows(dataset.rows, cfg['eval_samples'], seed)
            datasets[name] = AudioDataset(rows, cache=args.subset_root/'.sdeseg_cache/test')
            inspected[name] = subset_record(rows, len(dataset), seed)
        assert_disjoint(ProtocolDataset(args.subset_root, 'train'),
                        ProtocolDataset(args.subset_root, 'dev'), *datasets.values())
        save_json(out/'logs/evaluation_subsets.json', inspected)
        from .evaluate import evaluate_run
        for checkpoint in checkpoints:
            for name, dataset in datasets.items():
                progress('evaluation', checkpoint=str(checkpoint), dataset=name)
                evaluate_run(checkpoint, dataset, out/f'logs/final/{checkpoint.parent.name}_{name}.json',
                             name, cfg['batch_size'], args.workers, cfg['explain_samples'],
                             windowed=name != 'LA19_eval')
        progress('figures_and_tables')
        from .figures import create_figures
        from .efficiency import benchmark
        from .tables import make_tables
        from .report import write_report
        for name, dataset in datasets.items():
            create_figures(checkpoints[0], dataset, out/'figures'/name)
        benchmark(out/'logs/efficiency.json')
        make_tables(out)
        write_report(out)
        progress('complete')
    except BaseException as error:
        progress('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
