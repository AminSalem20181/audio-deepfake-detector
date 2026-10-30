"""Dev-selected training; this module never opens final-test datasets."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from .data.dataset import ProtocolDataset, assert_disjoint
from .models import Sdeseg, count_parameters
from .losses import sde_loss
from .metrics import eer
from .utils import versions, save_json, seed_everything, source_fingerprint


def predict(model, loader, device):
    model.eval()
    labels, scores = [], []
    with torch.no_grad():
        for x, y in loader:
            scores.extend(model(x.to(device)).sigmoid().flatten().cpu().tolist())
            labels.extend(y.tolist())
    return labels, scores


def train_run(cfg, data_root, output, variant='full', seed=0, lam=0.1,
              smoke=False, epochs=None, workers=2):
    fixed_features = dict(sample_rate=16000, duration_seconds=4, n_fft=512, win_length=400,
                          hop_length=160, n_mels=64, frames=400)
    if any(cfg.get(key, value) != value for key, value in fixed_features.items()):
        raise ValueError('Feature configuration must match the paper preprocessing constants')
    cfg = dict(fixed_features, **cfg)
    seed_everything(seed)
    if variant not in ('full', 'no_segm', 'segm_no_stab'):
        raise ValueError(variant)
    lam = float(lam) if variant == 'full' else 0.0
    effective = dict(cfg, variant=variant, seed=seed, lambda_value=lam,
                     smoke=smoke, epochs=epochs or (1 if smoke else cfg['max_epochs']),
                     data_root=str(Path(data_root).resolve()), workers=workers, amp=False,
                     implementation_sha256=source_fingerprint(('train.py', 'data/*.py', 'models/*.py', 'losses.py', 'metrics.py')))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train = ProtocolDataset(data_root, 'train', True)
    dev = ProtocolDataset(data_root, 'dev', cache=Path(data_root).resolve() / '.sdeseg_cache/dev')
    assert_disjoint(train, dev)
    if smoke:
        train.rows = sum(
            [[r for r in train.rows if r[1] == y][:250] for y in (0, 1)], [])
        dev.rows = sum([[r for r in dev.rows if r[1] == y][:100] for y in (0, 1)], [])
    effective['train_manifest_sha256'] = hashlib.sha256(json.dumps(train.rows).encode()).hexdigest()
    effective['dev_manifest_sha256'] = hashlib.sha256(json.dumps(dev.rows).encode()).hexdigest()
    labels = [y for _, y in train.rows]
    if set(labels) != {0, 1}:
        raise ValueError('Training needs both classes')
    pos_weight = labels.count(0) / labels.count(1)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train, batch_size=cfg['batch_size'], shuffle=True,
                        num_workers=workers, generator=generator)
    dev_loader = DataLoader(dev, batch_size=cfg['batch_size'], num_workers=workers)
    model = Sdeseg(use_segm=variant != 'no_segm').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['max_epochs'], eta_min=cfg['min_lr'])
    latest, best = output / 'latest.pt', output / 'best.pt'
    history, start, best_eer, stale = [], 0, float('inf'), 0
    if latest.exists():
        state = torch.load(latest, map_location='cpu', weights_only=False)
        if state['config'] != effective:
            raise ValueError(f'Run configuration mismatch at {latest}; use a new output directory')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        for values in optimizer.state.values():
            for key, value in values.items():
                if torch.is_tensor(value):
                    values[key] = value.to(device)
        scheduler.load_state_dict(state['scheduler'])
        start, history = state['epoch'] + 1, state['history']
        best_eer, stale = state['best_dev_eer'], state['stale']
        random.setstate(state['rng_python']); np.random.set_state(state['rng_numpy'])
        torch.set_rng_state(state['rng_torch']); generator.set_state(state['rng_loader'])
        if device.type == 'cuda' and state['rng_cuda'] is not None:
            torch.cuda.set_rng_state_all(state['rng_cuda'])
    for epoch in range(start, effective['epochs']):
        if stale >= cfg['patience']:
            break
        begin = time.perf_counter()
        model.train()
        totals = np.zeros(3)
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, bce, stability = sde_loss(model, x, y, pos_weight, lam, cfg['sigma'])
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            loss.backward(); optimizer.step()
            totals += np.array([loss.item(), bce.item(), stability.item()]) * len(y)
        ys, scores = predict(model, dev_loader, device)
        dev_eer = eer(ys, scores)
        improved = dev_eer < best_eer
        best_eer, stale = (dev_eer, 0) if improved else (best_eer, stale + 1)
        scheduler.step()
        row = dict(epoch=epoch, loss=totals[0]/len(train), bce=totals[1]/len(train),
                   stability=totals[2]/len(train), dev_eer=dev_eer, seconds=time.perf_counter()-begin)
        history.append(row)
        state = dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), config=effective, seed=seed, variant=variant,
                     pos_weight=pos_weight, best_dev_eer=best_eer, stale=stale, history=history,
                     rng_python=random.getstate(), rng_numpy=np.random.get_state(),
                     rng_torch=torch.get_rng_state(), rng_loader=generator.get_state(),
                     rng_cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
        for target in ([latest, best] if improved else [latest]):
            tmp = target.with_suffix('.tmp'); torch.save(state, tmp); tmp.replace(target)
        save_json(output/'train.json', dict(config=effective, versions=versions(), device=str(device),
                  parameters=count_parameters(model), pos_weight=pos_weight, history=history,
                  best_dev_eer=best_eer, train_count=len(train), dev_count=len(dev),
                  train_class_counts={0: labels.count(0), 1: labels.count(1)},
                  hardware=dict(cpu=platform.processor(),
                                gpu=torch.cuda.get_device_name() if device.type == 'cuda' else None)))
        print(row, flush=True)
    return best, best_eer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--data-root', default='data/raw/LA')
    ap.add_argument('--variant', choices=['full','no_segm','segm_no_stab'], default='full')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int)
    ap.add_argument('--checkpoint-dir', default='results/checkpoints')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--lambda-value', type=float, default=0.1)
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    tag = f'{a.variant}_seed{a.seed}_lambda{a.lambda_value:g}' + ('_smoke' if a.smoke else '')
    train_run(cfg, a.data_root, Path(a.checkpoint_dir)/tag, a.variant, a.seed,
              a.lambda_value, a.smoke, a.epochs, a.workers)


if __name__ == '__main__':
    main()
