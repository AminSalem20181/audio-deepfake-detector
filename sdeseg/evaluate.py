"""Final scoring, prediction provenance and bounded explainability evaluation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader
from .data.dataset import ProtocolDataset, manifest_dataset
from .data.features import waveform_logmel
from .metrics import eer, auc
from .models import Sdeseg
from .explain import cam_stability, deletion_drop
from .utils import save_json, versions, seed_everything, source_fingerprint


def load_model(checkpoint):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if 'best_dev_eer' not in state or state['config'].get('smoke'):
        raise ValueError('Final evaluation requires a dev-selected, non-smoke checkpoint')
    if not state.get('history') or state['history'][-1]['dev_eer'] != state['best_dev_eer']:
        raise ValueError('Checkpoint is not from a best-dev epoch; use best.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Sdeseg(use_segm=state['variant'] != 'no_segm').to(device)
    model.load_state_dict(state['model']); model.eval()
    return model, state, device


def window_score(model, path, device):
    wav, sr = sf.read(path, always_2d=True)
    wav = torch.tensor(wav.mean(1), dtype=torch.float32)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    if not wav.numel() or not torch.isfinite(wav).all():
        raise ValueError(f'Invalid audio: {path}')
    wav = wav / wav.abs().max().clamp_min(1e-8)
    # Non-overlapping complete windows; discard trailing partial window.
    # Utterances shorter than 4 s use repeat padding as in centre scoring.
    chunks = list(wav[:(len(wav)//64000)*64000].split(64000)) if len(wav) >= 64000 else [wav.repeat((64000+len(wav)-1)//len(wav))[:64000]]
    scores = []
    with torch.no_grad():
        for chunk in chunks:
            scores.append(float(model(waveform_logmel(chunk)[None].to(device)).sigmoid().item()))
    return float(np.mean(scores))


def evaluate_run(checkpoint, dataset, output, dataset_name, batch_size=64, workers=2,
                 explain_samples=500, windowed=False):
    output = Path(output)
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    manifest_hash = hashlib.sha256(json.dumps(dataset.rows).encode()).hexdigest()
    identity = dict(checkpoint_sha256=digest, manifest_sha256=manifest_hash,
                    dataset=dataset_name, windowed=windowed, explain_samples=explain_samples,
                    implementation_sha256=source_fingerprint(('evaluate.py', 'data/*.py', 'models/*.py', 'explain/*.py', 'metrics.py')))
    if output.exists():
        result = json.loads(output.read_text())
        if result['identity'] != identity:
            raise ValueError(f'Refusing to overwrite final evaluation: {output}')
        return result
    model, state, device = load_model(checkpoint)
    seed_everything(state['seed'])
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_suffix('.started.json')
    # A crashed final test requires explicit recovery; never silently retest.
    with lock.open('x') as f:
        json.dump(identity, f, indent=2)
    labels, scores = [], []
    with torch.no_grad():
        for x, y in DataLoader(dataset, batch_size=batch_size, num_workers=workers):
            labels.extend(y.tolist())
            scores.extend(model(x.to(device)).sigmoid().flatten().cpu().tolist())
    result = dict(identity=identity, checkpoint=str(checkpoint), variant=state['variant'],
                  seed=state['seed'], count=len(labels), eer_percent=100*eer(labels, scores),
                  auc=auc(labels, scores), versions=versions(), crop_policy='centre_4s', min_tdcf=None,
                  class_counts={'bonafide': labels.count(0), 'spoof': labels.count(1)},
                  duration_hours=sum(sf.info(p).duration for p, _ in dataset.rows)/3600)
    predictions = dict(paths=[p for p, _ in dataset.rows], labels=labels, scores=scores)
    if windowed:
        ws = [window_score(model, p, device) for p, _ in dataset.rows]
        predictions['window_scores'] = ws
        result['windowed'] = dict(eer_percent=100*eer(labels, ws), auc=auc(labels, ws),
                                  policy='mean_probability_complete_nonoverlapping_4s; short_repeat_pad')
    n = min(explain_samples, len(dataset))
    indices = np.random.default_rng(0).choice(len(dataset), n, replace=False)
    stability, deletion = [], []
    for index in indices:
        x, y = dataset[int(index)]; x = x[None].to(device)
        stability.append(cam_stability(model, x))
        if int(y) == 1:
            deletion.append(dict(salient=deletion_drop(model, x), random=deletion_drop(model, x, random_control=True)))
    result['stability'] = {}
    for sigma in (0.05, 0.1, 0.2):
        values = [v[sigma] for v in stability]
        correlations = [v['spearman'] for v in values if v['spearman'] is not None]
        result['stability'][str(sigma)] = dict(count=n,
            mean_abs_difference=float(np.mean([v['mean_abs_difference'] for v in values])) if values else None,
            spearman=float(np.mean(correlations)) if correlations else None,
            valid_correlations=len(correlations))
    result['deletion'] = {kind: {str(p): float(np.mean([v[kind][p] for v in deletion])) if deletion else None
                          for p in (5,10,20,30)} for kind in ('salient','random')}
    result['deletion_spoof_count'] = len(deletion)
    result['explanation_indices'] = indices.tolist()
    save_json(output.with_suffix('.predictions.json'), predictions)
    save_json(output, result)
    lock.unlink()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--data-root', default='data/raw/LA')
    ap.add_argument('--manifest')
    ap.add_argument('--dataset', default='LA19_eval')
    ap.add_argument('--output', default='results/logs/evaluation.json')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--explain-samples', type=int, default=500)
    ap.add_argument('--windowed', action='store_true')
    a = ap.parse_args()
    dataset = manifest_dataset(a.manifest, 'results/cache') if a.manifest else ProtocolDataset(a.data_root, 'eval', cache='results/cache')
    print(json.dumps(evaluate_run(a.checkpoint, dataset, a.output, a.dataset, a.batch_size,
                                 a.workers, a.explain_samples, a.windowed), indent=2))


if __name__ == '__main__':
    main()
