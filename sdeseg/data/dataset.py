"""Strict audio manifests shared by training and final evaluation."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .asvspoof19 import find_protocol, read_protocol
from .features import logmel
from .augment import spec_augment


class AudioDataset(Dataset):
    def __init__(self, rows, train=False, cache=None):
        self.rows = [(str(Path(p).resolve()), int(y)) for p, y in rows]
        self.train = train
        self.cache = Path(cache) if cache else None
        if not self.rows:
            raise ValueError('Empty audio manifest')
        if len({p for p, _ in self.rows}) != len(self.rows):
            raise ValueError('Duplicate audio files in manifest')
        for path, label in self.rows:
            if label not in (0, 1):
                raise ValueError(f'Invalid binary label: {label}')
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, label = self.rows[index]
        cached = None
        if self.cache and not self.train:
            stat = Path(path).stat()
            key = hashlib.sha256(f'logmel-v1:{path}:{stat.st_size}:{stat.st_mtime_ns}'.encode()).hexdigest()
            cached = self.cache / f'{key}.npy'
        if cached and cached.exists():
            x = torch.from_numpy(np.load(cached).astype(np.float32))
        else:
            x = logmel(path, self.train)
            if cached:
                # Each utterance has one worker per loader; rename avoids partial writes.
                tmp = cached.with_suffix(f'.{os.getpid()}.tmp.npy')
                np.save(tmp, x.numpy().astype(np.float16))
                tmp.replace(cached)
                x = x.half().float()
        if x.shape != (1, 64, 400) or not torch.isfinite(x).all():
            raise ValueError(f'Invalid feature tensor or corrupt cache: {path}')
        if self.train:
            x = spec_augment(x)
        return x, torch.tensor(label, dtype=torch.float32)


class ProtocolDataset(AudioDataset):
    def __init__(self, root, split, train=False, cache=None):
        root = Path(root)
        if split not in ('train', 'dev', 'eval') or (train and split != 'train'):
            raise ValueError('Only the train split may be used for training')
        folders = list(root.glob(f'**/ASVspoof2019_LA_{split}/flac'))
        if len(folders) != 1:
            raise ValueError(f'Expected one {split} audio directory under {root}, found {len(folders)}')
        rows = [(folders[0] / (r['utt_id'] + '.flac'), r['label'])
                for r in read_protocol(find_protocol(root, split))]
        super().__init__(rows, train, cache)


def manifest_dataset(path, cache=None):
    """CSV columns path,label; labels explicitly encoded spoof=1, bonafide=0.

    Relative paths are resolved against the manifest directory. ML-ITW mappings
    must be inspected and documented before creating this manifest.
    """
    path = Path(path)
    df = pd.read_csv(path)
    if not {'path', 'label'} <= set(df):
        raise ValueError('Manifest requires path,label columns')
    if not df.label.isin([0, 1]).all():
        raise ValueError('Manifest labels must be numeric 0=bonafide, 1=spoof')
    return AudioDataset([(path.parent / str(p), y) for p, y in zip(df.path, df.label)], cache=cache)


def assert_disjoint(*datasets):
    seen = set()
    for ds in datasets:
        paths = {str(Path(p).resolve()) for p, _ in ds.rows}
        if seen & paths:
            raise ValueError('Audio file overlap between dataset splits')
        seen |= paths
