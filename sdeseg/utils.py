"""Reproducibility and atomic JSON logging."""
from __future__ import annotations
import importlib.metadata
import json
import random
import platform
from pathlib import Path
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def versions() -> dict:
    packages = {}
    for package in ('torchaudio','numpy','scipy','scikit-learn','soundfile','pandas','fvcore'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return dict(python=platform.python_version(), pytorch=torch.__version__,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                platform=platform.platform(), packages=packages)


def save_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False))
    tmp.replace(path)


def source_fingerprint(patterns: tuple[str, ...]) -> str:
    """Hash implementation sources for resumable run/evaluation provenance."""
    import hashlib
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted({p for pattern in patterns for p in root.glob(pattern)}):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
