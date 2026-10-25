from __future__ import annotations
import json, random, platform
from pathlib import Path
import numpy as np
import torch

def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def versions() -> dict:
    return {"python": platform.python_version(), "pytorch": torch.__version__, "cuda": torch.version.cuda}

def save_json(path: str | Path, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True); Path(path).write_text(json.dumps(obj, indent=2, default=str))
