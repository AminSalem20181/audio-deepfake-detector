"""Parse official LA protocols without silently accepting unknown labels."""
from pathlib import Path


def read_protocol(path: str | Path) -> list[dict]:
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        p = line.split()
        if len(p) != 5 or p[4] not in ('bonafide', 'spoof'):
            raise ValueError(f'Invalid protocol row {path}:{number}: {line}')
        rows.append(dict(speaker=p[0], utt_id=p[1], attack=p[3], label=int(p[4] == 'spoof'), key=p[4]))
    if not rows or len({r['utt_id'] for r in rows}) != len(rows):
        raise ValueError(f'Empty protocol or duplicate utterance IDs: {path}')
    return rows


def find_protocol(root: str | Path, split: str) -> Path:
    matches = sorted(Path(root).rglob(f'ASVspoof2019.LA.cm.{split}.*.txt'))
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one {split} protocol below {root}; found {len(matches)}')
    return matches[0]
