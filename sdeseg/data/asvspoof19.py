from pathlib import Path

def read_protocol(path: str | Path):
    rows=[]
    for line in Path(path).read_text().splitlines():
        p=line.split()
        if len(p) >= 5: rows.append({'speaker':p[0], 'utt_id':p[1], 'attack':p[3], 'label':0 if p[4]=='bonafide' else 1, 'key':p[4]})
    return rows

def find_protocol(root: str | Path, split: str) -> Path:
    root = Path(root)
    matches = sorted(root.rglob(f'ASVspoof2019.LA.cm.{split}.*.txt'))
    if not matches:
        raise FileNotFoundError(f'No ASVspoof protocol for split={split} below {root}')
    return matches[0]
