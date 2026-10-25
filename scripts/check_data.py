#!/usr/bin/env python3
from pathlib import Path
import soundfile as sf
import pandas as pd

ROOT = Path('data/raw')

def inspect_audio(folder):
    files = list(folder.rglob('*.flac')) + list(folder.rglob('*.wav'))
    total = 0.0
    for p in files:
        try:
            total += sf.info(p).duration
        except Exception:
            pass
    print(f'{folder}: files={len(files)} duration_hours={total/3600:.3f}')

def main():
    if not ROOT.exists():
        raise SystemExit('Missing data/raw. Download official datasets and place them according to README.md.')
    missing=[]
    for p in [ROOT/'LA', ROOT/'in_the_wild', ROOT/'ml_itw']:
        if p.exists(): inspect_audio(p)
        else: missing.append(p); print(f'MISSING {p} (required for final evaluation)')
    for meta in ROOT.rglob('meta.csv'):
        df = pd.read_csv(meta)
        print(meta, 'columns=', list(df.columns))
        for c in df.columns:
            print(' ', c, df[c].dropna().astype(str).value_counts().head(20).to_dict())
    la=ROOT/'LA'
    for split in ('train','dev','eval'):
        protocols=list(la.rglob(f'ASVspoof2019.LA.cm.{split}.*.txt')) if la.exists() else []
        if protocols:
            labels=[line.split()[-1] for line in protocols[0].read_text().splitlines() if line.strip()]
            print(f'LA {split}: protocol_rows={len(labels)} labels={pd.Series(labels).value_counts().to_dict()}')
    if missing: raise SystemExit(2)

if __name__ == '__main__': main()
