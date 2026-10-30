"""Inspect real cross-domain metadata, then create an explicit binary manifest.

No dataset download and no label guessing. Use --inspect first, then supply
column names and an exact JSON mapping after reviewing the actual labels.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import soundfile as sf
from sdeseg.utils import save_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--metadata',required=True)
    ap.add_argument('--audio-root',required=True)
    ap.add_argument('--inspect',action='store_true')
    ap.add_argument('--path-column');ap.add_argument('--label-column')
    ap.add_argument('--language-column')
    ap.add_argument('--label-map',help='JSON, e.g. {"bona-fide":0,"spoof":1}')
    ap.add_argument('--output',default='data/manifests/cross_domain.csv')
    ap.add_argument('--mapping-note',help='Required rationale based on inspected dataset metadata/card')
    a=ap.parse_args();df=pd.read_csv(a.metadata)
    print('Columns:',list(df.columns))
    for col in df:
        print(col,'unique=',df[col].nunique(), 'values=',df[col].astype(str).value_counts().head(25).to_dict())
    if a.inspect: return
    if not all([a.path_column,a.label_column,a.label_map,a.mapping_note]):
        ap.error('Inspect first; provide path/label columns, exact label-map and mapping-note')
    mapping=json.loads(a.label_map)
    if not mapping or any(type(v) is not int for v in mapping.values()) or not set(mapping.values()) <= {0,1}:
        raise ValueError('Only 0=bonafide, 1=spoof allowed')
    if df.empty:
        raise ValueError('Metadata is empty')
    raw=df[a.label_column].astype(str)
    if not set(raw) <= set(mapping):
        raise ValueError(f'Unmapped labels: {set(raw)-set(mapping)}')
    root=Path(a.audio_root).resolve()
    paths=[(root/str(p)).resolve() for p in df[a.path_column]]
    if len(set(paths)) != len(paths): raise ValueError('Duplicate paths')
    duration=0.0
    for path in paths:
        if not path.is_relative_to(root): raise ValueError(f'Path escapes audio root: {path}')
        duration+=sf.info(path).duration
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(dict(path=[str(p) for p in paths],label=raw.map(mapping))).to_csv(output,index=False)
    summary=dict(metadata=str(Path(a.metadata).resolve()), columns=list(df.columns),
                 label_mapping=mapping, mapping_note=a.mapping_note, files=len(paths),
                 labels=raw.value_counts().to_dict(),duration_hours=duration/3600,
                 languages=df[a.language_column].value_counts().to_dict() if a.language_column else None)
    save_json(output.with_suffix('.inspection.json'),summary);print(summary)


if __name__ == '__main__': main()
