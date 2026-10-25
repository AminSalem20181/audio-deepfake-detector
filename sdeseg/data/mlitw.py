from pathlib import Path
import json
import pandas as pd

def summarize(root: str | Path):
    root=Path(root); files=[p for p in root.rglob('*') if p.suffix.lower() in {'.wav','.flac','.mp3'}]
    out={'files':len(files), 'root':str(root)}
    metas=list(root.rglob('*.csv'))
    if metas:
        frames=[]
        for p in metas:
            try: frames.append(pd.read_csv(p))
            except Exception: pass
        if frames:
            df=pd.concat(frames, ignore_index=True)
            out['columns']=list(df.columns)
            out['unique_values']={c:df[c].dropna().astype(str).unique().tolist()[:50] for c in df.columns}
    return out
