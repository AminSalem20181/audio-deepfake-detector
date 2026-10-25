from pathlib import Path
import pandas as pd

def inspect_metadata(path: str | Path):
    df=pd.read_csv(path); return {'columns':list(df.columns), 'values':{c:df[c].dropna().astype(str).unique().tolist() for c in df.columns}, 'rows':len(df)}
