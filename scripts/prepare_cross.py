"""Prepare verified cross-domain audio without running any model or metric."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
import requests
import soundfile as sf

ORIGINAL_ML_HASHES={
 'data/eval-00000-of-00004.parquet':'a138b6a97597c260b18bfc8c223cbc0fe65ff34f422e52527fc1772a28c91efb',
 'data/eval-00001-of-00004.parquet':'838422c0d9489d96a166609e79e47dc24d57531a4584704074ff65b042eede65',
 'data/eval-00002-of-00004.parquet':'d3ad4e19d777366007a092f848c31bb339712181d774e13be3b812b62b06b58a',
 'data/eval-00003-of-00004.parquet':'b338416488dcb8edeaddef9f0d904ea087de6b755c01fd9bb8e8d4735f053f74'}
LABELS={'bona-fide':0,'spoof':1}


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False))
    temporary.replace(path)


def digest(path: Path) -> str:
    result=hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda:file.read(8*1024*1024),b''):result.update(block)
    return result.hexdigest()


def download(repo: dict, record: dict, root: Path, progress: Path) -> Path:
    target=root/record['path'];target.parent.mkdir(parents=True,exist_ok=True)
    expected=record['lfs']['sha256']
    if target.exists():
        if digest(target)!=expected:raise ValueError(f'Existing file hash mismatch: {target}')
        return target
    partial=target.with_suffix(target.suffix+'.part')
    offset=partial.stat().st_size if partial.exists() else 0
    url=f"https://huggingface.co/datasets/{repo['id']}/resolve/{repo['sha']}/{record['path']}?download=true"
    headers={'Range':f'bytes={offset}-'} if offset else {}
    with requests.get(url,headers=headers,stream=True,timeout=(30,120)) as response:
        response.raise_for_status()
        if offset and response.status_code!=206:offset=0
        with partial.open('ab' if offset else 'wb') as file:
            copied=offset;reported=-1
            for block in response.iter_content(8*1024*1024):
                if not block:continue
                file.write(block);copied+=len(block)
                if copied//(64*1024*1024)!=reported:
                    reported=copied//(64*1024*1024)
                    save(progress,dict(stage='download',repo=repo['id'],file=record['path'],bytes=copied,total_bytes=record['size']))
    if digest(partial)!=expected:raise ValueError(f'Downloaded file SHA256 mismatch: {target}')
    partial.replace(target)
    return target


def manifest(path: Path, rows: list[dict], inspection: dict, report: Path) -> None:
    if {row['label'] for row in rows}!={0,1}:raise ValueError('Both binary classes are required')
    inspection.update(files=len(rows),labels=dict(Counter(row['label'] for row in rows)),label_mapping=LABELS,
        mapping_decision='Observed bona-fide maps to 0 and spoof maps to 1; no other labels accepted.')
    save(path.with_suffix('.inspection.json'),inspection)
    report.parent.mkdir(parents=True,exist_ok=True)
    with report.open('a') as file:
        file.write('\n## Cross-domain data inspection\n\n'+json.dumps(inspection,indent=2)+'\n')
    temporary=path.with_suffix('.tmp.csv');path.parent.mkdir(parents=True,exist_ok=True)
    with temporary.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    temporary.replace(path)
    print(json.dumps(inspection,indent=2),flush=True)


def prepare_itw(repo: dict, root: Path, out: Path, progress: Path, report: Path) -> None:
    import pandas as pd
    record=next(f for f in repo['files'] if f['path']=='release_in_the_wild.zip')
    archive=download(repo,record,root,progress)
    extracted=root/'audio';marker=extracted/'.complete.json'
    if not marker.exists():
        extracted.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            for index,member in enumerate(bundle.infolist(),1):
                if not (extracted/member.filename).resolve().is_relative_to(extracted.resolve()):raise ValueError('Unsafe ZIP path')
                bundle.extract(member,extracted)
                if index%5000==0:save(progress,dict(stage='extract_itw',entries=index))
        save(marker,dict(sha256=record['lfs']['sha256']))
    metadata=list(extracted.rglob('meta.csv'))
    if len(metadata)!=1:raise ValueError(f'Expected one actual ITW meta.csv; found {metadata}')
    frame=pd.read_csv(metadata[0]);print('ITW columns:',list(frame.columns),'labels:',frame['label'].unique(),flush=True)
    if not {'file','label'}<=set(frame):raise ValueError('Inspect the actual ITW column names before adapting')
    if not set(frame['label'].astype(str))<=set(LABELS):raise ValueError('Unknown ITW labels')
    rows=[];seconds=0.0
    for name,label in zip(frame['file'],frame['label']):
        audio=(metadata[0].parent/str(name)).resolve()
        if not audio.is_relative_to(extracted.resolve()):raise ValueError('ITW path escapes audio root')
        info=sf.info(audio)
        if info.frames<=0:raise ValueError(f'Empty audio: {audio}')
        seconds+=info.duration;rows.append(dict(path=str(audio),label=LABELS[str(label)]))
    manifest(out,rows,dict(dataset='In-The-Wild',repo=repo['id'],revision=repo['sha'],
        archive_sha256=record['lfs']['sha256'],columns=list(frame.columns),duration_hours=seconds/3600,
        license=(repo.get('card') or {}).get('license')),report)


def prepare_ml(repo: dict, root: Path, out: Path, progress: Path, report: Path) -> None:
    import pyarrow.parquet as pq
    records=[f for f in repo['files'] if f['path'].endswith('.parquet')]
    actual={f['path']:f['lfs']['sha256'] for f in records}
    if actual!=ORIGINAL_ML_HASHES:raise ValueError('ML-ITW copy does not match all original-author shard hashes')
    files=[download(repo,record,root,progress) for record in records]
    labels=Counter();languages=Counter();schemas=[]
    for file in files:
        table=pq.ParquetFile(file);schemas.append(str(table.schema_arrow))
        if not {'file','label','language'}<=set(table.schema_arrow.names):raise ValueError('Unexpected ML-ITW schema')
        for batch in table.iter_batches(batch_size=256,columns=['label','language']):
            for row in batch.to_pylist():labels[str(row['label'])]+=1;languages[str(row['language'])]+=1
    print('ML-ITW actual schemas:',schemas,'labels:',dict(labels),'languages:',dict(languages),flush=True)
    if not set(labels)<=set(LABELS):raise ValueError('Unknown ML-ITW labels; mapping requires inspection')
    exported=root/'exported_audio';exported.mkdir(parents=True,exist_ok=True)
    rows=[];seconds=0.0
    for file in files:
        for batch in pq.ParquetFile(file).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                value=row['file']
                if not isinstance(value,dict):raise ValueError('ML-ITW audio storage is not a bytes/path struct')
                blob=value.get('bytes')
                if blob is None:
                    source=(root/str(value.get('path',''))).resolve()
                    if not source.is_relative_to(root.resolve()):raise ValueError('External ML-ITW audio path')
                    blob=source.read_bytes()
                info=sf.info(io.BytesIO(blob))
                if info.frames<=0:raise ValueError('Empty ML-ITW audio')
                extension={'WAV':'wav','FLAC':'flac','OGG':'ogg'}.get(info.format,'audio')
                audio=exported/f'{len(rows):06d}.{extension}'
                if audio.exists() and audio.read_bytes()!=blob:raise ValueError('Existing exported audio differs')
                if not audio.exists():audio.write_bytes(blob)
                seconds+=info.duration
                rows.append(dict(path=str(audio.resolve()),label=LABELS[str(row['label'])],language=str(row['language'])))
                if len(rows)%1000==0:save(progress,dict(stage='export_ml_itw',files=len(rows)))
    if len(rows)!=sum(labels.values()):raise ValueError('ML-ITW export row-count mismatch')
    manifest(out,rows,dict(dataset='ML-ITW',repo=repo['id'],revision=repo['sha'],
        original_source='chlo1/ML-ITW',original_commit='13037020c09d93814ba13715b1ca085f9ba25c9e',
        original_hash_match=True,parquet_files=actual,schemas=schemas,
        observed_labels=dict(labels),languages=dict(languages),duration_hours=seconds/3600,
        license=(repo.get('card') or {}).get('license','not declared in dataset card')),report)


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--data-root',type=Path,default=Path('data/raw'))
    parser.add_argument('--results',type=Path,default=Path('results'));args=parser.parse_args()
    progress=args.results/'logs/cross_progress.json'
    audit=json.loads((args.results/'logs/dataset_sources.json').read_text())
    outputs=Path('data/manifests');outputs.mkdir(parents=True,exist_ok=True)
    try:
        prepare_itw(audit['mueller91/In-The-Wild'],args.data_root/'in_the_wild',outputs/'itw.csv',progress,args.results/'REPORT.md')
        prepare_ml(audit['nccm2p2/ML-ITW'],args.data_root/'ml_itw',outputs/'mlitw.csv',progress,args.results/'REPORT.md')
        save(progress,dict(stage='ready',itw_manifest=str((outputs/'itw.csv').resolve()),mlitw_manifest=str((outputs/'mlitw.csv').resolve())))
    except Exception as error:
        save(progress,dict(stage='failed',error=str(error)))
        raise


if __name__=='__main__':main()
