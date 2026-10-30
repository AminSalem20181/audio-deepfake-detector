"""Copy a provided LA archive to local disk, extract safely, and audit train/dev.

No network download. Archive SHA256 and preparation progress are recorded.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile


def prepare(archive: Path, data_root: Path, results: Path) -> None:
    from sdeseg.utils import save_json
    if not archive.is_file():
        raise FileNotFoundError(f'Provide the official LA.zip archive: {archive}')
    logs=results/'logs';logs.mkdir(parents=True,exist_ok=True)
    progress=logs/'data_progress.json'
    local=data_root.parent/'LA.local.zip'
    if archive.resolve()==local.resolve():
        raise ValueError('The input archive must differ from the local copy path')
    data_root.mkdir(parents=True,exist_ok=True)
    marker=data_root/'.extraction_complete.json'
    started=time.perf_counter()
    try:
        if not marker.exists():
            digest=hashlib.sha256();copied=0
            with archive.open('rb') as source, local.open('wb') as destination:
                while chunk:=source.read(16*1024*1024):
                    destination.write(chunk);digest.update(chunk);copied+=len(chunk)
                    save_json(progress,dict(stage='copy_archive',bytes=copied,total_bytes=archive.stat().st_size))
                    if copied%(256*1024*1024)==0:print(f'Copied {copied} bytes',flush=True)
            identity=dict(sha256=digest.hexdigest(),bytes=copied,source=str(archive))
            previous=logs/'la_archive.json'
            if previous.exists() and json.loads(previous.read_text())['sha256']!=identity['sha256']:
                raise ValueError('LA archive changed; do not resume this experiment with different data')
            save_json(previous,identity)
            with zipfile.ZipFile(local) as bundle:
                members=bundle.infolist()
                for index,member in enumerate(members,1):
                    target=(data_root/member.filename).resolve()
                    if not target.is_relative_to(data_root.resolve()):raise ValueError('Unsafe archive path')
                    bundle.extract(member,data_root)
                    if index%10000==0:
                        save_json(progress,dict(stage='extract_archive',files=index,total_files=len(members)))
                        print(f'Extracted {index}/{len(members)} entries',flush=True)
            save_json(marker,identity)
            local.unlink()
        save_json(progress,dict(stage='audit_train_dev'))
        subprocess.run([sys.executable,'scripts/check_data.py','--data-root',str(data_root),
                        '--output',str(logs/'data_check.json')],check=True)
        save_json(progress,dict(stage='ready',seconds=time.perf_counter()-started))
        print('LA_TRAIN_DEV_READY',flush=True)
    except Exception as error:
        save_json(progress,dict(stage='failed',error=str(error)))
        raise


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--archive',required=True,type=Path)
    ap.add_argument('--data-root',default='data/raw/LA',type=Path)
    ap.add_argument('--results',default='results',type=Path)
    args=ap.parse_args()
    prepare(args.archive,args.data_root,args.results)
