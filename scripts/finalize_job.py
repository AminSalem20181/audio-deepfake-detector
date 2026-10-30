"""Wait for frozen training and inspected data, then run final evaluation once."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--itw-manifest', type=Path, required=True)
    parser.add_argument('--mlitw-manifest', type=Path, required=True)
    args = parser.parse_args()
    logs = args.results / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    status = logs / 'final_job.json'
    record = dict(pid=os.getpid(), state='waiting', started_utc=dt.datetime.now(dt.timezone.utc).isoformat())

    def update(state: str, **details) -> None:
        record.update(state=state, updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(), **details)
        temporary = status.with_suffix('.tmp.json')
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(status)

    def read(name: str) -> dict:
        path = logs / name
        return json.loads(path.read_text()) if path.exists() else {}

    try:
        while True:
            training = read('training_job.json')
            cross = read('cross_progress.json')
            if training.get('state') == 'failed' or cross.get('stage') == 'failed':
                raise RuntimeError(f'Prerequisite failed: training={training}, cross={cross}')
            ready = (training.get('state') == 'training_complete'
                     and (logs / 'frozen_selection.json').is_file()
                     and cross.get('stage') == 'ready'
                     and args.itw_manifest.is_file() and args.mlitw_manifest.is_file())
            if ready:
                break
            update('waiting', training_state=training.get('state'), cross_stage=cross.get('stage'))
            time.sleep(30)
        command = [sys.executable, '-u', '-m', 'sdeseg.experiment',
                   '--data-root', str(args.data_root), '--results', str(args.results),
                   '--itw-manifest', str(args.itw_manifest), '--mlitw-manifest', str(args.mlitw_manifest)]
        update('evaluating', command=command)
        result = subprocess.run(command, check=False)
        update('complete' if result.returncode == 0 else 'failed', returncode=result.returncode)
        return result.returncode
    except Exception as error:
        update('failed', error=str(error))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
