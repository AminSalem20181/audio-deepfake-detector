"""Run dev-only research training and persist progress for Colab monitoring."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
from sdeseg.utils import save_json


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--config',default='configs/default.yaml')
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    logs=args.results/'logs';logs.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-u','-m','sdeseg.experiment','--train-only',
             '--data-root',str(args.data_root),'--results',str(args.results),
             '--config',args.config,'--workers',str(args.workers)]
    record=dict(state='running',pid=os.getpid(),command=command,
                started_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    save_json(logs/'training_job.json',record)

    def progress(last_line: str) -> None:
        summaries=[]
        for path in sorted((args.results/'checkpoints').glob('*/train.json')):
            run=json.loads(path.read_text())
            if run['config'].get('smoke'):continue
            history=run.get('history',[])
            summaries.append(dict(run=path.parent.name,epochs=len(history),
                best_dev_eer_percent=100*run['best_dev_eer'],
                last_epoch=history[-1] if history else None))
        save_json(logs/'data_progress.json',dict(stage='training' if record['state']=='running' else record['state'],
                  updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  last_line=last_line.strip(),runs=summaries,
                  frozen_selection=(logs/'frozen_selection.json').exists()))
    progress('Starting dev-only lambda tuning and three-seed training')
    try:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line,end='',flush=True)
            progress(line)
        returncode=process.wait()
        record['state']='training_complete' if returncode==0 else 'failed'
        record['returncode']=returncode
        return returncode
    except Exception as error:
        record.update(state='failed',error=str(error))
        raise
    finally:
        record['updated_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        save_json(logs/'training_job.json',record)
        progress(record['state'])


if __name__=='__main__':
    raise SystemExit(main())
