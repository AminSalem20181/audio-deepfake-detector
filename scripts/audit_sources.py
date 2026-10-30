"""Record pinned public dataset metadata before downloading cross-domain audio."""
import argparse
import json
from pathlib import Path
import requests

REVISIONS = {
    'mueller91/In-The-Wild': 'eee168f92c367f8c82ff2cf42b6f61e362fd6211',
    'nccm2p2/ML-ITW': 'c4a69001e61acee4ea77404ee3d7406c9a377c5f',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    target = args.results / 'logs/dataset_sources.json'
    if target.exists():
        records = json.loads(target.read_text())
        for repo, revision in REVISIONS.items():
            if records.get(repo, {}).get('sha') != revision:
                raise ValueError(f'Existing dataset audit has a different revision: {repo}')
        print('Using recorded source audit:', target)
        return
    records = {}
    for repo, revision in REVISIONS.items():
        response = requests.get(f'https://huggingface.co/api/datasets/{repo}/revision/{revision}',
                                params={'blobs': 'true'}, timeout=30)
        response.raise_for_status()
        info = response.json()
        if info.get('sha') != revision or info.get('private') or info.get('gated'):
            raise ValueError(f'Unexpected dataset access or revision: {repo}')
        records[repo] = dict(id=info['id'], sha=info['sha'], card=info.get('cardData'),
            files=[dict(path=f.get('rfilename'), size=f.get('size'), lfs=f.get('lfs'))
                   for f in info.get('siblings', [])])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(records, indent=2))
    print('Saved source audit:', target)


if __name__ == '__main__':
    main()
