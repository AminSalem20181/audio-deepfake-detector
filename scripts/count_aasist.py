"""Count an official AASIST-L configuration without training or scoring audio."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('results/logs/aasist_parameters.json'))
    args = parser.parse_args()
    source = args.repository/'models/AASIST.py'
    config_path = args.repository/'config/AASIST-L.conf'
    config = json.loads(config_path.read_text())
    spec = importlib.util.spec_from_file_location('official_aasist', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(0)
    model = module.Model(config['model_config'])
    record = dict(repository='https://github.com/clovaai/aasist',
        commit=subprocess.check_output(['git', '-C', str(args.repository), 'rev-parse', 'HEAD'], text=True).strip(),
        configuration=config, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_parameters=sum(p.numel() for p in model.parameters()), torch=torch.__version__,
        measurement='Official AASIST-L configuration instantiated on CPU; no training or evaluation')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
