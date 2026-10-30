import argparse
from sdeseg.data.dataset import ProtocolDataset, manifest_dataset
from sdeseg.figures import create_figures

if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--data-root',default='data/raw/LA');ap.add_argument('--manifest')
    ap.add_argument('--output',default='results/figures')
    a=ap.parse_args()
    ds=manifest_dataset(a.manifest) if a.manifest else ProtocolDataset(a.data_root,'eval')
    create_figures(a.checkpoint,ds,a.output)
