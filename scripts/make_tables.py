"""Regenerate CSV tables from stored measurements."""
import argparse
from sdeseg.tables import make_tables

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--results', default='results')
    make_tables(ap.parse_args().results)
