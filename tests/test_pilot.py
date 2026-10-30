"""Synthetic fixtures only; pilot results from these tests are not research data."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import yaml

from sdeseg.pilot import stratified_rows


def test_stratified_subset_is_fixed_and_preserves_priors():
    rows = [(str(i), int(i >= 20)) for i in range(100)]
    subset = stratified_rows(rows, 30, 2026)
    assert len(subset) == 30 and sum(y == 0 for _, y in subset) == 6
    assert subset == stratified_rows(rows, 30, 2026)
    assert len({p for p, _ in subset}) == len(subset)
    assert subset != stratified_rows(rows, 30, 2027)
    assert {y for _, y in stratified_rows(rows, 2, 0)} == {0, 1}
    with pytest.raises(ValueError):
        stratified_rows(rows, 1, 0)


def test_pilot_freezes_before_test_access_and_labels_outputs(tmp_path, monkeypatch):
    import sdeseg.pilot as pilot
    root = tmp_path/'raw'
    out = tmp_path/'pilot'
    for split in ('train', 'dev', 'eval'):
        audio = root/f'ASVspoof2019_LA_{split}/flac'
        audio.mkdir(parents=True)
        lines = []
        for i in range(4):
            sf.write(audio/f'{split}_{i}.flac', np.sin(np.arange(4000)*(.02+i*.01)), 16000)
            lines.append(f'speaker {split}_{i} - A01 {"spoof" if i % 2 else "bonafide"}')
        (root/f'ASVspoof2019.LA.cm.{split}.trl.txt').write_text('\n'.join(lines))
    manifests = []
    for name in ('itw', 'mlitw'):
        rows = []
        for i in range(4):
            path = tmp_path/f'{name}_{i}.wav'
            sf.write(path, np.sin(np.arange(4000)*(.03+i*.01)), 16000)
            rows.append(dict(path=str(path), label=i % 2))
        path = tmp_path/f'{name}.csv'
        pd.DataFrame(rows).to_csv(path, index=False)
        manifests.append(path)
    cfg = yaml.safe_load(Path('configs/pilot.yaml').read_text())
    cfg.update(train_samples=2, dev_samples=2, eval_samples=2, explain_samples=2,
               max_epochs=1, batch_size=2, lambda_grid=[0., .1])
    config = tmp_path/'pilot.yaml'
    config.write_text(yaml.safe_dump(cfg))
    original_protocol = pilot.ProtocolDataset
    original_manifest = pilot.manifest_dataset

    def guarded_protocol(root, split, *args, **kwargs):
        if split == 'eval':
            assert (out/'logs/frozen_selection.json').exists()
        return original_protocol(root, split, *args, **kwargs)

    def guarded_manifest(*args, **kwargs):
        assert (out/'logs/frozen_selection.json').exists()
        return original_manifest(*args, **kwargs)

    monkeypatch.setattr(pilot, 'ProtocolDataset', guarded_protocol)
    monkeypatch.setattr(pilot, 'manifest_dataset', guarded_manifest)
    monkeypatch.setattr(sys, 'argv', ['pilot', '--data-root', str(root), '--subset-root',
        str(tmp_path/'subset'), '--results', str(out), '--itw-manifest', str(manifests[0]),
        '--mlitw-manifest', str(manifests[1]), '--config', str(config), '--workers', '0'])
    pilot.main()
    assert json.loads((out/'logs/pilot_job.json').read_text())['stage'] == 'complete'
    counts = json.loads((out/'logs/evaluation_subsets.json').read_text())
    assert all(r['files'] == 2 and r['original_files'] == 4 for r in counts.values())
    runs = pd.read_csv(out/'tables/eer_runs.csv')
    assert len(runs) == 27 and set(runs['count']) == {2}
    assert set(runs.protocol) == {'pilot_subset'}
    report = (out/'REPORT.md').read_text()
    assert 'not full-dataset benchmark results' in report
