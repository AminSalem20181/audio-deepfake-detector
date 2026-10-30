"""Exercise actual split isolation; do not rely on function signatures."""
from pathlib import Path
import pytest
from sdeseg.data.dataset import ProtocolDataset, assert_disjoint


def make_split(root, split, utt='example', label='spoof'):
    audio=root/f'ASVspoof2019_LA_{split}'/'flac';audio.mkdir(parents=True)
    (audio/f'{utt}.flac').touch()
    (root/f'ASVspoof2019.LA.cm.{split}.trl.txt').write_text(f'speaker {utt} - A01 {label}\n')


def test_disjoint_and_overlap(tmp_path):
    make_split(tmp_path,'train');make_split(tmp_path,'dev')
    train=ProtocolDataset(tmp_path,'train',True);dev=ProtocolDataset(tmp_path,'dev')
    assert_disjoint(train,dev)
    with pytest.raises(ValueError,match='overlap'): assert_disjoint(train,train)
    with pytest.raises(ValueError): ProtocolDataset(tmp_path,'dev',True)


def test_training_opens_only_train_dev():
    import ast
    import sdeseg.train as training
    tree=ast.parse(Path(training.__file__).read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='ProtocolDataset']
    assert {n.args[1].value for n in calls} == {'train','dev'}
    assert 'manifest_dataset' not in Path(training.__file__).read_text()


def test_missing_audio_fails(tmp_path):
    make_split(tmp_path,'train');next(tmp_path.rglob('*.flac')).unlink()
    with pytest.raises(FileNotFoundError): ProtocolDataset(tmp_path,'train',True)
