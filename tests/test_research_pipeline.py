import json
import numpy as np
import pytest
import soundfile as sf
import torch
from sdeseg.data.features import logmel
from sdeseg.models import Sdeseg, count_parameters
from sdeseg.losses import sde_loss
from sdeseg.explain import deletion_drop, cam_stability
from sdeseg.metrics import eer


def test_features_and_empty_audio(tmp_path):
    path=tmp_path/'short.wav'
    sf.write(path,np.sin(np.arange(8000)*.04),8000)
    x=logmel(str(path))
    assert x.shape==(1,64,400) and torch.isfinite(x).all()
    assert abs(float(x.mean()))<1e-5
    sf.write(path,np.array([]),16000)
    with pytest.raises(ValueError): logmel(str(path))


def test_no_segm_has_no_unused_parameters():
    model=Sdeseg(False)
    assert model.segm is None and count_parameters(model)==30305
    loss,_,_=sde_loss(model,torch.randn(2,1,64,400),torch.tensor([0.,1.]),1.,0.)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())


def test_stability_gradients_and_deletion():
    model=Sdeseg().eval();x=torch.randn(2,1,64,400)
    loss,_,_=sde_loss(model,x,torch.tensor([0.,1.]),1.,1.)
    loss.backward()
    assert model.segm.weight.grad.abs().sum()>0
    result=deletion_drop(model,x,percentages=(100,))
    with torch.no_grad():
        expected=(model(x).sigmoid()-model(x.mean((1,2,3),keepdim=True).expand_as(x)).sigmoid()).mean()
    assert result[100]==pytest.approx(float(expected),abs=1e-6)
    result=cam_stability(model,x,sigmas=(0.,))
    assert result[0.]['mean_abs_difference']==0.


def test_invalid_metric_inputs():
    for y,s in [([0,0],[.1,.2]),([0,1],[float('nan'),.1]),([0,1],[.1])]:
        with pytest.raises(ValueError): eer(y,s)
    assert eer([0,1],[1.,0.])==1.


def test_training_resume_and_final_guard(tmp_path):
    from sdeseg.train import train_run
    from sdeseg.evaluate import evaluate_run
    from sdeseg.data.dataset import ProtocolDataset
    for split in ('train','dev','eval'):
        audio=tmp_path/'data'/f'ASVspoof2019_LA_{split}'/'flac';audio.mkdir(parents=True)
        lines=[]
        for i,label in enumerate(('bonafide','spoof')):
            name=f'{split}_{i}'
            sf.write(audio/f'{name}.flac',np.sin(np.arange(6400)*(.02+i*.03)),16000)
            lines.append(f'speaker {name} - A01 {label}')
        (tmp_path/'data'/f'ASVspoof2019.LA.cm.{split}.trl.txt').write_text('\n'.join(lines))
    cfg=dict(batch_size=2,lr=.001,weight_decay=.00001,max_epochs=1,min_lr=.00001,patience=10,sigma=.1)
    args=(cfg,tmp_path/'data',tmp_path/'run')
    best,value=train_run(*args,workers=0)
    assert best.exists() and 0<=value<=1
    assert train_run(*args,workers=0)[1]==value
    ds=ProtocolDataset(tmp_path/'data','eval')
    output=tmp_path/'evaluation.json'
    r=evaluate_run(best,ds,output,'synthetic_test_only',2,0,2,True)
    assert r['count']==2 and r['windowed']['auc']>=0
    assert evaluate_run(best,ds,output,'synthetic_test_only',2,0,2,True)==r
    with pytest.raises(ValueError): evaluate_run(best,ds,output,'changed',2,0,2,True)
    with pytest.raises(ValueError,match='mismatch'): train_run(dict(cfg,lr=.02),tmp_path/'data',tmp_path/'run',workers=0)


def test_tables_use_three_seeds_and_sources(tmp_path):
    from sdeseg.tables import make_tables
    import pandas as pd
    make_tables(tmp_path)
    table=pd.read_csv(tmp_path/'tables/table4_eer.csv',keep_default_na=False)
    assert table.iloc[0]['LA19_eval']=='N/A'
    logs=tmp_path/'logs/final';logs.mkdir()
    for seed in range(3):
        (logs/f'full_{seed}.json').write_text(json.dumps(dict(variant='full',seed=seed,eer_percent=float(seed),auc=.5,
            identity=dict(dataset='LA19_eval',checkpoint_sha256=f'hash{seed}'))))
    make_tables(tmp_path)
    table=pd.read_csv(tmp_path/'tables/table4_eer.csv',keep_default_na=False)
    assert table.iloc[0]['LA19_eval']=='1.000000 ± 1.000000'
    assert len(pd.read_csv(tmp_path/'tables/eer_runs.csv'))==3


def test_resume_reproduces_uninterrupted_weights(tmp_path, monkeypatch):
    import sdeseg.train as training
    root=tmp_path/'data'
    for split in ('train','dev'):
        audio=root/f'ASVspoof2019_LA_{split}'/'flac';audio.mkdir(parents=True)
        for i in range(2):
            sf.write(audio/f'{split}_{i}.flac',np.sin(np.arange(6400)*(.02+i*.03)),16000)
        (root/f'ASVspoof2019.LA.cm.{split}.trl.txt').write_text(
            '\n'.join(f'speaker {split}_{i} - A01 {label}' for i,label in enumerate(('bonafide','spoof'))))
    cfg=dict(batch_size=2,lr=.001,weight_decay=.00001,max_epochs=2,min_lr=.00001,patience=10,sigma=.1)
    training.train_run(cfg,root,tmp_path/'uninterrupted',workers=0)
    original=training.predict
    calls=0
    def interrupt_second_validation(*args):
        nonlocal calls
        calls+=1
        if calls==2: raise RuntimeError('simulated disconnect')
        return original(*args)
    monkeypatch.setattr(training,'predict',interrupt_second_validation)
    with pytest.raises(RuntimeError,match='simulated disconnect'):
        training.train_run(cfg,root,tmp_path/'resumed',workers=0)
    monkeypatch.setattr(training,'predict',original)
    training.train_run(cfg,root,tmp_path/'resumed',workers=0)
    left=torch.load(tmp_path/'uninterrupted/latest.pt',weights_only=False)
    right=torch.load(tmp_path/'resumed/latest.pt',weights_only=False)
    assert left['epoch']==right['epoch']==1
    assert all(torch.equal(value,right['model'][key]) for key,value in left['model'].items())


@pytest.mark.parametrize("train_only", [False, True])
def test_experiment_runs_all_stages_before_test_access(tmp_path, monkeypatch, train_only):
    """Synthetic fixture only: exercise tuning, seeds, freeze, test and artifacts."""
    import sys
    import yaml
    import pandas as pd
    import sdeseg.experiment as experiment
    root=tmp_path/'data';out=tmp_path/'synthetic_results'
    for split in ('train','dev','eval'):
        audio=root/f'ASVspoof2019_LA_{split}'/'flac';audio.mkdir(parents=True)
        for i in range(2):
            sf.write(audio/f'{split}_{i}.flac',np.sin(np.arange(6400)*(.02+i*.03)),16000)
        (root/f'ASVspoof2019.LA.cm.{split}.trl.txt').write_text(
            '\n'.join(f'speaker {split}_{i} - A01 {label}' for i,label in enumerate(('bonafide','spoof'))))
    cfg=dict(batch_size=2,lr=.001,weight_decay=.00001,max_epochs=1,min_lr=.00001,
             patience=10,sigma=.1,seeds=[0,1,2],lambda_grid=[0.,.01,.1,1.])
    config=tmp_path/'test.yaml';config.write_text(yaml.safe_dump(cfg))
    original=experiment.ProtocolDataset
    def guarded(root,split,*args,**kwargs):
        if split=='eval': assert (out/'logs/frozen_selection.json').exists()
        return original(root,split,*args,**kwargs)
    monkeypatch.setattr(experiment,'ProtocolDataset',guarded)
    monkeypatch.setattr(sys,'argv',['experiment','--config',str(config),'--data-root',str(root),
                                  '--results',str(out),'--workers','0','--train-only' if train_only else '--la-only'])
    experiment.main()
    if train_only:
        assert (out/'logs/frozen_selection.json').exists()
        assert not (out/'logs/final').exists()
        return
    table=pd.read_csv(out/'tables/table4_eer.csv',keep_default_na=False)
    assert all(table.iloc[:3]['LA19_eval']!='N/A')
    assert all(table['In-The-Wild']=='N/A')
    assert len(pd.read_csv(out/'tables/eer_runs.csv'))==9
    assert (out/'figures/LA19_eval/attention_noise.png').exists()
    assert (out/'logs/efficiency.json').exists()
