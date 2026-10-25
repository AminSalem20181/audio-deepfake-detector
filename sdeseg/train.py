"""Resumable SDE-Seg training entry point for local and Colab execution."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch, yaml
from torch.utils.data import Dataset, DataLoader
from .data.asvspoof19 import read_protocol, find_protocol
from .data.features import logmel
from .models import Sdeseg, count_parameters
from .losses import sde_loss
from .utils import versions, save_json, seed_everything

class ProtocolDataset(Dataset):
    def __init__(self, root: str, split: str, train: bool=False):
        self.root=Path(root); self.train=train
        p=find_protocol(self.root, split)
        name={'train':'train','dev':'dev','eval':'eval'}[split]
        audio=next(self.root.glob(f'**/ASVspoof2019_LA_{name}/flac'))
        self.rows=[]
        for r in read_protocol(p):
            f=audio/(r['utt_id']+'.flac')
            if f.exists(): self.rows.append((str(f),r['label']))
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        f,y=self.rows[i]; return logmel(f,self.train), torch.tensor(y,dtype=torch.float32)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/default.yaml'); ap.add_argument('--data-root',default='data/raw/LA'); ap.add_argument('--variant',default='full'); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--epochs',type=int); ap.add_argument('--checkpoint-dir',default='results/checkpoints'); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--lambda-value',type=float,default=None); ap.add_argument('--max-items',type=int,default=None); args=ap.parse_args()
    cfg=yaml.safe_load(open(args.config)); seed_everything(args.seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=Sdeseg(use_segm=args.variant!='no_segm').to(device)
    if args.variant!='no_segm': assert count_parameters(model)==30434
    train=ProtocolDataset(args.data_root,'train',True)
    if args.smoke:
        # Keep both classes in the bounded smoke subset; the protocol starts with bona fide rows.
        by_class = {0: [r for r in train.rows if r[1] == 0], 1: [r for r in train.rows if r[1] == 1]}
        n_each = min(250, len(by_class[0]), len(by_class[1]))
        train.rows = by_class[0][:n_each] + by_class[1][:n_each]
    elif args.max_items:
        train.rows = train.rows[:args.max_items]
    labels=[y for _,y in train.rows]; n_bon=sum(y == 0 for y in labels); n_spoof=sum(y == 1 for y in labels)
    if n_spoof == 0: raise RuntimeError('Training protocol contains no spoof examples')
    pos_weight=n_bon / n_spoof
    loader=DataLoader(train,batch_size=cfg['batch_size'],shuffle=True,num_workers=2,pin_memory=torch.cuda.is_available(),persistent_workers=True)
    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg['max_epochs'],eta_min=cfg['min_lr'])
    ckpt=Path(args.checkpoint_dir); ckpt.mkdir(parents=True,exist_ok=True); latest=ckpt/f'{args.variant}_seed{args.seed}_latest.pt'; start=0
    if latest.exists():
        state=torch.load(latest,map_location=device); model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); sched.load_state_dict(state['scheduler']); start=state['epoch']+1
    epochs=args.epochs or (1 if args.smoke else cfg['max_epochs']); history=[]
    for epoch in range(start,epochs):
        model.train(); total=0.0
        for step,(x,y) in enumerate(loader, 1):
            x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True)
            lam=args.lambda_value if args.lambda_value is not None else (0.1 if args.variant == 'full' else 0.0)
            loss,_,_=sde_loss(model,x,y,pos_weight,lam,float(cfg['sigma'])); loss.backward(); opt.step(); total+=float(loss.detach())
            if step == 1 or step % 25 == 0 or step == len(loader):
                print(f'epoch={epoch} step={step}/{len(loader)} loss={float(loss.detach()):.4f}', flush=True)
        sched.step(); state={'epoch':epoch,'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'config':cfg,'seed':args.seed,'variant':args.variant,'pos_weight':pos_weight}; torch.save(state,latest); torch.save(state,ckpt/f'{args.variant}_seed{args.seed}_epoch{epoch:03d}.pt')
        history.append({'epoch':epoch,'loss':total/max(1,len(loader))}); save_json(f'results/logs/train_{args.variant}_{args.seed}.json',{'config':cfg,'seed':args.seed,'versions':versions(),'parameters':count_parameters(model),'history':history})
    print(json.dumps(history[-1] if history else {'status':'resumed'},indent=2))
if __name__=='__main__': main()
