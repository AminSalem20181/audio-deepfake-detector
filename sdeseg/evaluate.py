import argparse, json
from pathlib import Path
import torch
from .metrics import eer, auc
from .models import Sdeseg
from .train import ProtocolDataset

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint', required=True); ap.add_argument('--data-root', default='data/raw/LA'); ap.add_argument('--split', default='eval'); ap.add_argument('--output', default='results/logs/evaluation.json'); ap.add_argument('--batch-size', type=int, default=64); a=ap.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); state=torch.load(a.checkpoint,map_location=device); model=Sdeseg(use_segm=state.get('variant','full') != 'no_segm').to(device); model.load_state_dict(state['model']); model.eval()
    dataset=ProtocolDataset(a.data_root, a.split, train=False); loader=torch.utils.data.DataLoader(dataset,batch_size=a.batch_size,num_workers=2,pin_memory=torch.cuda.is_available())
    ys=[]; scores=[]
    with torch.no_grad():
        for x,y in loader:
            scores.extend(torch.sigmoid(model(x.to(device))).squeeze(1).cpu().tolist()); ys.extend(y.tolist())
    if not ys: raise RuntimeError('No evaluation files matched protocol')
    result={'split':a.split,'count':len(ys),'eer_percent':100*eer(ys,scores),'auc':auc(ys,scores)}; Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__ == '__main__': main()
