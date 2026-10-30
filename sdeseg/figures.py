"""Qualitative centre-crop explanations from a selected trained checkpoint."""
from pathlib import Path
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'sdeseg-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from .evaluate import load_model
from .explain import gradcam
from .utils import save_json, seed_everything


def create_figures(checkpoint, dataset, output):
    model, state, device = load_model(checkpoint)
    seed_everything(0)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    selected = sum([[i for i, (_,y) in enumerate(dataset.rows) if y == label][:6] for label in (0,1)], [])
    for index in selected:
        x, y = dataset[index]; batch=x[None].to(device)
        cam = gradcam(model, batch)[0].cpu().numpy()
        fig, ax=plt.subplots(1,2,figsize=(12,3))
        spec=x[0].numpy()
        ax[0].imshow(spec,origin='lower',aspect='auto'); ax[0].set_title('Normalised log-Mel')
        ax[1].imshow(spec,origin='lower',aspect='auto'); ax[1].imshow(cam,origin='lower',aspect='auto',cmap='jet',alpha=.45)
        ax[1].set_title(f'Spoof-logit Grad-CAM; label={int(y)}')
        for axis in ax: axis.set_xlabel('Time frame'); axis.set_ylabel('Mel bin')
        fig.tight_layout(); fig.savefig(output/f'example_{index}.png',dpi=160); plt.close(fig)
    if selected and model.use_segm:
        x=dataset[selected[0]][0][None].to(device)
        with torch.no_grad():
            clean=model(x,return_maps=True)[2][0,0].cpu()
            noisy=model(x+torch.randn_like(x)*.1,return_maps=True)[2][0,0].cpu()
        fig,ax=plt.subplots(1,2,figsize=(10,3))
        for axis, value, title in zip(ax,(clean,noisy),('A(x)','A(x + noise), sigma=0.1')):
            axis.imshow(value.numpy(),origin='lower',aspect='auto',vmin=0,vmax=1);axis.set_title(title)
        fig.tight_layout();fig.savefig(output/'attention_noise.png',dpi=160);plt.close(fig)
    save_json(output/'provenance.json',dict(checkpoint=str(checkpoint),seed=state['seed'],
              examples=[dict(index=i,path=dataset.rows[i][0],label=dataset.rows[i][1]) for i in selected]))
