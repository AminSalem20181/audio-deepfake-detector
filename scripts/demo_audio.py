"""Offline classroom inference on one WAV/FLAC, with a self-contained HTML report."""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'sdeseg-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf
import torch

from sdeseg.data.features import load_audio, waveform_logmel
from sdeseg.evaluate import load_model
from sdeseg.explain import gradcam


def demonstrate(audio: Path, checkpoint: Path, output: Path, label: str | None = None) -> dict:
    """Score one example at fixed threshold 0.5; do not compute dataset accuracy."""
    if not audio.is_file():
        raise FileNotFoundError(f'Audio not found: {audio}')
    model, state, device = load_model(checkpoint)
    waveform = load_audio(str(audio), train=False)
    # Match the float16 feature-cache round trip used in recorded centre scoring.
    features = waveform_logmel(waveform).half().float()
    batch = features[None].to(device)
    with torch.no_grad():
        score = float(model(batch).sigmoid().item())
    cam = gradcam(model, batch)[0].cpu().numpy()
    prediction = 'spoof' if score >= .5 else 'bonafide'
    result = dict(audio=str(audio.resolve()), checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        variant=state['variant'], seed=state['seed'], device=str(device),
        score_spoof=score, threshold=0.5, prediction=prediction,
        expected_label=label, correct=prediction == label if label is not None else None,
        scope=state['config'].get('experiment_kind', 'full_protocol'),
        train_samples=state['config'].get('train_samples'),
        max_epochs=state['config']['max_epochs'],
        preprocessing='16 kHz mono; peak normalisation; centre 4 s; repeat-pad short audio; '
                      '64x400 normalised log-Mel; float16 cache-equivalent rounding',
        note='Single-example demonstration; score is not calibrated confidence or dataset accuracy.')
    output.mkdir(parents=True, exist_ok=True)
    (output/'prediction.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    sf.write(output/'processed.wav', waveform.numpy(), 16000, subtype='PCM_16')
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    spec = features[0].numpy()
    axes[0].imshow(spec, origin='lower', aspect='auto')
    axes[0].set_title('Actual model input: normalised log-Mel')
    axes[1].imshow(spec, origin='lower', aspect='auto')
    axes[1].imshow(cam, origin='lower', aspect='auto', cmap='jet', alpha=.45)
    axes[1].set_title(f'Spoof-logit Grad-CAM; score={score:.4f}')
    for ax in axes:
        ax.set_xlabel('Time frame')
        ax.set_ylabel('Mel bin')
    fig.tight_layout()
    fig.savefig(output/'gradcam.png', dpi=160)
    plt.close(fig)
    image = base64.b64encode((output/'gradcam.png').read_bytes()).decode()
    sound = base64.b64encode((output/'processed.wav').read_bytes()).decode()
    expected = 'مشخص نشده' if label is None else html.escape(label)
    correctness = 'بدون برچسب مرجع' if label is None else ('درست' if result['correct'] else 'نادرست')
    report = f'''<!doctype html>
<html lang="fa" dir="rtl"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SDE-Seg — دموی یک فایل صوتی</title>
<style>body{{font:17px/1.8 Tahoma,Arial,sans-serif;max-width:1050px;margin:30px auto;padding:0 20px;color:#18212b}}
section{{background:#f4f6f8;padding:18px;border-radius:12px;margin:16px 0}}img{{width:100%}}
code{{direction:ltr;display:inline-block}}audio{{width:100%}}meter{{width:100%;height:28px}}</style>
<h1>دموی SDE-Seg</h1>
<p>فایل: <code>{html.escape(audio.name)}</code> — نسخهٔ آزمایشی <code>{html.escape(result['scope'])}</code></p>
<section><b>پیش‌بینی: <code>{prediction}</code></b>
<p>امتیاز spoof: <code>{score:.6f}</code> — آستانهٔ ثابت: <code>0.5</code></p>
<meter min="0" max="1" value="{score}"></meter>
<p>برچسب مرجع: <code>{expected}</code> — نتیجهٔ این نمونه: {correctness}</p>
<p>spoof یعنی جعلی و bonafide یعنی واقعی. امتیاز مدل، دقت یا احتمال کالیبره‌شدهٔ صحت نیست.
این یک مثال است؛ عملکرد کلی در جدول‌های ارزیابی سه seed گزارش شده است.</p></section>
<h2>چهار ثانیهٔ پردازش‌شده‌ای که به مدل داده شد</h2>
<audio controls src="data:audio/wav;base64,{sound}"></audio>
<p>فایل بلند crop مرکزی و فایل کوتاه تکرار شده است. تصویر سمت راست، Grad-CAM برای امتیاز spoof است؛ نقشهٔ رنگی به‌تنهایی درستی تصمیم را ثابت نمی‌کند.</p>
<img alt="Log-Mel and spoof-logit Grad-CAM" src="data:image/png;base64,{image}">
<p>مدل pilot روی داده‌های بیرونی عملکرد ضعیفی داشته است؛ نمونهٔ درست یا غلط این دمو جایگزین ارزیابی مجموعه‌داده نیست.</p>
<details><summary>مشخصات مدل و پیش‌پردازش</summary><pre dir="ltr">{html.escape(json.dumps(result, indent=2))}</pre></details>
</html>'''
    (output/'report.html').write_text(report, encoding='utf-8')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True, help='A WAV or FLAC speech file')
    parser.add_argument('--checkpoint', type=Path, default=Path('pretrained/full_seed0.pt'))
    parser.add_argument('--output', type=Path, default=Path('results/demo/example'))
    parser.add_argument('--label', choices=('bonafide', 'spoof'), help='Optional known label, used only for display')
    args = parser.parse_args()
    result = demonstrate(args.audio, args.checkpoint, args.output, args.label)
    print(json.dumps(result, indent=2))
    print('OPEN_IN_BROWSER:', (args.output/'report.html').resolve())


if __name__ == '__main__':
    main()
