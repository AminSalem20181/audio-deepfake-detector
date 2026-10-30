"""Summarize existing predictions; never train, rescore audio, or tune a threshold."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'sdeseg-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_curve)
from sdeseg.metrics import auc, eer

DATASETS = ('LA19_eval', 'In-The-Wild', 'ML-ITW')
VARIANTS = ('full', 'no_segm', 'segm_no_stab')
METRICS = ('eer_percent', 'auc', 'accuracy_percent', 'balanced_accuracy_percent',
           'precision_spoof_percent', 'recall_spoof_percent', 'f1_spoof_percent')


def fixed_threshold_metrics(labels, scores) -> dict:
    """Positive class spoof=1; 0.5 is fixed, not fitted on final-test labels."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or set(labels) != {0, 1}:
        raise ValueError('Two-class, matching one-dimensional predictions required')
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Scores must be finite probabilities between zero and one')
    predicted = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average='binary', pos_label=1, zero_division=0)
    return dict(count=len(labels), threshold=0.5, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                bonafide_count=int(tn+fp), spoof_count=int(fn+tp),
                majority_baseline_accuracy_percent=100*max(tn+fp, fn+tp)/len(labels),
                accuracy_percent=100*accuracy_score(labels, predicted),
                balanced_accuracy_percent=100*balanced_accuracy_score(labels, predicted),
                precision_spoof_percent=100*precision, recall_spoof_percent=100*recall,
                f1_spoof_percent=100*f1, eer_percent=100*eer(labels, scores), auc=auc(labels, scores))


def save_figure(fig, folder: Path, name: str) -> None:
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(folder/f'{name}.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def summarize(root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figures = output/'figures'
    figures.mkdir(exist_ok=True)
    method = dict(scope='pilot_subset', positive_class='spoof=1', threshold=0.5,
        threshold_policy='Fixed conventional threshold; not fitted or chosen using test outcomes.',
        source='Previously saved final predictions; no training or audio rescoring.',
        variability='Sample standard deviation across seeds 0,1,2; not a confidence interval.')
    method_path = output/'summary_method.json'
    if method_path.exists() and json.loads(method_path.read_text()) != method:
        raise ValueError('Summary method changed; use another output directory')
    method_path.write_text(json.dumps(method, indent=2)+'\n')
    rows, predictions = [], {}
    for path in sorted((root/'logs/final').glob('*.json')):
        if path.name.endswith(('.started.json', '.predictions.json')):
            continue
        record = json.loads(path.read_text())
        if 'eer_percent' not in record:
            continue
        saved_path = path.with_suffix('.predictions.json')
        saved = json.loads(saved_path.read_text())
        metrics = fixed_threshold_metrics(saved['labels'], saved['scores'])
        if not np.isclose(metrics['eer_percent'], record['eer_percent'], rtol=0, atol=1e-10):
            raise ValueError(f'Saved EER does not match predictions: {path}')
        if not np.isclose(metrics['auc'], record['auc'], rtol=0, atol=1e-10):
            raise ValueError(f'Saved AUC does not match predictions: {path}')
        key = (record['variant'], record['identity']['dataset'], record['seed'])
        if key in predictions:
            raise ValueError(f'Duplicate measurement: {key}')
        predictions[key] = saved
        rows.append(dict(variant=key[0], dataset=key[1], seed=key[2], **metrics,
                         protocol='pilot_subset', source=path.relative_to(root).as_posix(),
                         prediction_source=saved_path.relative_to(root).as_posix(),
                         checkpoint_sha256=record['identity']['checkpoint_sha256']))
    expected = {(v, d, s) for v in VARIANTS for d in DATASETS for s in (0, 1, 2)}
    if set(predictions) != expected:
        raise ValueError('All 27 final variant/dataset/seed measurements are required')
    runs = pd.DataFrame(rows)
    runs.to_csv(output/'metrics_by_run.csv', index=False)
    summaries = []
    for variant in VARIANTS:
        for dataset in DATASETS:
            group = runs[(runs.variant == variant) & (runs.dataset == dataset)]
            row = dict(variant=variant, dataset=dataset, seeds=3, count_per_seed=int(group['count'].iloc[0]),
                       majority_baseline_accuracy_percent=float(group.majority_baseline_accuracy_percent.iloc[0]),
                       threshold=0.5, protocol='pilot_subset')
            for metric in METRICS:
                row[metric+'_mean'] = float(group[metric].mean())
                row[metric+'_std'] = float(group[metric].std(ddof=1))
            summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output/'metrics_summary.csv', index=False)

    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for index, variant in enumerate(VARIANTS):
        group = summary[summary.variant == variant].set_index('dataset').loc[list(DATASETS)]
        ax.bar(x+(index-1)*.24, group.eer_percent_mean, .24,
               yerr=group.eer_percent_std, capsize=3, label=variant)
    ax.axhline(50, linestyle='--', color='gray', linewidth=1, label='50% reference')
    ax.set(xticks=x, xticklabels=DATASETS, ylabel='Centre-crop EER (%)', ylim=(0, 100),
           title='Compact pilot: mean and sample SD over 3 seeds; 1,000 items per dataset')
    ax.legend(fontsize=8)
    save_figure(fig, figures, 'eer_comparison')

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, dataset in zip(axes, DATASETS):
        for seed in (0, 1, 2):
            saved = predictions['full', dataset, seed]
            fpr, tpr, _ = roc_curve(saved['labels'], saved['scores'])
            ax.plot(fpr, tpr, label=f"seed {seed}; AUC={auc(saved['labels'], saved['scores']):.3f}")
        ax.plot([0, 1], [0, 1], '--', color='gray')
        ax.set(title=dataset, xlabel='False positive rate', ylabel='True positive rate', xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8)
    save_figure(fig, figures, 'full_model_roc')

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, dataset in zip(axes, DATASETS):
        saved = predictions['full', dataset, 0]
        cm = confusion_matrix(saved['labels'], np.asarray(saved['scores']) >= .5, labels=[0, 1])
        normalized = cm/cm.sum(axis=1, keepdims=True)
        ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i,j]}\n{100*normalized[i,j]:.1f}%', ha='center', va='center',
                        color='white' if normalized[i,j] > .5 else 'black')
        ax.set(title=dataset, xticks=[0,1], yticks=[0,1], xticklabels=['Bona fide','Spoof'],
               yticklabels=['Bona fide','Spoof'], xlabel='Predicted', ylabel='Actual')
    fig.suptitle('Full model, illustrative seed 0; fixed threshold 0.5; row-normalized colours')
    save_figure(fig, figures, 'confusion_matrices_seed0')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for seed in (0,1,2):
        history = json.loads((root/f'checkpoints/full_seed{seed}/train.json').read_text())['history']
        epochs = [r['epoch']+1 for r in history]
        axes[0].plot(epochs, [r['loss'] for r in history], marker='o', label=f'seed {seed}')
        values = [100*r['dev_eer'] for r in history]
        axes[1].plot(epochs, values, marker='o', label=f'seed {seed}')
        best = int(np.argmin(values))
        axes[1].scatter(epochs[best], values[best], marker='*', s=160, edgecolors='black', zorder=5)
    axes[0].set(title='Training loss (BCE + stability)', xlabel='Epoch', ylabel='Loss')
    axes[1].set(title='Dev EER; star = selected best epoch', xlabel='Epoch', ylabel='Subset-dev EER (%)')
    for ax in axes:
        ax.legend()
    save_figure(fig, figures, 'training_curves')

    stability = pd.read_csv(root/'tables/stability.csv')
    stability = stability[np.isclose(stability.sigma, .1)]
    fig, ax = plt.subplots(figsize=(9, 4))
    for index, variant in enumerate(VARIANTS):
        values = [stability[(stability.variant==variant)&(stability.dataset==d)].mad for d in DATASETS]
        ax.bar(x+(index-1)*.24, [v.mean() for v in values], .24,
               yerr=[v.std(ddof=1) for v in values], capsize=3, label=variant)
    ax.set(xticks=x, xticklabels=DATASETS, ylabel='Mean absolute CAM difference (lower is better)',
           title='Noise sigma=0.1; same 30 sampled items per dataset, 3 seeds')
    ax.legend(fontsize=8)
    save_figure(fig, figures, 'cam_stability')
    for original, name in [('LA19_eval/example_0.png', 'example_spoof_gradcam.png'),
                           ('LA19_eval/attention_noise.png', 'attention_clean_noisy.png')]:
        if (root/'figures'/original).exists():
            shutil.copy2(root/'figures'/original, figures/name)

    lines = ['# خلاصهٔ خروجی‌های واقعی pilot', '',
        'این خروجی‌ها از پیش‌بینی‌های ذخیره‌شده ساخته شده‌اند؛ آموزش و امتیازدهی مجدد صوت انجام نشده است.', '',
        '## Accuracy با EER فرق دارد', '',
        'Accuracy نسبت پاسخ‌های درست در آستانهٔ ثابت ۰٫۵ است. Balanced Accuracy میانگین نرخ تشخیص درست دو کلاس است. '
        'EER نقطهٔ برابرشدن دو نوع خطا روی ROC است و Accuracy برابر با «۱۰۰ منهای EER» نیست. '
        'آستانهٔ ۰٫۵ روی test بهینه نشده و کالیبره‌بودن احتمال‌ها نیز اثبات نشده است.', '',
        'اعداد جدول زیر میانگین ± انحراف معیار نمونه‌ای سه seed هستند؛ فاصلهٔ اطمینان نیستند.', '',
        '| مدل | داده | EER % ↓ | AUC ↑ | Accuracy % | Balanced Accuracy % | F1 spoof % |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for row in summaries:
        values = [f"{row[m+'_mean']:.2f} ± {row[m+'_std']:.2f}" for m in METRICS if m in
                  ('eer_percent','auc','accuracy_percent','balanced_accuracy_percent','f1_spoof_percent')]
        lines.append('| '+ ' | '.join([row['variant'], row['dataset'], *values])+' |')
    lines.extend(['', '## اثر عدم توازن کلاس‌ها', '',
        '| داده | Accuracy پیش‌بینی همیشگی کلاس اکثریت % |', '| --- | --- |'])
    for dataset in DATASETS:
        value = summary[(summary.variant=='full') & (summary.dataset==dataset)].majority_baseline_accuracy_percent.iloc[0]
        lines.append(f'| {dataset} | {value:.2f} |')
    lines.extend(['', 'Accuracy بالا به‌تنهایی کافی نیست؛ آن را همراه Balanced Accuracy، AUC، EER و ماتریس آشفتگی می‌خوانم. '
        'عملکرد بین‌دامنه‌ای این pilot ضعیف است؛ این آزمایش کوچک برتری قطعی SEGM یا تعمیم قابل اتکا را ثابت نمی‌کند.', '',
        '## فایل‌ها', '',
        '- `metrics_by_run.csv`: اعداد هر ۲۷ اجرا، آستانه، شمارش TP/TN/FP/FN و مسیر منبع.',
        '- `metrics_summary.csv`: میانگین و انحراف معیار سه seed برای هر مدل و دیتاست.',
        '- `summary_method.json`: روش محاسبه و آستانهٔ ثابت.',
        '- `figures/`: نمودارهای EER، ROC، ماتریس آشفتگی، روند آموزش و پایداری CAM؛ نسخهٔ PDF و PNG.', '',
        'ماتریس‌های آشفتگی فقط seed صفر را به‌عنوان مثال از پیش تعیین‌شده نشان می‌دهند؛ بهترین seed آزمون انتخاب نشده است. '
        'نمونه‌های تکرارشده میان seedها مشاهدهٔ مستقل جدید نیستند. تمام نتایج مربوط به `pilot_subset` هستند.', ''])
    (output/'README_FA.md').write_text('\n'.join(lines))
    print(summary[summary.variant=='full'].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path('results/quick-pilot'))
    parser.add_argument('--output', type=Path, default=Path('results/pilot-summary'))
    args = parser.parse_args()
    summarize(args.results, args.output)
