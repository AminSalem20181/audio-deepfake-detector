"""Produce a human-readable report from recorded research artifacts."""
import csv
import json
from pathlib import Path


def write_report(root='results'):
    root = Path(root)
    logs = root / 'logs'
    profile_path = logs / 'experiment_profile.json'
    profile = json.loads(profile_path.read_text()) if profile_path.exists() else {}
    pilot = profile.get('kind') == 'pilot_subset'
    selection_path = logs / 'frozen_selection.json'
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    records = []
    for path in sorted((logs / 'final').glob('*.json')):
        record = json.loads(path.read_text())
        if 'eer_percent' in record and 'identity' in record:
            records.append(record)
    expected = {(variant, seed, dataset)
                for variant in ('full', 'no_segm', 'segm_no_stab')
                for seed in (0, 1, 2)
                for dataset in ('LA19_eval', 'In-The-Wild', 'ML-ITW')}
    observed = [(r['variant'], r['seed'], r['identity']['dataset']) for r in records]
    if len(observed) != len(set(observed)):
        raise ValueError('Duplicate final measurements; report cannot identify one experiment')
    complete = set(observed) == expected and bool(selection)
    lines = ['# SDE-Seg compact pilot report' if pilot else '# SDE-Seg measured experiment report', '',
             f"Status: {'all 27 required final evaluations recorded' if complete else 'incomplete experiment'}. "
             f'{len(records)} final measurement records are available.', '',
             '## Model selection', '',
             f"Selected lambda: {selection.get('lambda_value', 'not frozen')}. "
             'Selection uses LA train/dev only. Each run uses its own best-dev checkpoint. '
             'Final test measurements do not select the model.', '']
    if pilot:
        cfg = profile['config']
        lines[2:2] = [
            '**Scope: pilot_subset. These measurements are not full-dataset benchmark results.**', '',
            f"User-requested short run: up to {cfg['train_samples']} training items, "
            f"{cfg['dev_samples']} dev items, {cfg['eval_samples']} test items per dataset, "
            f"{cfg['max_epochs']} epochs per run, patience {cfg['patience']}, "
            f"and {cfg['explain_samples']} explanation items per dataset/checkpoint.", '',
            'Sampling approximately preserves original class proportions and uses fixed seeds. '
            'All variants share the same subsets. Model selection uses subset dev only. '
            'Subset manifests and original/evaluated counts are recorded in '
            '`logs/training_subsets.json` and `logs/evaluation_subsets.json`. '
            'These limits supersede the original full-experiment protocol at the user’s request.', '']

    def table(name, heading):
        path = root / 'tables' / name
        lines.extend([f'## {heading}', ''])
        if not path.exists():
            lines.extend(['N/A: measurement table is missing.', ''])
            return
        with path.open(newline='') as file:
            rows = list(csv.reader(file))
        if not rows:
            lines.extend(['N/A: measurement table is empty.', ''])
            return
        lines.append('| ' + ' | '.join(rows[0]) + ' |')
        lines.append('| ' + ' | '.join('---' for _ in rows[0]) + ' |')
        lines.extend('| ' + ' | '.join(row) + ' |' for row in rows[1:])
        lines.extend(['', f'Source: `tables/{name}`.', ''])

    table('table4_eer.csv', 'Primary centre-crop EER (%)')
    lines.extend(['Mean and sample standard deviation are reported only for all three seeds. '
                  'Incomplete groups remain N/A. Small differences alone do not establish statistical significance.', ''])
    table('windowed_eer.csv', 'Cross-domain window-average EER (%)')
    lines.extend(['Window averaging uses complete non-overlapping four-second windows; '
                  'short recordings use repeat padding. Centre crops remain the primary metric.', ''])
    lines.extend(['## Descriptive ablation findings', ''])
    findings = []
    for dataset in ('LA19_eval', 'In-The-Wild', 'ML-ITW'):
        groups = {variant: [r for r in records if r['variant'] == variant
                            and r['identity']['dataset'] == dataset]
                  for variant in ('full', 'no_segm', 'segm_no_stab')}
        if not all({r['seed'] for r in group} == {0, 1, 2} for group in groups.values()):
            continue
        full_eer = sum(r['eer_percent'] for r in groups['full']) / 3
        for variant in ('no_segm', 'segm_no_stab'):
            other_eer = sum(r['eer_percent'] for r in groups[variant]) / 3
            difference = full_eer - other_eer
            direction = 'higher' if difference > 0 else 'lower' if difference < 0 else 'equal'
            findings.append(f'- {dataset}: full mean EER is {direction} than {variant} '
                            f'by {abs(difference):.6f} percentage points.')
        for sigma in ('0.05', '0.1', '0.2'):
            if not all(sigma in r.get('stability', {}) for group in groups.values() for r in group):
                continue
            full_mad = sum(r['stability'][sigma]['mean_abs_difference'] for r in groups['full']) / 3
            other_mad = sum(r['stability'][sigma]['mean_abs_difference'] for r in groups['segm_no_stab']) / 3
            direction = 'higher' if full_mad > other_mad else 'lower' if full_mad < other_mad else 'equal'
            findings.append(f'- {dataset}, noise sigma={sigma}: full mean CAM difference '
                            f'is {direction} than segm_no_stab ({full_mad:.8f} versus {other_mad:.8f}).')
    lines.extend(findings or ['N/A: complete three-seed comparisons are not yet available.'])
    lines.extend(['', 'Higher EER or CAM difference is a negative result for the corresponding objective. '
                  'These comparisons are descriptive and do not establish statistical significance.', ''])
    table('table3_parameters.csv', 'Executed parameter counts')
    table('latency.csv', 'Measured inference latency')
    table('flops.csv', 'Counted network operations')
    lines.extend(['## Explanations and provenance', '',
                  'Per-seed AUC, dataset counts, class counts, duration and checkpoint SHA256 are in '
                  '`logs/final/*.json` and `tables/eer_runs.csv`. Prediction files retain paths, labels and scores. '
                  'Grad-CAM stability and matched-random deletion measurements are in `tables/stability.csv` '
                  'and `tables/deletion.csv`; qualitative examples are in `figures/`.', '',
                  'Undefined constant-map rank correlations are recorded as null with valid counts retained. '
                  'Optional AASIST-L retraining and official minimum t-DCF remain N/A. '
                  'Efficiency excludes preprocessing; unsupported FLOP operators and hardware are recorded '
                  'in `logs/efficiency.json`.', '',
                  '## Limits', '',
                  'Split checks reject repeated resolved paths; they do not detect independently copied or '
                  're-encoded recordings. ML-ITW source revision, shard checksums and observed label mapping '
                  'are recorded separately before scoring. Results describe these datasets and training seeds; '
                  'they do not establish performance for all unseen generators or languages.', ''])
    target = root / 'FINAL_REPORT.md'
    target.write_text('\n'.join(lines))
    if complete:
        # Preserve the pre-scoring data inspection and put measured findings first.
        report = root / 'REPORT.md'
        inspection_backup = root / 'REPORT_DATA_INSPECTION.md'
        if report.exists() and not inspection_backup.exists():
            inspection_backup.write_text(report.read_text())
        appendix = []
        source_report = Path(__file__).resolve().parents[1] / 'REPORT.md'
        if source_report.exists():
            audit = source_report.read_text()
            start = audit.find('## Method choices and deviations')
            if start >= 0:
                end = audit.find('## Validation', start)
                appendix.append(audit[start:end if end >= 0 else None])
        efficiency_path = logs / 'efficiency.json'
        if efficiency_path.exists():
            efficiency = json.loads(efficiency_path.read_text())
            hardware = {key: efficiency.get(key) for key in ('cpu', 'gpu', 'platform', 'versions')}
            appendix.append('## Executed environment\n\n```json\n'
                            + json.dumps(hardware, indent=2) + '\n```\n')
        if inspection_backup.exists():
            appendix.append('## Recorded data inspection before scoring\n\n' + inspection_backup.read_text())
        report.write_text(target.read_text() + '\n\n' + '\n\n'.join(appendix))
    return target


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='results')
    print(write_report(parser.parse_args().results))
