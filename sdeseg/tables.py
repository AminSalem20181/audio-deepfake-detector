"""Generate paper tables exclusively from measured logs and executed counts."""
from pathlib import Path
import csv
import json
import numpy as np
from .models import Sdeseg, standard_conv_twin, count_parameters
from .utils import save_json, versions


def write_table(path, header, rows):
    with path.open('w', newline='') as f:
        w = csv.writer(f, lineterminator='\n'); w.writerow(header)
        w.writerows([['N/A' if x is None else x for x in row] for row in rows])


def make_tables(root='results'):
    root = Path(root); out = root/'tables'; out.mkdir(parents=True, exist_ok=True)
    profile_path = root/'logs/experiment_profile.json'
    protocol = json.loads(profile_path.read_text())['kind'] if profile_path.exists() else 'full_protocol'
    counts = {'SDE-Seg': count_parameters(Sdeseg()), 'standard_conv_twin': count_parameters(standard_conv_twin())}
    baseline_path = root/'logs/aasist_parameters.json'
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    baseline_count = baseline['trainable_parameters'] if baseline else None
    if baseline_count is not None and (not isinstance(baseline_count, int) or baseline_count <= 0):
        raise ValueError('Invalid measured baseline parameter count')
    reductions = {name: 100*(1-count/counts['standard_conv_twin']) for name,count in counts.items()}
    save_json(root/'logs/parameters.json', dict(counts=counts, reduction_vs_standard_percent=reductions, versions=versions()))
    write_table(out/'table3_parameters.csv', ['model','trainable_parameters','reduction_vs_standard_percent','reduction_vs_aasist_percent','source'],
                [[k,v,reductions[k],100*(1-v/baseline_count) if baseline_count else None,'logs/parameters.json'] for k,v in counts.items()]
                + [['AASIST-L',baseline_count,100*(1-baseline_count/counts['standard_conv_twin']) if baseline_count else None,
                    0 if baseline_count else None,'logs/aasist_parameters.json' if baseline else 'optional baseline not measured']])
    records = []
    for path in sorted((root/'logs/final').glob('*.json')):
        r = json.loads(path.read_text())
        if 'eer_percent' in r and 'identity' in r:
            r['_source'] = str(path.relative_to(root)); records.append(r)
    keys = [(r['variant'], r['seed'], r['identity']['dataset']) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate final measurements for variant/seed/dataset')
    rows, provenance, windowed = [], [], []
    for variant in ('full','no_segm','segm_no_stab','AASIST-L'):
        row, wr = [variant], [variant]
        for dataset in ('LA19_eval','In-The-Wild','ML-ITW'):
            group = [r for r in records if r['variant'] == variant and r['identity']['dataset'] == dataset]
            complete = {r['seed'] for r in group} == {0,1,2}
            values = [r['eer_percent'] for r in group]
            row.append(f'{np.mean(values):.6f} ± {np.std(values, ddof=1):.6f}' if complete else None)
            wv = [r['windowed']['eer_percent'] for r in group if 'windowed' in r]
            wr.append(f'{np.mean(wv):.6f} ± {np.std(wv, ddof=1):.6f}' if complete and len(wv)==3 else None)
            provenance.extend([[variant, dataset, r['seed'], r['eer_percent'], r['auc'], r['_source'], r['identity']['checkpoint_sha256'],r.get('count'),protocol] for r in group])
        rows.append(row+[protocol]); windowed.append(wr+[protocol])
    header = ['variant','LA19_eval','In-The-Wild','ML-ITW','protocol']
    write_table(out/'table4_eer.csv', header, rows)
    write_table(out/'windowed_eer.csv', header, windowed)
    write_table(out/'eer_runs.csv', ['variant','dataset','seed','eer_percent','auc','source','checkpoint_sha256','count','protocol'], provenance)
    stability, deletion = [], []
    for r in records:
        prefix = [r['variant'], r['identity']['dataset'], r['seed']]
        for sigma, metrics in r.get('stability', {}).items():
            stability.append(prefix+[sigma, metrics['count'], metrics['mean_abs_difference'], metrics['spearman'], metrics['valid_correlations'], r['_source']])
        for control, values in r.get('deletion', {}).items():
            for p, value in values.items():
                deletion.append(prefix+[control,p,r['deletion_spoof_count'],value,r['_source']])
    write_table(out/'stability.csv', ['variant','dataset','seed','sigma','count','mad','spearman','valid_correlations','source'], stability)
    write_table(out/'deletion.csv', ['variant','dataset','seed','control','percent','spoof_count','probability_drop','source'], deletion)
    efficiency_path = root/'logs/efficiency.json'
    efficiency = json.loads(efficiency_path.read_text()) if efficiency_path.exists() else {}
    write_table(out/'latency.csv', ['device','threads','median_ms','iqr_ms','source'],
                [[r['device'],r['threads'],r['median_ms'],r['iqr_ms'],'logs/efficiency.json'] for r in efficiency.get('latency',[])])
    write_table(out/'flops.csv', ['model','flops','tool','source'],
                [['SDE-Seg',efficiency.get('flops'),efficiency.get('flops_tool'),'logs/efficiency.json']])
