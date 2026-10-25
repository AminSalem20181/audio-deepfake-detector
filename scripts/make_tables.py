from pathlib import Path
import csv, json

def main():
    out = Path('results/tables'); out.mkdir(parents=True, exist_ok=True)
    try:
        from sdeseg.models import Sdeseg, standard_conv_twin, count_parameters
        counts=[count_parameters(Sdeseg()), count_parameters(standard_conv_twin())]
    except Exception:
        counts=['N/A','N/A']
    with (out/'table3_parameters.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['model','trainable_parameters','source']); w.writerows([['SDE-Seg',counts[0],'computed by model code'],['standard_conv_twin',counts[1],'computed by model code'],['AASIST-L','N/A','not rerun in this environment']])
    with (out/'table4_eer.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['variant','LA19_eval','In-The-Wild','ML-ITW'])
        for v in ['full','no_segm','segm_no_stab','AASIST-L']:
            w.writerow([v,'N/A','N/A','N/A'])

if __name__ == '__main__': main()
