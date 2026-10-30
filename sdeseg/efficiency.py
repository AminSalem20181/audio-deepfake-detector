"""Measured batch-one network latency; preprocessing is excluded."""
from __future__ import annotations
import argparse
import platform
import subprocess
import time
import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis
from .models import Sdeseg, count_parameters
from .utils import save_json, versions, seed_everything


def benchmark(output):
    seed_everything(0)
    model = Sdeseg().eval()
    x = torch.randn(1,1,64,400)
    analysis = FlopCountAnalysis(model, x)
    flops = analysis.total()
    unsupported = dict(analysis.unsupported_ops())
    default = torch.get_num_threads()
    rows = []
    try:
        for device, threads in [('cpu',1),('cpu',default)] + ([('cuda',default)] if torch.cuda.is_available() else []):
            torch.set_num_threads(threads)
            model, sample = model.to(device), x.to(device)
            def sync():
                if device == 'cuda':
                    torch.cuda.synchronize()
            with torch.no_grad():
                for _ in range(20): model(sample)
                sync()
                times = []
                for _ in range(200):
                    sync(); start=time.perf_counter(); model(sample); sync()
                    times.append((time.perf_counter()-start)*1000)
            rows.append(dict(device=device, threads=threads, median_ms=float(np.median(times)),
                             iqr_ms=float(np.percentile(times,75)-np.percentile(times,25)), samples_ms=times))
    finally:
        torch.set_num_threads(default)
    cpu = platform.processor()
    cpu_query_error = None
    if platform.system() == 'Darwin':
        query = subprocess.run(['sysctl','-n','machdep.cpu.brand_string'], text=True, capture_output=True)
        if query.returncode == 0:
            cpu = query.stdout.strip()
        else:
            cpu = None
            cpu_query_error = query.stderr.strip()
    elif platform.system() == 'Linux':
        from pathlib import Path
        cpu = next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), cpu)
    result = dict(parameters=count_parameters(model), flops=flops,
                  flops_tool='fvcore; one multiply-add counts as one FLOP; unsupported operators excluded',
                  unsupported_ops=unsupported, cpu=cpu, cpu_arch=platform.machine(), cpu_query_error=cpu_query_error,
                  platform=platform.platform(), versions=versions(),
                  gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                  input='synthetic (1,1,64,400); untrained network; inference timing only', latency=rows)
    save_json(output, result)
    return result


if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='results/logs/efficiency.json')
    benchmark(ap.parse_args().output)
