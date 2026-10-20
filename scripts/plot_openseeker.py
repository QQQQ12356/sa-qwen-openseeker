#!/usr/bin/env python
"""Optional histogram from per-row measurements; requires matplotlib and numpy."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurements',type=Path,required=True)
    p.add_argument('--model',default='Qwen/Qwen3.5-4B')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    with args.measurements.open() as f:
        totals=[json.loads(line)['lengths'][args.model]['total'] for line in f]
    fig,ax=plt.subplots(figsize=(8,4))
    ax.hist(totals,bins=np.geomspace(min(totals),max(totals),60),color='#2563eb')
    ax.set_xscale('log');ax.set_xlabel(f'Full trajectory tokens ({args.model})');ax.set_ylabel('Trajectories')
    for n,color in [(16384,'#e11d48'),(32768,'#059669')]:
        ax.axvline(n,color=color,label=f'{n:,} tokens')
    ax.legend();fig.tight_layout();args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output)


if __name__=='__main__':main()
