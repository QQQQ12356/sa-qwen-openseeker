#!/usr/bin/env python
"""Download pinned HF snapshots, retain originals and export every split as JSONL."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', choices=['polar','zephyr','all'], default='all')
    p.add_argument('--endpoint', default=os.getenv('HF_ENDPOINT','https://huggingface.co'))
    p.add_argument('--revision', default='main')
    p.add_argument('--output', type=Path, default=Path('outputs/data/raw'))
    args = p.parse_args()
    os.environ['HF_ENDPOINT'] = args.endpoint
    from huggingface_hub import HfApi, snapshot_download
    repos = dict(polar='PolarSeeker/OpenSeeker-v1-Data', zephyr='Zephyr271828/openseeker_v1_sft')
    for source, repo in repos.items():
        if args.source not in ('all', source):
            continue
        info = HfApi(endpoint=args.endpoint).dataset_info(repo, revision=args.revision)
        directory = args.output / source / info.sha
        snapshot_download(repo, repo_type='dataset', revision=info.sha, endpoint=args.endpoint,
                          local_dir=str(directory), allow_patterns=['*.jsonl','*.parquet','README.md'])
        files = []
        for path in sorted(directory.rglob('*')):
            if path.suffix not in ('.jsonl','.parquet'):
                continue
            h=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
            files.append(dict(path=str(path),sha256=h.hexdigest(),bytes=path.stat().st_size))
            if path.suffix == '.parquet':
                import pyarrow.parquet as pq
                target = path.with_suffix('.jsonl')
                with target.with_suffix('.jsonl.tmp').open('w') as out:
                    for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
                        for row in batch.to_pylist():
                            out.write(json.dumps(row,ensure_ascii=False)+'\n')
                target.with_suffix('.jsonl.tmp').replace(target)
        (directory/'download_manifest.json').write_text(json.dumps(dict(repo=repo,revision=info.sha,
            endpoint=args.endpoint,files=files),indent=2)+'\n')
        print(directory)


if __name__=='__main__': main()
