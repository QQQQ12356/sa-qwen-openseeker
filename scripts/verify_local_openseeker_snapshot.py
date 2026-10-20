#!/usr/bin/env python
"""Compare every JSON object, preserving provenance when only JSON formatting changed."""
import argparse
import hashlib
from itertools import zip_longest
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',type=Path,required=True)
    p.add_argument('--converted',type=Path,required=True)
    p.add_argument('--revision',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();count=0;digest=hashlib.sha256()
    with args.original.open('rb') as a,args.converted.open('rb') as b:
        for x,y in zip_longest(a,b):
            if x is None or y is None or json.loads(x)!=json.loads(y):
                raise ValueError(f'JSON objects differ at row {count}')
            digest.update(x);count+=1
    result=dict(rows=count,semantic_equality=True,upstream_revision=args.revision,
        upstream_sha256=digest.hexdigest(),local_file=str(args.converted),
        note='Compared every parsed JSON object in order; formatting may differ.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
