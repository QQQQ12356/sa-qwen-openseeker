#!/usr/bin/env python
"""Create a new length-limited dataset from measured normalized JSONL, without retokenizing.
Uses the exact tokenizer fingerprints recorded by prepare_openseeker.py.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-length',type=int,required=True)
    p.add_argument('--require-correct', action=argparse.BooleanOptionalAction, default=True)
    args=p.parse_args()
    manifest=json.loads((args.prepared/'manifest.json').read_text())
    options=dict(manifest['arguments'])
    options['require_correct']=args.require_correct
    # Published split membership must be retained: recover from original input paths.
    from prepare_openseeker import rows
    from search_comp.data.openseeker import normalize
    fixed={}
    for s in options['input']:
        path=Path(s)
        if path.stem in ('val','validation','test'):
            for i,row in enumerate(rows(path)):
                try:fixed[normalize(row,index=i)['question_id']]='test' if path.stem=='test' else 'validation'
                except (ValueError,TypeError,KeyError):pass
    args.output.mkdir(parents=True,exist_ok=False)
    handles={s:(args.output/f'{s}.jsonl').open('w') for s in ('train','validation','test','rejected')}
    counts=Counter();seen=set()
    try:
        with (args.prepared/'normalized.jsonl').open() as source:
            for line in source:
                row=json.loads(line);counts['input']+=1;key=row['question_id'];reason=None
                if any(v['total']>args.max_length for v in row['lengths'].values()):reason='over_length'
                elif options['require_correct'] and row['correctness'] not in (None,'Correct'):reason='incorrect'
                elif key in seen:reason='duplicate_question'
                if reason:
                    counts[reason]+=1
                    handles['rejected'].write(json.dumps(dict(id=row['id'],reason=reason))+'\n')
                    continue
                seen.add(key)
                u=int(hashlib.sha256(f"{options['seed']}:{key}".encode()).hexdigest()[:12],16)/16**12
                split=fixed.get(key) or ('test' if u<options['test_fraction'] else
                      'validation' if u<options['test_fraction']+options['val_fraction'] else 'train')
                handles[split].write(line);counts[split]+=1
    finally:
        for f in handles.values():f.close()
    result=dict(parent=str(args.prepared.resolve()),tokenizers=manifest['tokenizers'],
                arguments={**options,'max_length':args.max_length},counts=dict(counts),
                note='Derived selection; normalized full data and length distributions remain in parent.')
    (args.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
