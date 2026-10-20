#!/usr/bin/env python
"""Compare quality/language/length strata from a full measured dataset; emit JSON."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from itertools import zip_longest
from prepare_openseeker import stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    groups=defaultdict(lambda:dict(count=0,cjk=0,tools=Counter(),lengths=[]))
    declared=Counter();totals=[]
    with (args.prepared/'normalized.jsonl').open() as f,(args.prepared/'measurements.jsonl').open() as g:
        for a,b in zip_longest(f,g):
            row,measure=json.loads(a),json.loads(b)
            assert row['id']==measure['id']
            total=max(v['total'] for v in row['lengths'].values());totals.append(total)
            cjk=any('\u4e00'<=c<='\u9fff' for c in row['question'])
            turns=sum(m['role']=='assistant' for m in row['messages'])
            declared['equals_assistant_turns']+=row['declared_tool_calls']==turns
            declared['equals_actual_calls']+=row['declared_tool_calls']==measure['calls']
            declared['equals_calls_plus_one']+=row['declared_tool_calls']==measure['calls']+1
            keys=['all',str(row['correctness'])]
            for threshold in (8192,16384,32768,65536):
                if total<=threshold:
                    keys.append(f'all_le_{threshold}')
                    if row['correctness']=='Correct':keys.append(f'correct_le_{threshold}')
            for key in keys:
                entry=groups[key];entry['count']+=1;entry['cjk']+=cjk
                entry['tools'].update(measure['tools']);entry['lengths'].append(total)
    result=dict(declared_field=declared,groups={k:{**v,'lengths':stats(v['lengths'])} for k,v in groups.items()})
    (args.output/'quality_length.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('Saved',args.output)


if __name__=='__main__':main()
