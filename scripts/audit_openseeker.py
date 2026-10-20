#!/usr/bin/env python
"""Full-source structural audit, including invalid rows; no tokenizer assumptions."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from search_comp.data.openseeker import parse_calls, normalize, question_key
from prepare_openseeker import rows, stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    counts=Counter();roles=Counter();tools=Counter();correctness=Counter();schemas=Counter()
    parameters=defaultdict(Counter);metrics=defaultdict(list);questions=Counter();failures=Counter()
    groups=defaultdict(Counter)
    for path in args.input:
        for raw in rows(path):
            counts['rows']+=1;schemas['|'.join(sorted(raw))]+=1
            msgs=raw.get('trajectory',raw.get('messages',[]))
            if isinstance(msgs,str):msgs=json.loads(msgs)
            question=raw.get('question') or next((m.get('content','') for m in msgs if m['role']=='user'),'')
            key=question_key(question);questions[key]+=1
            language='contains_cjk' if re.search('[\u4e00-\u9fff]',question) else 'no_cjk'
            counts[language]+=1;correctness[str(raw.get('trajectory correctness','unspecified'))]+=1
            metrics['messages'].append(len(msgs));metrics['question_characters'].append(len(question))
            role_chars=Counter();calls=[]
            for msg in msgs:
                role=msg['role'];roles[role]+=1;text=msg.get('content') or ''
                if isinstance(text,str):role_chars[role]+=len(text)
                if role=='assistant':
                    try:
                        if msg.get('tool_calls'):
                            for c in msg['tool_calls']:
                                fn=dict(c.get('function',c));a=fn.get('arguments',{})
                                fn['arguments']=json.loads(a) if isinstance(a,str) else a
                                calls.append(fn)
                        else:calls.extend(parse_calls(text))
                    except (ValueError,TypeError,KeyError):counts['malformed_action_messages']+=1
            metrics['trajectory_characters'].append(sum(role_chars.values()))
            for role,n in role_chars.items():metrics[f'{role}_characters'].append(n)
            metrics['parsed_calls'].append(len(calls))
            for c in calls:
                tools[c['name']]+=1;parameters[c['name']].update(c['arguments'].keys())
            groups[language]['rows']+=1;groups[language]['calls']+=len(calls)
            declared=raw.get('number of tool calls',raw.get('n_tool_calls'))
            if declared is not None:
                if declared!=len(calls):counts['declared_calls_mismatch']+=1
                if declared==sum(m['role']=='assistant' for m in msgs):counts['declared_matches_assistant_messages']+=1
                if declared==len(calls)+1:counts['declared_equals_calls_plus_one']+=1
            try:normalize(raw)
            except (ValueError,KeyError,TypeError) as e:failures[str(e)]+=1
    result=dict(counts=dict(counts),schemas=dict(schemas),roles=dict(roles),tools=dict(tools),
        tool_argument_keys={k:dict(v) for k,v in parameters.items()},correctness=dict(correctness),
        duplicate_question_rows=sum(v-1 for v in questions.values()),unique_questions=len(questions),
        language_heuristic={k:dict(v) for k,v in groups.items()},normalization_failures=dict(failures),
        distributions={k:stats(v) for k,v in metrics.items()},sources=[str(p.resolve()) for p in args.input])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
