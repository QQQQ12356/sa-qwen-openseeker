#!/usr/bin/env python
"""Evaluate a plain or merged Beacon checkpoint using live tools or exact replay."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from search_comp.data.openseeker import normalize
from search_comp.evaluation.tool_agent import HTTPTools, ReplayTools, CorpusTools, ModelGenerator, run_agent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--kind', choices=['plain','beacon'], default='beacon')
    p.add_argument('--backend', choices=['replay','http','corpus'], default='replay')
    p.add_argument('--endpoint')
    p.add_argument('--corpus',type=Path)
    p.add_argument('--limit', type=int, default=100)
    p.add_argument('--max-context', type=int, default=16384)
    p.add_argument('--max-new-tokens', type=int, default=8192)
    p.add_argument('--max-tool-tokens', type=int, default=16384)
    p.add_argument('--max-turns', type=int, default=32)
    args=p.parse_args()
    if args.backend=='http' and not args.endpoint:p.error('--endpoint required for http')
    if args.backend=='corpus' and not args.corpus:p.error('--corpus required for corpus backend')
    if args.output.exists():p.error('output exists; use a new evaluation name')
    from search_comp.models.model_loader import load_beacon_model, load_tokenizer
    from search_comp.models.plain_qwen3 import load_sft_model
    tok=load_tokenizer(args.model)
    model=(load_beacon_model if args.kind=='beacon' else load_sft_model)(args.model)
    model.eval()
    generate=ModelGenerator(model,tok,args.kind=='beacon')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    from collections import Counter
    statuses=Counter(); exact=0; count=0
    shared_backend = (HTTPTools(args.endpoint) if args.backend=='http' else
                      CorpusTools(args.corpus) if args.backend=='corpus' else None)
    start=time.time()
    with args.data.open() as source, args.output.open('x') as out:
        for i,line in enumerate(source):
            if i>=args.limit:break
            row=normalize(json.loads(line),index=i)
            backend=ReplayTools(row['messages']) if args.backend=='replay' else shared_backend
            t=time.time()
            result=run_agent(row['messages'],tok,generate,backend,args.max_turns,args.max_context,
                             args.max_new_tokens,args.max_tool_tokens)
            # Conservative text EM, includes Chinese; no unsupported LLM-judge claims.
            correct=result['status']=='completed' and bool(str(row['answer']).strip()) and (
                ' '.join(result['prediction'].casefold().split()) == ' '.join(str(row['answer']).casefold().split()))
            result.update(id=row['id'],answer=row['answer'],exact_match=correct,seconds=time.time()-t)
            out.write(json.dumps(result,ensure_ascii=False)+'\n');out.flush()
            statuses[result['status']]+=1;exact+=correct;count+=1
    summary=dict(mode=args.backend,interpretation='exact-action replay diagnostic' if args.backend=='replay'
        else 'local corpus tool rollout' if args.backend=='corpus' else 'live tool rollout',count=count,exact_match=exact/count if count else None,
        statuses=dict(statuses),seconds=time.time()-start,
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    args.output.with_suffix('.metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
