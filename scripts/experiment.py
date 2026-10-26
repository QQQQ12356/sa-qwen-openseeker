#!/usr/bin/env python
"""One run identity for resolved config, training, checkpoint, evaluation and logs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from search_comp.data.openseeker import VERSION


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['plan','train','eval'])
    p.add_argument('--config',type=Path,default=ROOT/'configs/openseeker.yaml')
    p.add_argument('--run-dir',type=Path)
    args=p.parse_args()
    if args.run_dir:
        run=args.run_dir.resolve()
        spec=json.loads((run/'experiment.json').read_text())
    else:
        if args.action=='eval':p.error('eval requires --run-dir from its training run')
        spec=yaml.safe_load(args.config.read_text())
        family=spec['model_family']; kind=spec['kind']; data=spec['dataset']
        cfg=yaml.safe_load((ROOT/spec['base_configs'][family][kind]).read_text())
        cfg.update(spec.get('training',{}))
        cfg.update(model_name_or_path=spec['models'][family],data_mode=spec['data_mode'],
                   train_data_path=str((ROOT/data/'train.jsonl').resolve()),
                   val_data_path=str((ROOT/data/'validation.jsonl').resolve()))
        cfg.setdefault('beacon',{}).update(spec.get('beacon',{}))
        if spec['data_mode']=='openseeker':
            manifest=json.loads((ROOT/data/'manifest.json').read_text())
            if manifest.get('protocol') != VERSION:
                p.error('dataset uses an old protocol; rerun prepare_openseeker.py for final_answer data')
            if cfg['model_name_or_path'] not in manifest['tokenizers']:
                p.error('dataset was not measured with this model tokenizer; prepare with --models first')
            if manifest['arguments']['max_length']>cfg['max_length']:
                p.error('prepared dataset length cap exceeds training max_length')
            spec['dataset_manifest']=manifest

        fingerprints={}
        for key in ('train_data_path','val_data_path'):
            path=Path(cfg[key])
            if not path.is_file() or not path.stat().st_size:
                p.error(f'missing/empty dataset: {path}; run prepare_openseeker.py first')
            fingerprints[key]=digest(path)
        cfg['test_data_path']=str((ROOT/data/'test.jsonl').resolve())
        fingerprints['test_data_path']=digest(cfg['test_data_path'])
        spec['data_fingerprints']=fingerprints
        spec['code_fingerprints']={str(p.relative_to(ROOT)):digest(p) for folder in ('search_comp','scripts')
            for p in sorted((ROOT/folder).rglob('*.py'))}
        taghash=hashlib.sha256(json.dumps(dict(spec=spec,cfg=cfg),sort_keys=True).encode()).hexdigest()[:10]
        tag=f"{family}_{kind}_{spec['data_mode']}_L{cfg['max_length']}_r{cfg['beacon'].get('beacon_ratio',16)}_lr{cfg['learning_rate']}_s{cfg['seed']}_{taghash}"
        name=tag+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        run=(ROOT/'outputs/experiments'/name).resolve();run.mkdir(parents=True)
        cfg.update(output_dir=str(run),exp_name='model')
        spec.update(name=name,training_config=cfg,created_at=datetime.now(timezone.utc).isoformat(),
                    python=sys.version,platform=platform.platform())
        (run/'experiment.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n')
        (run/'train.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
        (run/'source_config.yaml').write_text(args.config.read_text())
    cfg=spec['training_config']
    for path, expected in spec.get('code_fingerprints',{}).items():
        if digest(ROOT/path)!=expected:raise RuntimeError(f'code changed since plan: {path}; create a new run')
    for key, expected in spec['data_fingerprints'].items():
        if digest(cfg[key])!=expected:raise RuntimeError(f'dataset changed: {key}')
    train_cmd=[sys.executable,'-m','search_comp.trainer.'+('beacon_trainer' if spec['kind']=='beacon' else 'plain_sft_trainer'),
               '--config',str(run/'train.yaml')]
    if args.action=='plan':
        print(json.dumps(dict(run_dir=str(run),train_command=train_cmd,
              next=f'{sys.executable} scripts/experiment.py train --run-dir {run}'),indent=2));return
    if args.action=='train':
        if (run/'models/model').exists():
            raise RuntimeError('training output already exists; create a new run to avoid overwrite')
        command=train_cmd
    else:
        summary=json.loads((run/'models/model/train_summary.json').read_text())
        command=[sys.executable,'scripts/evaluate_openseeker.py','--model',summary['final_model_path'],
                 '--kind',spec['kind'],'--data',cfg['test_data_path'],
                 '--output',str(run/'evaluation'/('predictions_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.jsonl'))]
        for key,value in spec.get('evaluation',{}).items():
            if value is not None:command.extend(['--'+key.replace('_','-'),str(value)])
    log=run/(args.action+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.log')
    start=time.time()
    with log.open('x') as out:
        process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in process.stdout:
            print(line,end='');out.write(line);out.flush()
        code=process.wait()
    with (run/'events.jsonl').open('a') as out:
        out.write(json.dumps(dict(action=args.action,command=command,log=str(log),returncode=code,
                                  seconds=time.time()-start))+'\n')
    raise SystemExit(code)


if __name__=='__main__':main()
