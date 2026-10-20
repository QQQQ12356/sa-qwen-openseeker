#!/usr/bin/env python
"""Stream raw JSONL/Parquet into normalized, grouped, length-filtered JSONL.
No GPU or model weights are loaded. Run from the project root.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from search_comp.data.openseeker import normalize, encode, parse_calls, VERSION

MODELS = ['Qwen/Qwen3-4B-Thinking-2507', 'Qwen/Qwen3.5-4B']


def rows(path):
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            yield from batch.to_pylist()
    else:
        with path.open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def stats(values):
    import numpy as np
    if not values:
        return {'n': 0}
    x = np.asarray(values)
    return dict(n=len(x), min=int(x.min()), max=int(x.max()), mean=float(x.mean()),
                **{f'p{p}': float(np.percentile(x, p)) for p in (50, 90, 95, 99)},
                **{f'over_{n}': int((x > n).sum()) for n in (8192, 16384, 32768, 65536)})


def initialize(models, offline):
    global TOKENIZERS
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from transformers import AutoTokenizer
    TOKENIZERS = {m: AutoTokenizer.from_pretrained(m, local_files_only=offline) for m in models}


def measure(item):
    source, index, raw = item
    try:
        row = normalize(raw, source, index)
        lengths = {}
        for name, tok in TOKENIZERS.items():
            encoded = encode(row['messages'], tok)
            lengths[name] = dict(total=len(encoded['input_ids']),
                                 observations=sum(b-a for a,b in encoded['regions']),
                                 assistant=sum(v != -100 for v in encoded['labels']))
        row['lengths'] = lengths
        calls = [c for m in row['messages'] if m['role'] == 'assistant' for c in parse_calls(m['content'])]
        facts = dict(roles=Counter(m['role'] for m in row['messages']),
                     tools=Counter(c['name'] for c in calls),
                     calls=len(calls), messages=len(row['messages']),
                     cjk=any('\u4e00' <= c <= '\u9fff' for c in row['question']),
                     correctness=row['correctness'] or 'unspecified')
        return row, facts, None
    except (ValueError, TypeError, KeyError) as e:
        return None, None, dict(source=source, index=index, reason=str(e))


def dump(f, obj):
    f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, action='append', required=True)
    p.add_argument('--source', required=True, choices=['polar', 'zephyr', 'combined'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--models', nargs='+', default=MODELS)
    p.add_argument('--offline', action='store_true')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--max-length', type=int, default=16384)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val-fraction', type=float, default=.02)
    p.add_argument('--test-fraction', type=float, default=.02)
    p.add_argument('--require-correct', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    if not 0 <= args.val_fraction + args.test_fraction < 1 or min(args.val_fraction, args.test_fraction) < 0:
        p.error('invalid split fractions')
    # Validate tokenizer availability before creating output artifacts.
    initialize(args.models, args.offline)
    args.output.mkdir(parents=True, exist_ok=False)
    counters = Counter(); roles = Counter(); tools = Counter(); correctness = Counter()
    metrics = defaultdict(list); seen = set(); known_splits = {}
    # Preserve published validation questions, even if duplicated in train.
    for path in args.input:
        if path.stem in ('val', 'validation', 'test'):
            for i, raw in enumerate(rows(path)):
                try:
                    key = normalize(raw, args.source, i)['question_id']
                    known_splits[key] = 'test' if path.stem == 'test' else 'validation'
                except (ValueError, KeyError, TypeError):
                    pass
    def items():
        n = 0
        for path in args.input:
            for i, raw in enumerate(rows(path)):
                if args.limit is not None and n >= args.limit:
                    return
                yield f'{args.source}/{path.name}', i, raw
                n += 1
    handles = {n: (args.output / f'{n}.jsonl').open('w') for n in
               ('normalized', 'train', 'validation', 'test', 'rejected', 'measurements')}
    try:
        with ProcessPoolExecutor(args.workers, initializer=initialize,
                                 initargs=(args.models, args.offline)) as pool:
            # Bounded chunks keep memory stable for multi-GB input.
            from itertools import islice
            stream = iter(items())
            while chunk := list(islice(stream, args.workers * 8)):
                for row, facts, error in pool.map(measure, chunk, chunksize=2):
                    counters['input'] += 1
                    if error:
                        counters['invalid'] += 1; dump(handles['rejected'], error); continue
                    dump(handles['normalized'], row)
                    roles.update(facts['roles']); tools.update(facts['tools'])
                    correctness[facts['correctness']] += 1
                    counters['questions_with_cjk'] += facts['cjk']
                    for k in ('calls', 'messages'):
                        metrics[k].append(facts[k])
                    for model, lengths in row['lengths'].items():
                        for kind, count in lengths.items():
                            metrics[f'{model}/{kind}'].append(count)
                    key = row['question_id']
                    reason = None
                    if any(v['total'] > args.max_length for v in row['lengths'].values()):
                        reason = 'over_length'
                    elif args.require_correct and row['correctness'] not in (None, 'Correct'):
                        reason = 'incorrect'
                    elif key in seen:
                        reason = 'duplicate_question'
                    if reason:
                        counters[reason] += 1
                        dump(handles['rejected'], dict(id=row['id'], reason=reason, lengths=row['lengths']))
                    else:
                        seen.add(key)
                        u = int(hashlib.sha256(f'{args.seed}:{key}'.encode()).hexdigest()[:12], 16) / 16**12
                        split = known_splits.get(key) or ('test' if u < args.test_fraction else
                            'validation' if u < args.test_fraction + args.val_fraction else 'train')
                        counters[split] += 1; dump(handles[split], row)
                    dump(handles['measurements'], dict(id=row['id'], question_id=key,
                         lengths=row['lengths'], tools=facts['tools'], calls=facts['calls'], rejection=reason))
                if counters['input'] % 256 == 0:
                    print(dict(counters), flush=True)
    finally:
        for f in handles.values():
            f.close()
    manifest = dict(protocol=VERSION, arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    counts=dict(counters), roles=dict(roles), tools=dict(tools), correctness=dict(correctness),
                    distributions={k:stats(v) for k,v in metrics.items()}, inputs=[])
    manifest['arguments']['input'] = [str(p.resolve()) for p in args.input]
    for path in args.input:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024), b''):
                h.update(block)
        manifest['inputs'].append(dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=h.hexdigest()))
    manifest['tokenizers'] = {m:dict(name=t.name_or_path, vocab_size=len(t),
        backend_sha256=hashlib.sha256(t.backend_tokenizer.to_str().encode()).hexdigest()) for m,t in TOKENIZERS.items()}
    (args.output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
