#!/usr/bin/env python3
"""Keep prepared JSONL trajectories containing nonempty observation bodies (stdlib only)."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile


RESPONSE_RE = re.compile(r'<tool_response>(.*?)</tool_response>', re.S)
INFORMATION_RE = re.compile(r'\s*<information>(.*?)</information>\s*', re.S)


def observation_count(row, data_format):
    """Mirror OpenSeeker pieces() / Search-R1 message boundaries without tokenizing."""
    messages = row.get('messages')
    if not isinstance(messages, list) or not messages:
        raise ValueError('expected prepared messages list; normalize raw trajectory data first')
    count = 0
    seen_assistant = False
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError('each message must be an object')
        role, content = message.get('role'), message.get('content', '')
        if not isinstance(content, str):
            raise ValueError('message content must be text')
        if role == 'tool':
            raise ValueError('raw tool role is unsupported; normalize to prepared user observations first')
        if data_format == 'openseeker' and role == 'user' and index > 1:
            count += sum(bool(match[1]) for match in RESPONSE_RE.finditer(content))
        elif data_format == 'searchr1' and role == 'user' and seen_assistant:
            match = INFORMATION_RE.fullmatch(content)
            if match is None:
                raise ValueError('Search-R1 user message after assistant must be a full information block')
            count += bool(match[1])
        if role == 'assistant':
            seen_assistant = True
    return count


def filter_dataset(source, output, data_format='openseeker'):
    source, output = Path(source).resolve(), Path(output).resolve()
    if data_format not in ('openseeker', 'searchr1'):
        raise ValueError('unsupported data format')
    if output.exists():
        raise ValueError(f'output already exists (will not overwrite): {output}')
    if source.is_dir():
        sources = [source / f'{split}.jsonl' for split in ('train', 'validation', 'test')
                   if (source / f'{split}.jsonl').is_file()]
        if not sources:
            raise ValueError('directory contains no train/validation/test JSONL files')
        manifest_path = source / 'manifest.json'
    elif source.is_file():
        if source.suffix != '.jsonl' or source.name == 'removed_records.jsonl':
            raise ValueError('input must be a dataset .jsonl file (not removed_records.jsonl)')
        sources = [source]
        manifest_path = None  # A single file must not inherit unrelated split metadata.
    else:
        raise ValueError(f'input not found: {source}')
    parent_manifest = None
    if manifest_path and manifest_path.is_file():
        parent_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}.', dir=output.parent))
    report = dict(source=str(source), format=data_format,
                  created_at=datetime.now(timezone.utc).isoformat(), splits={}, inputs=[],
                  rule='Keep at least one nonempty observation body at training message boundaries; '
                       'no tokenization, normalization, truncation, or split reassignment.',
                  note='Whitespace-only bodies are retained, as they may encode to training tokens.')
    totals = Counter(input=0, kept=0, removed=0)
    try:
        (staging / 'removed').mkdir()
        with (staging / 'removed_records.jsonl').open('w', encoding='utf-8') as audit:
            for path in sources:
                counts = Counter(input=0, kept=0, removed=0, observation_blocks=0)
                digest = hashlib.sha256()
                # Stream original bytes; retained and removed records are not rewritten.
                with path.open('rb') as stream, (staging / path.name).open('wb') as kept, \
                        (staging / 'removed' / path.name).open('wb') as removed:
                    for line_number, line in enumerate(stream, 1):
                        digest.update(line)
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line.decode('utf-8-sig'))
                            if not isinstance(row, dict):
                                raise ValueError('expected JSON object')
                            blocks = observation_count(row, data_format)
                        except (ValueError, TypeError) as exc:
                            raise ValueError(f'{path}:{line_number}: {exc}') from exc
                        counts['input'] += 1
                        if blocks:
                            kept.write(line)
                            counts['kept'] += 1
                            counts['observation_blocks'] += blocks
                        else:
                            removed.write(line)
                            counts['removed'] += 1
                            audit.write(json.dumps(dict(source=path.name, line=line_number,
                                                        id=row.get('id'), reason='no_nonempty_observation'),
                                                   ensure_ascii=False) + '\n')
                report['splits'][path.stem] = dict(counts)
                report['inputs'].append(dict(path=str(path), sha256=digest.hexdigest(), bytes=path.stat().st_size))
                for key in ('input', 'kept', 'removed'):
                    totals[key] += counts[key]
        report['totals'] = dict(totals)
        if parent_manifest is not None:
            # Preserve protocol and tokenizer fingerprints without claiming a
            # protocol upgrade or reusing aggregate statistics from the parent.
            manifest = {key: parent_manifest[key] for key in ('protocol', 'tokenizers', 'arguments')
                        if key in parent_manifest}
            if 'arguments' in manifest:
                manifest['arguments'] = {**manifest['arguments'], 'output': str(output)}
            manifest.update(parent=str(source),
                            counts={'input': totals['input'], 'no_nonempty_observation': totals['removed'],
                                    **{split: stats['kept'] for split, stats in report['splits'].items()}},
                            observation_filter=report,
                            note='Derived observation filter; original protocol, measurements, and splits retained. '
                                 'Parent aggregates are archived in parent_manifest.json.')
            (staging / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
            (staging / 'parent_manifest.json').write_bytes(manifest_path.read_bytes())
        (staging / 'filter_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Prepared dataset directory or one messages JSONL')
    parser.add_argument('-o', '--output', type=Path, required=True, help='New output directory (must not exist)')
    parser.add_argument('--format', choices=['openseeker', 'searchr1'], default='openseeker')
    args = parser.parse_args()
    try:
        report = filter_dataset(args.input, args.output, args.format)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    for split, counts in report['splits'].items():
        print(f"{split}: input={counts['input']}, kept={counts['kept']}, removed={counts['removed']}")
    print(f'Output: {args.output.resolve()}')
    if report['totals']['kept'] == 0:
        print('WARNING: no records remain; do not use this dataset for training.')
    elif any(stats['kept'] == 0 for stats in report['splits'].values()):
        print('WARNING: some splits are empty; check training/evaluation requirements before use.')


if __name__ == '__main__':
    main()
