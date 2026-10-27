#!/usr/bin/env python3
"""Render dataset JSONL files or prepared dataset directories as offline HTML."""
import argparse
import json
import re
from pathlib import Path


DEFAULT_TEMPLATE = Path(__file__).resolve().parent / 'templates' / 'dataset.html'
OBSERVATION = re.compile(r'<(tool_response|information)>(.*?)</\1>', re.S)


def prepare_record(row, source, line_number):
    messages = row.get('messages', row.get('trajectory', []))
    if isinstance(messages, str):
        messages = json.loads(messages)
    if not isinstance(messages, list) or any(not isinstance(m, dict) for m in messages):
        raise ValueError('messages/trajectory must be a list of objects')
    observations = []
    seen_assistant = False
    for index, message in enumerate(messages):
        role = message.get('role')
        content = message.get('content') or ''
        if not isinstance(content, str):
            continue  # Non-text content is preserved in the raw message view.
        if role == 'tool' and content:
            observations.append(dict(message=index, kind='raw_tool', text=content))
        elif role == 'user':
            for match in OBSERVATION.finditer(content):
                # Match the prepared OpenSeeker / Search-R1 role boundaries;
                # do not count protocol examples quoted in system/assistant text.
                eligible = (index > 1 if match[1] == 'tool_response' else
                            seen_assistant and match.group(0).strip() == content.strip())
                if eligible and match[2]:
                    observations.append(dict(message=index, kind=match[1], text=match[2]))
        if role == 'assistant':
            seen_assistant = True
    question = row.get('question') or next(
        (m.get('content', '') for m in messages if m.get('role') == 'user'), '')
    return dict(
        id=row.get('id', f'{source.name}:{line_number}'), source=str(source), line=line_number,
        question=question, messages=messages, observations=observations,
        observation_chars=sum(len(o['text']) for o in observations), raw=row,
    )


def render_report(source, destination, *, split='train', limit=0, offset=0,
                  template=DEFAULT_TEMPLATE, title='数据集轨迹浏览器'):
    source, destination, template = Path(source), Path(destination), Path(template)
    if limit < 0 or offset < 0:
        raise ValueError('limit and offset must be nonnegative')
    if source.is_dir():
        names = ['train', 'validation', 'test'] if split == 'all' else [split]
        sources = [source / f'{name}.jsonl' for name in names]
        if split == 'all':
            sources = [p for p in sources if p.is_file()]
        if not sources:
            raise ValueError(f'{source}: no train/validation/test JSONL files')
    else:
        sources = [source]
    protected = {p.resolve() for p in sources} | {template.resolve()}
    if source.is_dir():
        protected |= {p.resolve() for p in source.iterdir() if p.is_file()}
    if destination.resolve() in protected:
        raise ValueError('output must not overwrite a dataset file or template')
    records, scanned = [], 0
    for path in sources:
        with path.open(encoding='utf-8-sig') as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                if scanned < offset:
                    scanned += 1
                    continue
                if limit and len(records) >= limit:
                    break
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError('expected a JSON object')
                    record = prepare_record(row, path, line_number)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f'{path}:{line_number}: {exc}') from exc
                records.append(record)
        if limit and len(records) >= limit:
            break
    manifest = None
    manifest_path = source / 'manifest.json' if source.is_dir() else source.parent / 'manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    payload = json.dumps(dict(title=title, source=str(source), split=split if source.is_dir() else None,
                              offset=offset, limit=limit, records=records, manifest=manifest),
                         ensure_ascii=False, allow_nan=False)
    payload = payload.replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    markup = template.read_text(encoding='utf-8')
    if markup.count('__DATASET_DATA__') != 1:
        raise ValueError('template must contain exactly one __DATASET_DATA__ marker')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(markup.replace('__DATASET_DATA__', payload), encoding='utf-8')
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='JSONL file or prepared dataset directory')
    parser.add_argument('-o', '--output', type=Path, default=Path('outputs/reports/dataset.html'))
    parser.add_argument('--split', choices=['train', 'validation', 'test', 'rejected', 'all'], default='train')
    parser.add_argument('--limit', type=int, default=0, help='0: include all records')
    parser.add_argument('--offset', type=int, default=0, help='Skip this many nonempty records')
    parser.add_argument('--title', default='数据集轨迹浏览器')
    parser.add_argument('--template', type=Path, default=DEFAULT_TEMPLATE)
    args = parser.parse_args()
    try:
        count = render_report(args.input, args.output, split=args.split, limit=args.limit,
                              offset=args.offset, title=args.title, template=args.template)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(f'{args.input} -> {args.output.resolve()} ({count} records)')


if __name__ == '__main__':
    main()
