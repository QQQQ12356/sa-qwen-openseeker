#!/usr/bin/env python3
"""Render prediction JSONL files as standalone, offline HTML reports (stdlib only)."""
import argparse
import json
from pathlib import Path


DEFAULT_TEMPLATE = Path(__file__).resolve().parent / 'templates' / 'predictions.html'


def read_rows(path):
    rows = []
    with path.open(encoding='utf-8-sig') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{line_number}: invalid JSON: {exc.msg}') from exc
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{line_number}: expected a JSON object')
            rows.append(row)
    return rows


def render_report(source, destination, template=DEFAULT_TEMPLATE, title=None):
    rows = read_rows(source)
    payload = json.dumps(
        {'title': title or source.name, 'source': str(source), 'rows': rows},
        ensure_ascii=False, allow_nan=False,
    )
    # A JSON script element is still parsed by HTML: escape closing tags and entities.
    payload = payload.replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    markup = template.read_text(encoding='utf-8')
    marker = '__REPORT_DATA__'
    if markup.count(marker) != 1:
        raise ValueError(f'{template}: template must contain exactly one {marker} marker')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(markup.replace(marker, payload), encoding='utf-8')
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path, nargs='+', help='JSONL files (shell glob supported)')
    output = parser.add_mutually_exclusive_group()
    output.add_argument('-o', '--output', type=Path, help='HTML path for one input')
    output.add_argument('--output-dir', type=Path, help='Directory for batch reports')
    parser.add_argument('--template', type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument('--title', help='Report title (defaults to source filename)')
    args = parser.parse_args()
    if args.output and len(args.inputs) != 1:
        parser.error('--output requires exactly one input; use --output-dir for batches')
    targets = [args.output or ((args.output_dir / p.with_suffix('.html').name)
               if args.output_dir else p.with_suffix('.html')) for p in args.inputs]
    resolved = [p.resolve() for p in targets]
    if len(set(resolved)) != len(resolved):
        parser.error('output filenames collide; use distinct basenames or render separately')
    protected = {p.resolve() for p in args.inputs} | {args.template.resolve()}
    if protected.intersection(resolved):
        parser.error('output must not overwrite an input or template')
    try:
        for source, destination in zip(args.inputs, targets):
            count = render_report(source, destination, args.template, args.title)
            print(f'{source} -> {destination} ({count} records)')
    except (OSError, ValueError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
