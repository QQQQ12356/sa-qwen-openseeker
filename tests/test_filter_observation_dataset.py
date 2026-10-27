"""Observation filtering follows role boundaries and preserves measured data."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/filter_observation_dataset.py'
spec = importlib.util.spec_from_file_location('filter_observation_dataset', SCRIPT)
filterer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(filterer)


def row(body=None):
    messages = [dict(role='system', content='<tool_response>example</tool_response>'),
                dict(role='user', content='question'),
                dict(role='assistant', content='<tool_response>quoted</tool_response>')]
    if body is not None:
        messages += [dict(role='user', content=f'<tool_response>{body}</tool_response>'),
                     dict(role='assistant', content='answer')]
    return dict(id='example', messages=messages, lengths={'model': {'total': 100}}, answer='answer')


@pytest.mark.parametrize('body,expected', [(None, 0), ('', 0), ('evidence', 1), (' \n', 1)])
def test_openseeker_boundaries_and_empty_body(body, expected):
    assert filterer.observation_count(row(body), 'openseeker') == expected


def test_multiple_blocks_and_searchr1():
    sample = row('one</tool_response><tool_response>two')
    assert filterer.observation_count(sample, 'openseeker') == 2
    sample['messages'][3]['content'] = ' <information>docs</information>\n'
    assert filterer.observation_count(sample, 'searchr1') == 1
    assert filterer.observation_count(sample, 'openseeker') == 0
    sample['messages'][3]['content'] += 'invalid trailing text'
    with pytest.raises(ValueError, match='full information'):
        filterer.observation_count(sample, 'searchr1')


def test_directory_preserves_splits_bytes_manifest_and_audit(tmp_path):
    source, output = tmp_path / 'input', tmp_path / 'filtered'
    source.mkdir()
    keep = (json.dumps(row('证据'), ensure_ascii=False) + '\n').encode()
    remove = (json.dumps(row()) + '\n').encode()
    for split in ('train', 'validation', 'test'):
        (source / f'{split}.jsonl').write_bytes(keep + remove)
    manifest = dict(protocol='openseeker-chatml-v2-final-answer', tokenizers={'model': {'hash': 'abc'}},
                    arguments={'max_length': 6144, 'output': 'old'}, counts={'input': 999},
                    distributions={'obsolete': 123})
    (source / 'manifest.json').write_text(json.dumps(manifest))
    report = filterer.filter_dataset(source, output)
    assert report['totals'] == dict(input=6, kept=3, removed=3)
    for split in ('train', 'validation', 'test'):
        assert (output / f'{split}.jsonl').read_bytes() == keep
        assert (output / 'removed' / f'{split}.jsonl').read_bytes() == remove
        assert (source / f'{split}.jsonl').read_bytes() == keep + remove
    result = json.loads((output / 'manifest.json').read_text())
    assert result['protocol'] == manifest['protocol']
    assert result['tokenizers'] == manifest['tokenizers']
    assert result['arguments']['max_length'] == 6144
    assert result['counts']['train'] == 1
    assert 'distributions' not in result
    assert json.loads((output / 'parent_manifest.json').read_text()) == manifest
    audit = [json.loads(line) for line in (output / 'removed_records.jsonl').read_text().splitlines()]
    assert len(audit) == 3 and all(item['line'] == 2 for item in audit)
    with pytest.raises(ValueError, match='already exists'):
        filterer.filter_dataset(source, output)


def test_invalid_record_leaves_no_partial_dataset(tmp_path):
    source, output = tmp_path / 'train.jsonl', tmp_path / 'filtered'
    source.write_text(json.dumps(row('evidence')) + '\n[]\n')
    with pytest.raises(ValueError, match='train.jsonl:2:'):
        filterer.filter_dataset(source, output)
    assert not output.exists()
    assert not list(tmp_path.glob('.filtered.*'))


def test_raw_tool_requires_normalization(tmp_path):
    sample = row()
    sample['messages'].append(dict(role='tool', content='raw evidence'))
    with pytest.raises(ValueError, match='normalize'):
        filterer.observation_count(sample, 'openseeker')


def test_empty_output_split_is_reported(tmp_path):
    source = tmp_path / 'train.jsonl'
    source.write_text(json.dumps(row()) + '\n')
    report = filterer.filter_dataset(source, tmp_path / 'filtered')
    assert report['totals'] == dict(input=1, kept=0, removed=1)
    assert (tmp_path / 'filtered/train.jsonl').read_bytes() == b''
