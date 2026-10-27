"""Dataset rendering preserves source data and observation role boundaries."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/render_dataset_html.py'
spec = importlib.util.spec_from_file_location('render_dataset_html', SCRIPT)
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


def sample():
    return dict(id='sample', question='中文问题 </script><script>alert(1)</script>', answer='答案',
                lengths={'model': {'total': 10, 'observations': 3}}, messages=[
        dict(role='system', content='Example: <tool_response>not evidence</tool_response>'),
        dict(role='user', content='Question'),
        dict(role='assistant', content='<think>reason</think><tool_call>{"name":"search","arguments":{}}</tool_call>'),
        dict(role='user', content='<tool_response>证据 😀 <answer>quoted text</answer></tool_response>'),
        dict(role='assistant', content='<final_answer>答案</final_answer>')])


def payload(destination):
    markup = destination.read_text(encoding='utf-8')
    data = markup.split('<script id="dataset-data" type="application/json">')[1].split('</script>')[0]
    assert '<' not in data and '&' not in data
    return json.loads(data)


def test_preserves_data_and_excludes_prompt_examples(tmp_path):
    row = sample()
    source, output = tmp_path / 'data.jsonl', tmp_path / 'data.html'
    source.write_text('\ufeff' + json.dumps(row, ensure_ascii=False) + '\n')
    assert renderer.render_report(source, output) == 1
    result = payload(output)['records'][0]
    assert result['raw'] == row
    assert result['observations'] == [dict(message=3, kind='tool_response',
                                          text='证据 😀 <answer>quoted text</answer>')]
    assert result['observation_chars'] == len(result['observations'][0]['text'])


def test_raw_trajectory_and_searchr1(tmp_path):
    row = dict(trajectory=json.dumps([
        dict(role='user', content='Q'), dict(role='assistant', content='<search>q</search>'),
        dict(role='tool', content='raw result'),
        dict(role='user', content=' <information>docs</information> ')]))
    record = renderer.prepare_record(row, tmp_path / 'raw.jsonl', 1)
    assert [o['kind'] for o in record['observations']] == ['raw_tool', 'information']
    assert record['raw'] == row


def test_directory_offset_limit_manifest_and_overwrite(tmp_path):
    for split in ['train', 'validation']:
        (tmp_path / f'{split}.jsonl').write_text('\n' + json.dumps(sample()) + '\n')
    (tmp_path / 'manifest.json').write_text('{"protocol":"test"}')
    destination = tmp_path / 'report.html'
    assert renderer.render_report(tmp_path, destination, split='all', offset=1, limit=1) == 1
    result = payload(destination)
    assert result['manifest']['protocol'] == 'test'
    assert result['records'][0]['source'].endswith('validation.jsonl')
    assert result['records'][0]['line'] == 2
    with pytest.raises(ValueError, match='overwrite'):
        renderer.render_report(tmp_path, tmp_path / 'manifest.json')


def test_bad_data_reports_line_and_empty_is_supported(tmp_path):
    source, destination = tmp_path / 'bad.jsonl', tmp_path / 'report.html'
    source.write_text('\n[]\n')
    with pytest.raises(ValueError, match='bad.jsonl:2:'):
        renderer.render_report(source, destination)
    source.write_text('')
    assert renderer.render_report(source, destination) == 0
    assert payload(destination)['records'] == []


def test_offline_client_interactions(tmp_path):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js unavailable')
    source, destination = tmp_path / 'data.jsonl', tmp_path / 'data.html'
    rows = [sample(), dict(id='no-observation', messages=[dict(role='user', content='Second question')])]
    source.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    renderer.render_report(source, destination)
    result = subprocess.run([node, str(Path(__file__).with_name('dataset_html_client_test.js')),
                             str(destination)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
