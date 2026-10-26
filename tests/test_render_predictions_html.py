"""Standalone reports must preserve records without allowing HTML injection."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'render_predictions_html.py'
spec = importlib.util.spec_from_file_location('render_predictions_html', SCRIPT)
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


def test_report_preserves_records_and_escapes_script_termination(tmp_path):
    source = tmp_path / 'results.jsonl'
    row = {'question': '中文 </script><script>alert(1)</script> &',
           'messages': [{'role': 'assistant', 'content': '<think>思考</think>'}],
           'exact_match': False, 'f1': 0.25, 'custom_metric': {'value': 9}}
    source.write_text('\ufeff' + json.dumps(row, ensure_ascii=False) + '\n\n', encoding='utf-8')
    destination = tmp_path / 'report' / 'results.html'
    assert renderer.render_report(source, destination) == 1
    markup = destination.read_text(encoding='utf-8')
    data = markup.split('<script id="report-data" type="application/json">', 1)[1].split('</script>', 1)[0]
    assert '<' not in data and '&' not in data
    assert json.loads(data)['rows'] == [row]
    assert '__REPORT_DATA__' not in markup


@pytest.mark.parametrize('content', ['{}\ninvalid\n', '{}\n[]\n'])
def test_bad_rows_include_source_line_number(tmp_path, content):
    source = tmp_path / 'invalid.jsonl'
    source.write_text(content)
    with pytest.raises(ValueError, match=r'invalid.jsonl:2:'):
        renderer.read_rows(source)


def test_empty_report_and_template_validation(tmp_path):
    source = tmp_path / 'empty.jsonl'
    source.write_text('\n')
    assert renderer.render_report(source, tmp_path / 'empty.html') == 0
    template = tmp_path / 'bad.html'
    template.write_text('<html></html>')
    with pytest.raises(ValueError, match='exactly one'):
        renderer.render_report(source, tmp_path / 'bad-report.html', template)


def test_client_role_boundaries_and_interactions(tmp_path):
    import shutil
    import subprocess
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js is required for offline client execution checks')
    source = tmp_path / 'roles.jsonl'
    rows = [dict(id='first', answer='Alice', prediction='Alice', status='completed',
                 exact_match=False, f1=0.1, messages=[
        dict(role='system', content='Tools: <tool_calls_begin><tool_call>{"name":"example"}</tool_call></tool_calls_end>\nFinal format: <final_answer>Beijing</final_answer>'),
        dict(role='user', content='Who?'),
        dict(role='assistant', content='<think>Quoted <final_answer>example inside thought</final_answer></think>\n<tool_calls_begin><tool_call>{"name":"search","arguments":{"query":["Alice"]}}</tool_call><tool_call>{"name":"visit","arguments":{"url":"https://example.org"}}</tool_call></tool_calls_end><answer>Alice explained</answer><final_answer>Alice</final_answer>'),
        dict(role='user', content='<tool_response>Evidence: <answer>literal webpage content</answer></tool_response>')]),
        dict(id='second', question='Second question', status='completed', exact_match=True, f1=1.0)]
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    destination = tmp_path / 'roles.html'
    renderer.render_report(source, destination)
    harness = Path(__file__).with_name('html_report_client_test.js')
    result = subprocess.run([node, str(harness), str(destination)], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
