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
