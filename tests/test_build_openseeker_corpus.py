"""Recorded visit evidence extraction, attribution, artifacts and executable entry points."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

from scripts.build_openseeker_corpus import build_corpus, extract_evidence


def call(url="https://example.org/a"):
    return dict(name="visit", arguments=dict(url=url, goal="GOAL_SECRET"))


def trajectory(calls=None, responses=None):
    calls = calls if calls is not None else [call()]
    responses = responses if responses is not None else [
        "The useful information for GOAL_SECRET as follows:\n\nEvidence in page:\n"
        "Red flowers grow here.\n\nSummary:\nSUMMARY_SECRET"]
    block = "<tool_calls_begin>" + "".join(
        "<tool_call>" + json.dumps(item) + "</tool_call>" for item in calls) + "</tool_calls_end>"
    return dict(answer="REFERENCE_SECRET", trajectory=[
        dict(role="system", content="SYSTEM_SECRET"),
        dict(role="user", content="QUESTION_SECRET"),
        dict(role="assistant", content="<think>REASONING_SECRET</think>" + block),
        *[dict(role="user", content="<tool_response>" + text + "</tool_response>") for text in responses],
        dict(role="assistant", content="ASSISTANT_SECRET"),
    ])


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_extract_excludes_goal_and_summary():
    url, text, reason = extract_evidence(call(), "GOAL_SECRET\nEvidence in page:\n事实正文\nSummary:\nSECRET")
    assert (url, text, reason) == ("https://example.org/a", "事实正文", None)


@pytest.mark.parametrize("tool,response,reason", [
    (call(), "[visit] Failed to visit the url", "failed_visit"),
    (call(), "No known evidence section", "missing_evidence_marker"),
    (call(), "Evidence in page:\n\nSummary:\nsecret", "empty_evidence"),
    (call(["https://example.org/a", "https://example.org/b"]), "Evidence in page:\nbody", "multi_url_visit"),
    (call([]), "Evidence in page:\nbody", "invalid_url_argument"),
    (call(7), "Evidence in page:\nbody", "invalid_url_argument"),
    (call("file:///tmp/local"), "Evidence in page:\nbody", "invalid_url"),
])
def test_unusable_visit_rejected(tool, response, reason):
    assert extract_evidence(tool, response)[2] == reason


def test_merge_by_url_deduplicate_evidence_and_preserve_provenance(tmp_path):
    first = write_rows(tmp_path / "train.jsonl", [trajectory(), trajectory()])
    second = write_rows(tmp_path / "test.jsonl", [trajectory(responses=[
        "Evidence in page:\nBlue flowers also grow here.\nSummary:\nSECRET"])])
    output = tmp_path / "evidence"
    manifest = build_corpus([first, second], output)
    documents = [json.loads(line) for line in (output / "corpus.jsonl").read_text().splitlines()]
    assert len(documents) == 1
    assert documents[0]["text"] == "Red flowers grow here.\n\nBlue flowers also grow here."
    assert documents[0]["evidence_count"] == 2
    assert documents[0]["id"] == hashlib.sha256(documents[0]["url"].encode()).hexdigest()
    for secret in ["GOAL_SECRET", "SUMMARY_SECRET", "QUESTION_SECRET", "REFERENCE_SECRET",
                   "SYSTEM_SECRET", "REASONING_SECRET", "ASSISTANT_SECRET"]:
        assert secret not in (output / "corpus.jsonl").read_text()
    provenance = [json.loads(line) for line in (output / "provenance.jsonl").read_text().splitlines()]
    assert [record["duplicate"] for record in provenance] == [False, True, False]
    assert len(manifest["inputs"]) == 2
    assert manifest["inputs"][0]["sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    assert manifest["counts"]["accepted_observations"] == 3
    assert manifest["counts"]["unique_evidence"] == 2
    assert manifest["counts"]["duplicate_observations"] == 1
    assert manifest["independent_corpus"] is False
    with pytest.raises(FileExistsError):
        build_corpus([first], output)


def test_batch_pairing_never_shifts_visit_to_search_reply(tmp_path):
    search = dict(name="search", arguments=dict(query=["flowers"]))
    valid = trajectory([search, call()], ["search snippet", "Evidence in page:\nCorrect visit evidence"])
    mismatch = trajectory([search, call()], ["Evidence in page:\nWRONG"])
    source = write_rows(tmp_path / "test.jsonl", [valid, mismatch, dict(bad="row")])
    output = tmp_path / "evidence"
    manifest = build_corpus([source], output)
    doc = json.loads((output / "corpus.jsonl").read_text())
    assert doc["text"] == "Correct visit evidence"
    skipped = [json.loads(line) for line in (output / "skipped.jsonl").read_text().splitlines()]
    assert [row["reason"] for row in skipped] == ["response_count_mismatch", "invalid_row"]
    assert manifest["counts"]["visit_calls"] == 2
    assert manifest["counts"]["invalid_rows"] == 1


def test_multiple_response_wrappers_in_one_user_message(tmp_path):
    row = trajectory([call("https://example.org/a"), call("https://example.org/b")],
                     ["Evidence in page:\nFirst", "Evidence in page:\nSecond"])
    row["trajectory"][3]["content"] += row["trajectory"].pop(4)["content"]
    source = write_rows(tmp_path / "test.jsonl", [row])
    output = tmp_path / "evidence"
    manifest = build_corpus([source], output)
    assert manifest["counts"]["documents"] == 2
    assert [json.loads(line)["text"] for line in (output / "corpus.jsonl").read_text().splitlines()] == [
        "First", "Second"]


def test_structured_tool_messages_are_supported(tmp_path):
    row = dict(answer="SECRET", messages=[
        dict(role="user", content="question"),
        dict(role="assistant", tool_calls=[dict(id="v1", function=dict(
            name="visit", arguments=json.dumps(call()["arguments"])))]),
        dict(role="tool", tool_call_id="v1", content="Evidence in page:\nStructured evidence"),
        dict(role="assistant", content="answer")])
    source = write_rows(tmp_path / "raw.jsonl", [row])
    output = tmp_path / "evidence"
    assert build_corpus([source], output)["counts"]["documents"] == 1


def test_no_evidence_is_failure_with_skip_report(tmp_path):
    source = write_rows(tmp_path / "test.jsonl", [trajectory(responses=["[visit] Failed to visit"])])
    output = tmp_path / "empty"
    with pytest.raises(ValueError, match="no usable visit evidence"):
        build_corpus([source], output)
    assert (output / "skipped.jsonl").is_file()
    assert not (output / "manifest.json").exists()


def test_input_validation_does_not_create_output(tmp_path):
    source = write_rows(tmp_path / "test.jsonl", [trajectory()])
    output = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        build_corpus([tmp_path / "absent.jsonl"], output)
    with pytest.raises(ValueError, match="duplicate input"):
        build_corpus([source, source], output)
    assert not output.exists()


def test_shell_from_other_directory_and_corpus_backend(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = write_rows(tmp_path / "test.jsonl", [trajectory()])
    output = tmp_path / "corpus output"
    script = (root / "scripts/build_openseeker_corpus.sh").read_text()
    script = script.replace('PYTHON=""', f'PYTHON="{sys.executable}"', 1)
    script = re.sub(r"INPUTS=\(.*?\n\)", f'INPUTS=(\n  "{source}"\n)', script, flags=re.S)
    script = re.sub(r'^OUTPUT_DIR=".*"$', f'OUTPUT_DIR="{output}"', script, flags=re.M)
    script = script.replace('source "$(dirname "${BASH_SOURCE[0]}")/common.sh"', f'source "{root}/scripts/common.sh"')
    result = subprocess.run(["bash", "-c", script], cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert str(output / "corpus.jsonl") in result.stdout
    from search_comp.evaluation.tool_agent import CorpusTools
    tools = CorpusTools(output / "corpus.jsonl")
    assert "https://example.org/a" in tools.search(["red flowers"])
    assert "Red flowers grow here." in tools.visit("https://example.org/a", "find flowers")


@pytest.mark.parametrize('prepared', [False, True])
def test_final_answer_target_never_enters_corpus_or_changes_evidence(tmp_path, prepared):
    from search_comp.data.openseeker import normalize, VERSION
    original=trajectory()
    original['trajectory'][-1]['content']='<think>THINK_ONLY_SECRET</think><answer>EXPLANATION_ONLY_SECRET</answer>'
    first=build_corpus([write_rows(tmp_path/'original.jsonl',[original])],tmp_path/'original-corpus')
    changed=trajectory()
    changed['trajectory'][-1]['content']=original['trajectory'][-1]['content']
    changed['answer']='FINAL_ANSWER_ONLY_SECRET'
    if prepared:
        changed=normalize(changed)
        assert changed['protocol']==VERSION
        assert '<final_answer>FINAL_ANSWER_ONLY_SECRET</final_answer>' in changed['messages'][-1]['content']
    build_corpus([write_rows(tmp_path/'changed.jsonl',[changed])],tmp_path/'changed-corpus')
    a=(tmp_path/'original-corpus/corpus.jsonl').read_text()
    b=(tmp_path/'changed-corpus/corpus.jsonl').read_text()
    assert a==b
    assert 'Red flowers grow here.' in b
    for secret in ('FINAL_ANSWER_ONLY_SECRET','REFERENCE_SECRET','THINK_ONLY_SECRET',
                   'EXPLANATION_ONLY_SECRET','<final_answer>','</final_answer>'):
        assert secret not in b
    assert first['independent_corpus'] is False
