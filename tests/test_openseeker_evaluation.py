"""CPU regression tests for OpenSeeker scores, split selection and GPU worker orchestration."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import evaluate_openseeker as evaluation


def dataset(tmp_path, count=7):
    path = tmp_path / "data.jsonl"
    rows = [dict(answer="red blue", messages=[
        dict(role="system", content="Answer."),
        dict(role="user", content=f"Question {i}"),
        dict(role="assistant", content="red blue")]) for i in range(count)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def arguments(tmp_path, data):
    return argparse.Namespace(
        model="fake-model", data=data, output=tmp_path / "predictions.jsonl",
        kind="beacon", backend="replay", endpoint=None, corpus=None, limit=7,
        max_context=1024, max_new_tokens=64, max_tool_tokens=128, max_turns=4,
        data_split="train", num_workers=1, shard_index=0, shard_count=1)


@pytest.mark.parametrize("prediction,answer,status,em,f1", [
    ("red", "red blue", "completed", False, 2 / 3),
    (" The RED! ", "red", "completed", False, 1.0),
    ("RED", "red", "completed", True, 1.0),
    ("red", "red", "tool_error", False, 0.0),
    ("red", "", "completed", False, 0.0),
    ("", "red", "completed", False, 0.0),
])
def test_answer_scores(prediction, answer, status, em, f1):
    actual_em, actual_f1 = evaluation.score_answer(dict(prediction=prediction, status=status), answer)
    assert actual_em == em
    assert actual_f1 == pytest.approx(f1)


def test_shards_cover_same_limit_once_in_order(tmp_path):
    path = dataset(tmp_path)
    shards = [[i for i, row in evaluation.selected_rows(path, 5, rank, 3)] for rank in range(3)]
    assert shards == [[0, 3], [1, 4], [2]]
    assert sorted(i for shard in shards for i in shard) == list(range(5))


def test_evaluate_writes_scores_and_failed_denominator(tmp_path, monkeypatch):
    args = arguments(tmp_path, dataset(tmp_path, 2))
    monkeypatch.setitem(sys.modules, "search_comp.models.model_loader", SimpleNamespace(
        load_tokenizer=lambda _: object(), load_beacon_model=lambda _: SimpleNamespace(eval=lambda: None)))
    monkeypatch.setitem(sys.modules, "search_comp.models.plain_qwen3", SimpleNamespace(load_sft_model=None))
    monkeypatch.setattr(evaluation, "ModelGenerator", lambda *args: None)
    results = iter([dict(status="completed", prediction="red"), dict(status="tool_error", prediction="red blue")])
    monkeypatch.setattr(evaluation, "run_agent", lambda *args: next(results))
    summary = evaluation.evaluate(args)
    rows = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert [row["f1"] for row in rows] == pytest.approx([2 / 3, 0])
    assert summary["f1"] == pytest.approx(1 / 3)
    assert summary["exact_match"] == 0
    assert summary["count"] == 2
    assert summary["data_split"] == "train"


@pytest.mark.parametrize("visible", ["", "0", "0,0", "-1,0"])
def test_invalid_worker_devices(visible):
    with pytest.raises(ValueError):
        evaluation.worker_devices(2, visible)


def test_multi_gpu_shards_merged_and_weighted(tmp_path, monkeypatch):
    args = arguments(tmp_path, dataset(tmp_path))
    args.num_workers = 2
    args.limit = 5
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")
    assignments = []

    class FakeProcess:
        def __init__(self, command, env, stdout, stderr):
            get = lambda flag: command[command.index(flag) + 1]
            rank, count = int(get("--shard-index")), int(get("--shard-count"))
            assignments.append(env["CUDA_VISIBLE_DEVICES"])
            # Model each worker's output, including uneven shard sizes.
            with Path(get("--output")).open("x") as out:
                for index, row in evaluation.selected_rows(args.data, int(get("--limit")), rank, count):
                    score = float(index % 2 == 0)
                    out.write(json.dumps(dict(sample_index=index, exact_match=bool(score),
                                              f1=score, status="completed")) + "\n")

        def poll(self):
            return 0

        def wait(self):
            return 0

    monkeypatch.setattr(evaluation.subprocess, "Popen", FakeProcess)
    summary = evaluation.evaluate_parallel(args)
    rows = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert assignments == ["2", "5"]
    assert [row["sample_index"] for row in rows] == list(range(5))
    assert summary["count"] == 5
    assert summary["exact_match"] == pytest.approx(3 / 5)
    assert summary["f1"] == pytest.approx(3 / 5)
    assert summary["statuses"] == {"completed": 5}


def test_worker_failure_stops_other_workers_without_final_metrics(tmp_path, monkeypatch):
    args = arguments(tmp_path, dataset(tmp_path))
    args.num_workers = 2
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    processes = []

    class FakeProcess:
        def __init__(self, *a, **kw):
            self.code = 1 if not processes else None
            self.terminated = False
            processes.append(self)

        def poll(self):
            return self.code

        def terminate(self):
            self.terminated = True
            self.code = -15

        def wait(self):
            return self.code

    monkeypatch.setattr(evaluation.subprocess, "Popen", FakeProcess)
    with pytest.raises(RuntimeError, match="worker exited"):
        evaluation.evaluate_parallel(args)
    assert processes[1].terminated
    assert not args.output.exists()
    assert not args.output.with_suffix(".metrics.json").exists()


@pytest.mark.parametrize("split,key", [
    ("train", "train_data_path"), ("validation", "val_data_path"), ("test", "test_data_path"),
])
def test_shell_selects_experiment_split(tmp_path, split, key):
    root = Path(__file__).resolve().parents[1]
    run = tmp_path / "experiment"
    run.mkdir()
    data = dataset(tmp_path)
    (run / "experiment.json").write_text(json.dumps(dict(
        kind="beacon", training_config={key: str(data)})))
    source = (root / "scripts/test_openseeker.sh").read_text()
    replacements = {
        'PYTHON=""': f'PYTHON="{sys.executable}"',
        'DRY_RUN=0': 'DRY_RUN=1',
        'RUN_DIR=""': f'RUN_DIR="{run}"',
        'MODEL_PATH=""': 'MODEL_PATH="/tmp/model with spaces"',
        'SPLIT="test"': f'SPLIT="{split}"',
        'source "$(dirname "${BASH_SOURCE[0]}")/common.sh"': f'source "{root}/scripts/common.sh"',
    }
    for old, new in replacements.items():
        source = source.replace(old, new, 1)
    result = subprocess.run(["bash", "-c", source], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert f"--data-split {split}" in result.stdout
    assert str(data) in result.stdout
    assert not (run / "evaluation").exists()


def test_real_worker_processes_receive_separate_gpus_and_merge(tmp_path, monkeypatch):
    args = arguments(tmp_path, dataset(tmp_path))
    args.num_workers = 2
    args.limit = 5
    root = Path(__file__).resolve().parents[1]
    helper = tmp_path / "cpu_worker.py"
    helper.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(root)!r})\n"
        "from scripts.evaluate_openseeker import selected_rows\n"
        "options = dict(zip(sys.argv[1::2], sys.argv[2::2]))\n"
        "path = Path(options['--output'])\n"
        "path.with_suffix('.device').write_text(os.environ['CUDA_VISIBLE_DEVICES'])\n"
        "with path.open('x') as out:\n"
        "    for index, row in selected_rows(Path(options['--data']), int(options['--limit']),\n"
        "                                     int(options['--shard-index']), int(options['--shard-count'])):\n"
        "        out.write(json.dumps(dict(sample_index=index, exact_match=True, f1=1.0, status='completed')) + '\\n')\n"
        "print('worker completed')\n"
    )
    monkeypatch.setattr(evaluation, "__file__", str(helper))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7")
    summary = evaluation.evaluate_parallel(args)
    directory = args.output.with_suffix(".workers")
    assert (directory / "shard_0.device").read_text() == "3"
    assert (directory / "shard_1.device").read_text() == "7"
    assert "worker completed" in (directory / "worker_0.log").read_text()
    assert [json.loads(line)["sample_index"] for line in args.output.read_text().splitlines()] == list(range(5))
    assert summary["count"] == 5
    assert summary["f1"] == 1.0
