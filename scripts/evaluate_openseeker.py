#!/usr/bin/env python
"""Evaluate OpenSeeker checkpoints, with optional one-model-per-GPU workers."""
import argparse
from collections import Counter
from contextlib import ExitStack
import heapq
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from search_comp.data.openseeker import normalize
from search_comp.evaluation.em_f1 import compute_f1
from search_comp.evaluation.tool_agent import HTTPTools, ReplayTools, CorpusTools, ModelGenerator, run_agent


def score_answer(result, answer):
    """Keep historical strict EM; score failed/missing-reference answers as zero."""
    answer = str(answer)
    if result["status"] != "completed" or not answer.strip():
        return False, 0.0
    prediction = result["prediction"]
    exact = " ".join(prediction.casefold().split()) == " ".join(answer.casefold().split())
    return exact, compute_f1(prediction, answer)


def selected_rows(path, limit, shard_index=0, shard_count=1):
    """Shard the same first LIMIT rows; preserve source indices across workers."""
    with path.open() as source:
        for index, line in enumerate(source):
            if index >= limit:
                break
            if index % shard_count == shard_index:
                yield index, normalize(json.loads(line), index=index)


def write_summary(args, count, exact, f1, statuses, seconds):
    summary = dict(
        mode=args.backend,
        interpretation={"replay": "exact-action replay diagnostic",
                        "corpus": "local corpus tool rollout",
                        "http": "live tool rollout"}[args.backend],
        data_split=args.data_split,
        count=count,
        exact_match=exact / count if count else None,
        f1=f1 / count if count else None,
        statuses=dict(statuses),
        seconds=seconds,
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )
    args.output.with_suffix(".metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


class ProgressReporter:
    """Flush log lines immediately and keep long loading/generation phases visible."""
    def __init__(self, total, interval, label):
        self.total = total
        self.interval = interval
        self.label = label
        self.started = time.monotonic()
        self.phase_started = self.started
        self.evaluation_started = None
        self.current = "initializing"
        self.count = self.exact = 0
        self.f1 = 0.0
        self.lock = threading.Lock()
        self.stopped = threading.Event()

    def __enter__(self):
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stopped.set()
        self.thread.join()

    def _emit(self, detail):
        print(f"[eval {self.label}] {detail}", flush=True)

    def _heartbeat(self):
        while not self.stopped.wait(self.interval):
            with self.lock:
                self._emit(f"running phase={self.current}; done={self.count}/{self.total}; "
                           f"phase_elapsed={time.monotonic() - self.phase_started:.1f}s")

    def phase(self, name):
        with self.lock:
            self.current = name
            self.phase_started = time.monotonic()
            self._emit(name)

    def sample(self, index):
        with self.lock:
            if self.evaluation_started is None:
                self.evaluation_started = time.monotonic()
        self.phase(f"sample_index={index} generating")

    def advance(self, status, exact, f1, seconds):
        with self.lock:
            self.count += 1
            self.exact += exact
            self.f1 += f1
            elapsed = time.monotonic() - self.evaluation_started
            eta = elapsed / self.count * (self.total - self.count)
            self.current = "between samples"
            self.phase_started = time.monotonic()
            self._emit(f"done={self.count}/{self.total} ({100 * self.count / self.total:.1f}%) "
                       f"status={status} sample_time={seconds:.1f}s "
                       f"EM={self.exact / self.count:.4f} F1={self.f1 / self.count:.4f} "
                       f"elapsed={elapsed:.1f}s ETA={eta:.1f}s")


def sample_count(args):
    with args.data.open() as source:
        size = sum(1 for _ in itertools.islice(source, args.limit))
    return len(range(args.shard_index, size, args.shard_count))


def forward_progress(readers, buffers):
    """Tail only our progress lines, retaining partial lines until the next poll."""
    for index, reader in enumerate(readers):
        buffers[index] += reader.read()
        lines = buffers[index].split("\n")
        buffers[index] = lines.pop()
        for line in lines:
            if line.startswith("[eval "):
                print(line, flush=True)


def evaluate(args):
    total = sample_count(args)
    label = f"worker={args.shard_index}" if args.shard_count > 1 else "single"
    with ProgressReporter(total, args.progress_interval, label) as progress:
        progress.phase(f"loading model libraries; samples={total}; split={args.data_split}")
        from search_comp.models.model_loader import load_beacon_model, load_tokenizer
        from search_comp.models.plain_qwen3 import load_sft_model

        progress.phase(f"loading tokenizer: {args.model}")
        tok = load_tokenizer(args.model)
        progress.phase(f"loading model and moving to device: {args.model}")
        model = (load_beacon_model if args.kind == "beacon" else load_sft_model)(args.model)
        model.eval()
        generate = ModelGenerator(model, tok, args.kind == "beacon")
        progress.phase(f"initializing backend={args.backend}" +
                       (f"; loading corpus and building BM25: {args.corpus}" if args.backend == "corpus" else ""))
        shared_backend = (HTTPTools(args.endpoint) if args.backend == "http" else
                          CorpusTools(args.corpus) if args.backend == "corpus" else None)
        progress.phase("backend ready; starting evaluation")
        statuses = Counter()
        exact = count = 0
        f1 = 0.0
        start = time.time()
        with args.output.open("x") as out:
            for index, row in selected_rows(args.data, args.limit, args.shard_index, args.shard_count):
                backend = ReplayTools(row["messages"]) if args.backend == "replay" else shared_backend
                progress.sample(index)
                t = time.time()
                result = run_agent(row["messages"], tok, generate, backend, args.max_turns,
                                   args.max_context, args.max_new_tokens, args.max_tool_tokens)
                correct, answer_f1 = score_answer(result, row["answer"])
                result.update(id=row["id"], sample_index=index, answer=row["answer"],
                              exact_match=correct, f1=answer_f1, seconds=time.time() - t)
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
                statuses[result["status"]] += 1
                exact += correct
                f1 += answer_f1
                count += 1
                progress.advance(result["status"], correct, answer_f1, result["seconds"])
        progress.phase(f"finished; writing metrics: {args.output.with_suffix('.metrics.json')}")
        return write_summary(args, count, exact, f1, statuses, time.time() - start)


def worker_devices(num_workers, visible):
    devices = [device.strip() for device in visible.split(",") if device.strip()]
    if (len(devices) < num_workers or len(set(devices)) != len(devices)
            or any(device == "-1" for device in devices)):
        raise ValueError("set CUDA_VISIBLE_DEVICES to distinct GPUs, at least one per worker")
    return devices[:num_workers]


def merge_outputs(paths, output):
    """Merge ordered shards without loading full trajectories into memory."""
    count = exact = 0
    f1 = 0.0
    statuses = Counter()
    with ExitStack() as stack:
        sources = [stack.enter_context(path.open()) for path in paths]
        rows = [map(json.loads, source) for source in sources]
        with output.open("x") as out:
            for row in heapq.merge(*rows, key=lambda row: row["sample_index"]):
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
                exact += row["exact_match"]
                f1 += row["f1"]
                statuses[row["status"]] += 1
    return count, exact, f1, statuses


def evaluate_parallel(args):
    devices = worker_devices(args.num_workers, os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    # Keep worker artifacts for inspection, especially when one process fails.
    worker_dir = args.output.with_suffix(".workers")
    worker_dir.mkdir()
    processes = []
    paths = []
    start = time.time()
    with ExitStack() as stack:
        readers = []
        buffers = []
        try:
            for rank, device in enumerate(devices):
                path = worker_dir / f"shard_{rank}.jsonl"
                paths.append(path)
                command = [sys.executable, str(Path(__file__).resolve())]
                for key, value in vars(args).items():
                    if key not in {"output", "num_workers", "shard_index", "shard_count"} and value is not None:
                        command.extend(["--" + key.replace("_", "-"), str(value)])
                command.extend(["--output", str(path), "--num-workers", "1",
                                "--shard-index", str(rank), "--shard-count", str(len(devices))])
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, PYTHONUNBUFFERED='1')
                log = stack.enter_context((worker_dir / f"worker_{rank}.log").open("x"))
                print(f"[worker {rank}] GPU={device}; log={log.name}", flush=True)
                processes.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT))
                readers.append(stack.enter_context(Path(log.name).open(errors='replace')))
                buffers.append('')
            pending = list(processes)
            while pending:
                forward_progress(readers, buffers)
                for process in pending[:]:
                    code = process.poll()
                    if code is None:
                        continue
                    if code:
                        raise RuntimeError(f"evaluation worker exited with code {code}; see {worker_dir}")
                    pending.remove(process)
                if pending:
                    time.sleep(0.1)
            forward_progress(readers, buffers)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                process.wait()
    print(f'[eval all] workers finished; merging predictions: {args.output}', flush=True)
    count, exact, f1, statuses = merge_outputs(paths, args.output)
    return write_summary(args, count, exact, f1, statuses, time.time() - start)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--kind", choices=["plain", "beacon"], default="beacon")
    p.add_argument("--backend", choices=["replay", "http", "corpus"], default="replay")
    p.add_argument("--endpoint")
    p.add_argument("--corpus", type=Path)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--max-context", type=int, default=16384)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--max-tool-tokens", type=int, default=16384)
    p.add_argument("--max-turns", type=int, default=32)
    p.add_argument("--data-split", choices=["train", "validation", "test", "custom"], default="test")
    p.add_argument("--num-workers", type=int, default=1,
                   help="one model replica per GPU; explicitly set CUDA_VISIBLE_DEVICES for multiple workers")
    p.add_argument("--progress-interval", type=float, default=15.0,
                   help="seconds between heartbeat messages during long phases (default: 15)")
    p.add_argument("--shard-index", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--shard-count", type=int, default=1, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.backend == "http" and not args.endpoint:
        p.error("--endpoint required for http")
    if args.backend == "corpus" and not args.corpus:
        p.error("--corpus required for corpus backend")
    if not 0 < args.progress_interval < float("inf"):
        p.error("--progress-interval must be finite and positive")
    if args.num_workers < 1 or args.limit < 1:
        p.error("--num-workers and --limit must be positive")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        p.error("invalid shard index/count")
    if args.num_workers > 1:
        if args.shard_count != 1:
            p.error("cannot nest parallel workers")
        try:
            worker_devices(args.num_workers, os.environ.get("CUDA_VISIBLE_DEVICES", ""))
        except ValueError as error:
            p.error(str(error))
    if not args.data.is_file():
        p.error(f"data does not exist: {args.data}")
    if args.output.exists() or args.output.with_suffix(".metrics.json").exists():
        p.error("output exists; use a new evaluation name")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.num_workers > 1:
        evaluate_parallel(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
