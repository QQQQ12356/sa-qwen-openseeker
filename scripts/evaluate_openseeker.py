#!/usr/bin/env python
"""Evaluate OpenSeeker checkpoints, with optional one-model-per-GPU workers."""
import argparse
from collections import Counter
from contextlib import ExitStack
import heapq
import json
import os
from pathlib import Path
import subprocess
import sys
import time

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


def evaluate(args):
    from search_comp.models.model_loader import load_beacon_model, load_tokenizer
    from search_comp.models.plain_qwen3 import load_sft_model

    tok = load_tokenizer(args.model)
    model = (load_beacon_model if args.kind == "beacon" else load_sft_model)(args.model)
    model.eval()
    generate = ModelGenerator(model, tok, args.kind == "beacon")
    shared_backend = (HTTPTools(args.endpoint) if args.backend == "http" else
                      CorpusTools(args.corpus) if args.backend == "corpus" else None)
    statuses = Counter()
    exact = count = 0
    f1 = 0.0
    start = time.time()
    with args.output.open("x") as out:
        for index, row in selected_rows(args.data, args.limit, args.shard_index, args.shard_count):
            backend = ReplayTools(row["messages"]) if args.backend == "replay" else shared_backend
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
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=device)
                log = stack.enter_context((worker_dir / f"worker_{rank}.log").open("x"))
                print(f"[worker {rank}] GPU={device}; log={log.name}", flush=True)
                processes.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT))
            pending = list(processes)
            while pending:
                for process in pending[:]:
                    code = process.poll()
                    if code is None:
                        continue
                    if code:
                        raise RuntimeError(f"evaluation worker exited with code {code}; see {worker_dir}")
                    pending.remove(process)
                if pending:
                    time.sleep(0.1)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                process.wait()
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
    p.add_argument("--shard-index", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--shard-count", type=int, default=1, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.backend == "http" and not args.endpoint:
        p.error("--endpoint required for http")
    if args.backend == "corpus" and not args.corpus:
        p.error("--corpus required for corpus backend")
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
