#!/usr/bin/env python
"""Build a local diagnostic corpus from recorded OpenSeeker visit evidence.

No downloads, model loading, tokenization, or assistant-answer extraction.
Outputs are source-conditioned evidence, not full webpages or independent retrieval data.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from search_comp.data.openseeker import normalize, parse_calls, RESPONSE_RE

EVIDENCE = re.compile(r"(?:\A|\n)[ \t]*Evidence in page:[ \t]*(?:\r?\n)?", re.I)
SUMMARY = re.compile(r"(?:\A|\n)[ \t]*Summary:[ \t]*(?:\r?\n)?", re.I)
FAILED = re.compile(r"^\s*(?:\[visit\]\s*)?(?:failed\b|error\b)", re.I)


def dump(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_evidence(call, response):
    """Return (URL, evidence, rejection); never assign a multi-URL summary to every URL."""
    urls = call["arguments"].get("url", [])
    urls = [urls] if isinstance(urls, str) else urls
    if not isinstance(urls, list) or not urls or any(not isinstance(u, str) for u in urls):
        return None, None, "invalid_url_argument"
    if len(urls) != 1:
        return None, None, "multi_url_visit"
    url = urls[0].strip()
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None, None, "invalid_url"
    if FAILED.match(response):
        return url, None, "failed_visit"
    match = EVIDENCE.search(response)
    if match is None:
        return url, None, "missing_evidence_marker"
    text = SUMMARY.split(response[match.end():], maxsplit=1)[0].strip()
    if not text:
        return url, None, "empty_evidence"
    return url, text, None


def tool_batches(messages):
    """Pair entire batches first; mismatched reply counts must not shift attribution."""
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        calls = parse_calls(message["content"])
        if not calls:
            continue
        responses = []
        for following in messages[index + 1:]:
            if following["role"] != "user":
                break
            responses.extend(RESPONSE_RE.findall(following["content"]))
        yield index, calls, responses


def build_corpus(inputs, output_dir, progress_every=100):
    inputs = [Path(path).resolve() for path in inputs]
    output_dir = Path(output_dir)
    if not inputs:
        raise ValueError("at least one input is required")
    if len(set(inputs)) != len(inputs):
        raise ValueError("duplicate input paths")
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"input does not exist: {path}")
    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    output_dir.mkdir(parents=True, exist_ok=False)
    documents = {}
    counts = Counter()
    input_metadata = []
    with ExitStack() as stack:
        skipped = stack.enter_context((output_dir / "skipped.jsonl").open("x", encoding="utf-8"))
        provenance = stack.enter_context((output_dir / "provenance.jsonl").open("x", encoding="utf-8"))

        def reject(reason, origin, **extra):
            counts["skipped_" + reason] += 1
            dump(skipped, dict(origin, reason=reason, **extra))

        for path in inputs:
            print(f"[openseeker-corpus] reading: {path}", flush=True)
            digest = hashlib.sha256()
            byte_count = rows = 0
            with path.open("rb") as source:
                for line_number, line in enumerate(source, 1):
                    digest.update(line)
                    byte_count += len(line)
                    if not line.strip():
                        continue
                    rows += 1
                    counts["input_rows"] += 1
                    origin = dict(input=str(path), line=line_number)
                    try:
                        row = normalize(json.loads(line), source=path.stem, index=line_number - 1)
                    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
                        counts["invalid_rows"] += 1
                        reject("invalid_row", origin, error=str(error))
                        continue
                    origin.update(row_id=row["id"], question_id=row["question_id"])
                    counts["valid_rows"] += 1
                    row_has_evidence = False
                    for message_index, calls, responses in tool_batches(row["messages"]):
                        counts["tool_calls"] += len(calls)
                        visits = [(i, call) for i, call in enumerate(calls) if call["name"] == "visit"]
                        counts["visit_calls"] += len(visits)
                        if len(calls) != len(responses):
                            for call_index, call in visits:
                                reject("response_count_mismatch", dict(origin, message_index=message_index,
                                       call_index=call_index), calls=len(calls), responses=len(responses))
                            continue
                        for call_index, call in visits:
                            location = dict(origin, message_index=message_index, call_index=call_index)
                            try:
                                url, text, reason = extract_evidence(call, responses[call_index])
                            except ValueError:
                                url, text, reason = None, None, "invalid_url"
                            if reason:
                                reject(reason, location, url=url)
                                continue
                            document = documents.setdefault(url, {})
                            evidence_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                            duplicate = evidence_hash in document
                            document.setdefault(evidence_hash, text)
                            counts["accepted_observations"] += 1
                            counts["duplicate_observations" if duplicate else "unique_evidence"] += 1
                            row_has_evidence = True
                            dump(provenance, dict(location, url=url,
                                 document_id=hashlib.sha256(url.encode("utf-8")).hexdigest(),
                                 evidence_sha256=evidence_hash, duplicate=duplicate))
                    if row_has_evidence:
                        counts["rows_with_evidence"] += 1
                    if counts["input_rows"] % progress_every == 0:
                        print(f"[openseeker-corpus] rows={counts['input_rows']}; documents={len(documents)}; "
                              f"evidence={counts['unique_evidence']}", flush=True)
            input_metadata.append(dict(path=str(path), bytes=byte_count, sha256=digest.hexdigest(), rows=rows))

    corpus = output_dir / "corpus.jsonl"
    with corpus.open("x", encoding="utf-8") as out:
        for url, evidence in documents.items():
            dump(out, dict(id=hashlib.sha256(url.encode("utf-8")).hexdigest(),
                           url=url, title=url, text="\n\n".join(evidence.values()),
                           source="openseeker_recorded_visit_evidence", evidence_count=len(evidence)))
    if not documents:
        raise ValueError(f"no usable visit evidence; inspect {output_dir / 'skipped.jsonl'}; "
                         "no success manifest was written")
    counts["documents"] = len(documents)
    manifest = dict(
        schema_version=1,
        interpretation="local retrieval over source-conditioned recorded visit evidence",
        independent_corpus=False,
        evidence_policy="single-URL visit only; Evidence in page body; omit Summary and goal preamble",
        deduplication="exact URL; SHA256 of stripped evidence text within each URL; preserve input order",
        created_at=datetime.now(timezone.utc).isoformat(),
        inputs=input_metadata,
        counts=dict(counts),
        arguments=dict(progress_every=progress_every),
        corpus=dict(path="corpus.jsonl", bytes=corpus.stat().st_size,
                    sha256=hashlib.sha256(corpus.read_bytes()).hexdigest()),
        limitations=[
            "Recorded evidence was selected using source questions/goals; not an independent web corpus.",
            "Only visited evidence is included; search-only pages and failed/unrecognized visits are absent.",
            "Multiple evidence excerpts for one URL are merged; goal-conditioned Summary is excluded.",
            "No guarantee of answer coverage, full webpage content, or bounded tokenizer length.",
        ],
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")
    print(f"[openseeker-corpus] complete: {corpus}; documents={len(documents)}", flush=True)
    print(json.dumps(dict(counts), ensure_ascii=False, indent=2), flush=True)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, action="append", required=True,
                   help="OpenSeeker JSONL, repeat for train/validation/test")
    p.add_argument("--output-dir", type=Path, required=True, help="must not already exist")
    p.add_argument("--progress-every", type=int, default=100)
    args = p.parse_args()
    try:
        build_corpus(args.input, args.output_dir, args.progress_every)
    except (ValueError, OSError) as error:
        p.exit(2, f"[openseeker-corpus] error: {error}\n")


if __name__ == "__main__":
    main()
