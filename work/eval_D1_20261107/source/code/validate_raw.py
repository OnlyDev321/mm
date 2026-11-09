"""Validate raw generation records without loading images or model packages."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import yaml

try:
    from .common import check_invariants, validate_record
except ImportError:
    from common import check_invariants, validate_record

try:
    from .metadata import validate_run_metadata
except ImportError:
    from metadata import validate_run_metadata


def read_records(path: Path) -> tuple[list[dict], list[str]]:
    records, errors = [], []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        errors.append("last line has no newline")
    for number, line in enumerate(raw.splitlines(), 1):
        try:
            record = json.loads(line)
            validate_record(record)
            records.append(record)
        except Exception as exc:
            errors.append(f"line {number}: {exc!r}")
    return records, errors


def validate(path: Path, config: dict, expected: int, metadata_path: Path | None = None) -> tuple[int, dict]:
    records, errors = read_records(path)
    ids = [record.get("id") for record in records]
    if len(ids) != len(set(ids)):
        errors.append("duplicate ids")
    if len(records) != expected:
        errors.append(f"row count {len(records)} != expected {expected}")
    if expected == config["dataset"]["expected_total_rows"]:
        counts = Counter(record["subject"] for record in records)
        required = config["dataset"]["expected_rows_per_subject"]
        for subject in config["dataset"]["subjects"]:
            if counts[subject] != required:
                errors.append(f"subject {subject}: {counts[subject]} != {required}")
    invariant_count = 0
    mismatch_count = 0
    for record in records:
        invariant_count += len(check_invariants(record))
        if record["status"] == "ok" and record["precomputed_prompt_tokens"] != record["num_prompt_tokens"]:
            mismatch_count += 1
    if invariant_count:
        errors.append(f"invariant violations: {invariant_count}")
    if mismatch_count:
        errors.append(f"prompt token mismatches: {mismatch_count}")

    # Validate run_metadata.json if present
    meta_path = metadata_path or (path.parent / "run_metadata.json")
    if meta_path.exists():
        meta_code, meta_errors = validate_run_metadata(meta_path, raw_path=path)
        if meta_errors:
            errors.extend([f"metadata: {err}" for err in meta_errors])
        else:
            print(f"metadata check: OK ({meta_path.name})")

    statuses = Counter(record["status"] for record in records)
    summary = {"rows": len(records), "status": dict(statuses), "invariants": invariant_count,
               "prompt_token_mismatches": mismatch_count, "errors": errors,
               "skip": statuses["skip"], "error": statuses["error"]}
    print("status:", dict(statuses))
    print("invariant violations:", invariant_count)
    print("prompt token mismatches:", mismatch_count)
    print("skip:", statuses["skip"], "error:", statuses["error"])
    for error in errors:
        print("FAIL:", error)
    if errors:
        return 1, summary
    return (2 if statuses["skip"] or statuses["error"] else 0), summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expect", type=int)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    expected = args.expect if args.expect is not None else config["dataset"]["expected_total_rows"]
    code, _ = validate(args.raw, config, expected, metadata_path=args.metadata)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
