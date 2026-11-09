"""MMMU generation driver; the CPU path uses deterministic synthetic outputs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

import yaml

if __package__:
    from .common import (build_messages, build_prompt, check_invariants, load_samples, make_record,
                         select_smoke, validate_record)
else:
    from common import (build_messages, build_prompt, check_invariants, load_samples, make_record,
                        select_smoke, validate_record)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TokenCounter:
    def count(self, sample, prompt: str, messages: list[dict], ordinal: int, cfg: dict) -> int:
        raise NotImplementedError


class FakeTokenCounter(TokenCounter):
    """The third selected sample deliberately exceeds the configured input bound."""

    def count(self, sample, prompt: str, messages: list[dict], ordinal: int, cfg: dict) -> int:
        bound = (cfg["budget"]["max_model_len"] - cfg["budget"]["max_new_tokens"]
                 - cfg["budget"]["prompt_safety_margin"])
        return bound + 1 if ordinal == 2 else len(prompt.split())


class Engine:
    def generate(self, requests: list[dict]) -> list[dict | Exception]:
        raise NotImplementedError


class FakeEngine(Engine):
    """Synthetic only: selected positions 1, 2, 3 mean length, skip, error."""

    def __init__(self, stop_after_chunks: int | None = None, mismatch_ordinal: int | None = None):
        self.calls: list[list[str]] = []
        self.stop_after_chunks = stop_after_chunks
        self.mismatch_ordinal = mismatch_ordinal

    def generate(self, requests: list[dict]) -> list[dict | Exception]:
        self.calls.append([request["sample"].id for request in requests])
        outputs = []
        for request in requests:
            ordinal = request["ordinal"]
            if ordinal == 3:
                outputs.append(RuntimeError(f"synthetic failure for {request['sample'].id}"))
                continue
            length = ordinal == 1
            max_tokens = request["cfg"]["budget"]["max_new_tokens"]
            token_ids = [ordinal] * max_tokens if length else [ordinal]
            outputs.append({"raw_text": f"synthetic:{request['sample'].id}",
                            "output_token_ids": token_ids,
                            "finish_reason": "length" if length else "stop", "stop_reason": None,
                            "num_prompt_tokens": request["precomputed"] +
                            (1 if ordinal == self.mismatch_ordinal else 0)})
        return outputs


def git_info() -> dict:
    def run(*args):
        try:
            result = subprocess.run(["git", *args], cwd=Path(__file__).resolve().parent,
                                    capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = run("rev-parse", "HEAD")
    dirty = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit or "unknown", "dirty": bool(dirty) if dirty is not None else "unknown"}


def recover_raw(path: Path) -> list[dict]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if not raw:
        return []
    lines = raw.splitlines(keepends=True)
    records = []
    valid_bytes = 0
    for index, line in enumerate(lines):
        try:
            if not line.endswith(b"\n"):
                raise ValueError("line has no final newline")
            record = json.loads(line)
            validate_record(record)
        except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
            if index != len(lines) - 1:
                raise ValueError(f"invalid raw line {index + 1} before last line")
            torn = path.with_name(path.name + ".torn")
            torn.write_bytes(line)
            with path.open("r+b") as stream:
                stream.truncate(valid_bytes)
            print(f"Recovered torn final line to {torn}")
            break
        records.append(record)
        valid_bytes += len(line)
    return records


def resume_state(raw_path: Path, env_path: Path, selected: list, config_sha256: str,
                 dry_run: bool, limit: int | None, subjects: list[str] | None) -> list[dict]:
    if raw_path.exists() and not env_path.exists():
        raise ValueError("raw file exists without environment metadata")
    if env_path.exists():
        env = json.loads(env_path.read_text(encoding="utf-8"))
        if env.get("config_sha256") != config_sha256:
            raise ValueError("config_sha256 mismatch")
        if env.get("dry_run") is not dry_run:
            raise ValueError("dry_run mode mismatch")
        if env.get("limit") != limit:
            raise ValueError("limit mismatch")
        if env.get("subjects") != subjects:
            raise ValueError("subjects mismatch")
    records = recover_raw(raw_path)
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate id in raw file")
    selected_ids = {sample.id for sample in selected}
    foreign = set(ids) - selected_ids
    if foreign:
        raise ValueError(f"raw contains ids outside selection: {sorted(foreign)}")
    return records


def write_env(path: Path, env: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(env, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def print_summary(records: list[dict], mismatches: int, code_commits: list[str],
                  env: dict | None = None) -> int:
    statuses = Counter(record["status"] for record in records)
    finishes = Counter(record["finish_reason"] for record in records if record["finish_reason"] is not None)
    skips = Counter(record["reason"] for record in records if record["status"] == "skip")
    tokens = sorted(record["output_tokens"] for record in records if record["output_tokens"] is not None)
    distribution = ({"min": tokens[0], "median": statistics.median(tokens),
                     "p90": tokens[math.ceil(len(tokens) * 0.9) - 1], "max": tokens[-1]}
                    if tokens else "unknown")
    lengths = defaultdict(list)
    for record in records:
        if record["finish_reason"] == "length":
            lengths[record["subject"]].append(record["id"])
    violations = sum(len(check_invariants(record)) for record in records)
    print("status:", dict(statuses))
    print("finish_reason:", dict(finishes))
    print("output_tokens:", distribution)
    print("skip reasons:", dict(skips), "errors:", statuses["error"])
    print("length ids:", dict(lengths))
    print("invariant violations:", violations, "token mismatches:", mismatches)
    print("code_commits:", code_commits)
    if env is not None:
        invocations = env.get("invocations", [])
        elapsed = sum(item.get("duration_seconds") or 0 for item in invocations)
        produced = sum(record["output_tokens"] or 0 for record in records)
        print("total seconds:", round(elapsed, 3), "output tokens/s:",
              round(produced / elapsed, 3) if elapsed else "unknown")
        print("length percentage:", round(100 * finishes["length"] / max(1, statuses["ok"]), 2),
              "length by subject:", {key: len(value) for key, value in lengths.items()})
        last = next((item for item in reversed(invocations) if "engine" in item),
                    invocations[-1] if invocations else {})
        print("VRAM:", last.get("vram", "unknown"))
        print("vLLM logs:", last.get("engine", {}).get("logs", "unknown"))
        print("engine overrides:", env.get("engine_overrides", {}))
    return 2 if statuses["error"] or violations else 0


def parse_engine_overrides(values: list[str] | None) -> dict:
    if __package__:
        from .vllm_engine import OVERRIDE_KEYS
    else:
        from vllm_engine import OVERRIDE_KEYS
    overrides = {}
    for value in values or []:
        key, separator, encoded = value.partition("=")
        if not separator or key not in OVERRIDE_KEYS or key in overrides:
            raise ValueError(f"invalid engine override: {value!r}")
        try:
            overrides[key] = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON for engine override {key!r}") from exc
    return overrides


def chunk_plan(pending: list, first_size: int, size: int) -> list[list]:
    if first_size <= 0 or size <= 0:
        raise ValueError("chunk sizes must be positive")
    if not pending:
        return []
    chunks = [pending[:first_size]]
    chunks.extend(pending[index:index + size] for index in range(first_size, len(pending), size))
    return chunks


def run_pipeline(cfg: dict, config_bytes: bytes, out: Path, samples: list,
                 model_path: str, revision: str, data_root: str | None, dry_run: bool,
                 limit: int | None = None, subjects: list[str] | None = None,
                 engine: Engine | None = None, counter: TokenCounter | None = None,
                 invocation_args: dict | None = None,
                 engine_overrides: dict | None = None) -> int:
    if not dry_run and not cfg["prompt"]["verified"]:
        raise ValueError("prompt source is unverified")
    if limit is not None:
        samples = select_smoke(samples, limit)
    if len({sample.id for sample in samples}) != len(samples):
        raise ValueError("duplicate sample id")
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / cfg["run"]["raw_filename"]
    env_path = out / cfg["run"]["env_filename"]
    digest = hashlib.sha256(config_bytes).hexdigest()
    records = resume_state(raw_path, env_path, samples, digest, dry_run, limit, subjects)
    prior = json.loads(env_path.read_text(encoding="utf-8")) if env_path.exists() else None
    engine_overrides = engine_overrides or {}
    if prior and prior.get("engine_overrides", {}) != engine_overrides:
        raise ValueError("engine_overrides mismatch")
    if engine_overrides:
        print("WARNING: engine overrides:", engine_overrides)
    current_git = git_info()
    if prior and prior.get("invocations"):
        previous_git = prior["invocations"][-1].get("git", {})
        previous_commit = previous_git.get("commit", "unknown")
        if previous_commit != current_git["commit"]:
            print(f"WARNING: code commit changed since previous invocation: "
                  f"{previous_commit} -> {current_git['commit']}")
    env = prior or {"schema_version": cfg["schema"]["version"], "config": cfg,
                    "config_sha256": digest, "model": {"repo": cfg["model"]["repo_id"],
                    "revision": revision, "path": model_path},
                    "dataset": {"repo": cfg["dataset"]["repo_id"], "revision": cfg["dataset"]["revision"]},
                    "partial": limit is not None or subjects is not None, "dry_run": dry_run,
                    "limit": limit, "subjects": subjects,
                    "git_first": current_git, "engine_overrides": engine_overrides,
                    "invocations": []}
    started = time.monotonic()
    invocation = {"started_at": utc_now(), "finished_at": None, "duration_seconds": None,
                  "generate_seconds": 0,
                  "git": current_git,
                  "args": invocation_args or {"model_path": model_path, "revision": revision,
                                                 "data_root": data_root, "out": str(out),
                                                 "dry_run": dry_run, "limit": limit, "subjects": subjects}}
    sampler = None
    resolved_path = None
    record_start_count = len(records)
    done = {record["id"] for record in records}
    pending = [(ordinal, sample) for ordinal, sample in enumerate(samples) if sample.id not in done]
    env["invocations"].append(invocation)
    write_env(env_path, env)
    try:
        if dry_run:
            counter = counter or FakeTokenCounter()
            engine = engine or FakeEngine()
        elif pending:
            if counter is None or engine is None:
                if __package__:
                    from .inputs import HFTokenCounter, resolve_model_dir
                else:
                    from inputs import HFTokenCounter, resolve_model_dir
                resolved_path = Path(resolve_model_dir(model_path, revision, weights=True))
                if counter is None:
                    counter = HFTokenCounter(resolved_path, cfg)
            if resolved_path is not None:
                env["model"]["path"] = str(resolved_path)
            if __package__:
                from .envinfo import NvidiaSmiSampler, environment_info, package_versions
                from .vllm_engine import VLLMEngine
            else:
                from envinfo import NvidiaSmiSampler, environment_info, package_versions
                from vllm_engine import VLLMEngine
            if prior:
                current_packages = package_versions()
                if prior.get("environment", {}).get("packages") != current_packages:
                    print("WARNING: package versions changed since first invocation")
            sampler = NvidiaSmiSampler(cfg["run"]["vram_sample_interval_seconds"])
            sampler.start()
            engine = engine or VLLMEngine(cfg, resolved_path or Path(model_path), engine_overrides)
            if not prior and resolved_path is not None:
                env["environment"] = environment_info(cfg, resolved_path)
    except Exception:
        invocation["finished_at"] = utc_now()
        invocation["duration_seconds"] = round(time.monotonic() - started, 3)
        write_env(env_path, env)
        raise
    chunks = chunk_plan(pending, cfg["run"].get("first_chunk_size", cfg["run"]["chunk_size"]),
                        cfg["run"]["chunk_size"])
    max_input = (cfg["budget"]["max_model_len"] - cfg["budget"]["max_new_tokens"]
                 - cfg["budget"]["prompt_safety_margin"])
    mismatches = 0
    try:
        with raw_path.open("a", encoding="utf-8") as stream:
            for chunk_number, chunk in enumerate(chunks, 1):
                chunk_started = time.monotonic()
                prepared = []
                requests = []
                for ordinal, sample in chunk:
                    images = sample.images()
                    prompt = build_prompt(sample, cfg)
                    messages = build_messages(sample, cfg, images)
                    precomputed = counter.count(sample, prompt, messages, ordinal, cfg)
                    base = make_record(sample, cfg, synthetic=dry_run,
                                       image_indices=sorted(sample.image_indices),
                                       precomputed_prompt_tokens=precomputed)
                    if precomputed > max_input:
                        if hasattr(counter, "pop_prepared"):
                            counter.pop_prepared(sample.id)
                        prepared.append((base | {"status": "skip", "reason": "prompt_too_long",
                                                 "raw_text": "", "output_token_ids": [], "output_tokens": 0}, None))
                    else:
                        request = {"sample": sample, "ordinal": ordinal, "prompt": prompt,
                                   "messages": messages, "images": images, "precomputed": precomputed, "cfg": cfg}
                        if hasattr(counter, "pop_prepared"):
                            request["prepared"] = counter.pop_prepared(sample.id)
                        prepared.append((base, request))
                        requests.append(request)
                generation_started = time.monotonic()
                outputs = engine.generate(requests) if requests else []
                invocation["generate_seconds"] = round(
                    invocation.get("generate_seconds", 0) + time.monotonic() - generation_started, 3)
                if len(outputs) != len(requests):
                    raise AssertionError("engine returned wrong number of outputs")
                output_iter = iter(outputs)
                chunk_records = []
                chunk_mismatches = []
                for base, request in prepared:
                    if request is None:
                        record = base
                    else:
                        output = next(output_iter)
                        if isinstance(output, Exception):
                            record = base | {"status": "error", "error": repr(output),
                                             "raw_text": "", "output_token_ids": [], "output_tokens": 0}
                        else:
                            measured = output["num_prompt_tokens"]
                            if measured != request["precomputed"]:
                                mismatches += 1
                                chunk_mismatches.append((request["sample"].id,
                                                         request["precomputed"], measured))
                            record = base | {"status": "ok", "raw_text": output["raw_text"],
                                             "output_token_ids": output["output_token_ids"],
                                             "output_tokens": len(output["output_token_ids"]),
                                             "finish_reason": output["finish_reason"],
                                             "stop_reason": output["stop_reason"],
                                             "num_prompt_tokens": measured}
                    validate_record(record)
                    chunk_records.append(record)
                chunk_errors = [record for record in chunk_records if record["status"] == "error"]
                for error_record in chunk_errors[:1]:
                    error_text = error_record.get("error") or "unknown error"
                    print(f"first chunk exception: {error_text}")
                    output_index = next((index for index, item in enumerate(outputs)
                                         if isinstance(item, Exception)), None)
                    if output_index is not None and hasattr(outputs[output_index], "vllm_traceback"):
                        print(outputs[output_index].vllm_traceback)
                error_fraction = len(chunk_errors) / max(1, len(chunk_records))
                threshold = cfg["run"].get("abort_error_fraction", 1.0)
                if (chunk_number == 1 and record_start_count == 0 and chunk_errors) or error_fraction > threshold:
                    examples = [record["error"] for record in chunk_errors[:3]]
                    raise AssertionError(f"chunk {chunk_number} error gate: count={len(chunk_errors)}, "
                                         f"fraction={error_fraction:.3f}, threshold={threshold}; "
                                         f"examples={examples}")
                if chunk_mismatches:
                    differences = [measured - expected for _, expected, measured in chunk_mismatches]
                    raise AssertionError(f"precomputed prompt token mismatch in chunk {chunk_number}: "
                                         f"{chunk_mismatches[:20]}; difference min/max="
                                         f"{min(differences)}/{max(differences)}")
                for record in chunk_records:
                    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                records.extend(chunk_records)
                chunk_seconds = time.monotonic() - chunk_started
                chunk_tokens = sum(record["output_tokens"] or 0 for record in chunk_records)
                print(f"chunk {chunk_number}/{len(chunks)} done={len(records)} "
                      f"seconds={chunk_seconds:.3f} output_tokens/s="
                      f"{chunk_tokens / chunk_seconds:.2f} "
                      f"finish={dict(Counter(record['finish_reason'] for record in chunk_records if record['finish_reason']))} "
                      f"errors={len(chunk_errors)} "
                      f"elapsed={time.monotonic() - started:.3f}")
                if isinstance(engine, FakeEngine) and engine.stop_after_chunks == chunk_number:
                    raise RuntimeError(f"synthetic interruption after chunk {chunk_number}")
    finally:
        if sampler is not None:
            invocation["vram"] = {"nvidia_smi": sampler.stop(),
                                   "torch": engine.memory_report() if hasattr(engine, "memory_report") else "unknown"}
        if hasattr(engine, "describe"):
            invocation["engine"] = engine.describe()
        invocation["prompt_tokens_total"] = sum(record["num_prompt_tokens"] or 0
                                                for record in records[record_start_count:])
        invocation["output_tokens_total"] = sum(record["output_tokens"] or 0
                                                for record in records[record_start_count:])
        invocation["finished_at"] = utc_now()
        invocation["duration_seconds"] = round(time.monotonic() - started, 3)
        write_env(env_path, env)
        try:
            if __package__:
                from .metadata import write_run_metadata
            else:
                from metadata import write_run_metadata
            cmd_line = " ".join(sys.argv) if hasattr(sys, "argv") else None
            meta_path = write_run_metadata(out, cfg, records=records, env=env,
                                           invocation_args=invocation_args,
                                           command_line=cmd_line)
            print(f"run_metadata: written to {meta_path}")
        except Exception as meta_exc:
            print(f"WARNING: failed to write run_metadata.json: {meta_exc}")
    code_commits = list(dict.fromkeys(item.get("git", {}).get("commit", "unknown")
                                      for item in env["invocations"]))
    return print_summary(records, mismatches, code_commits, env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path")
    parser.add_argument("--revision")
    parser.add_argument("--data_root")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--subjects")
    parser.add_argument("--engine_override", action="append")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("limit must be nonnegative")
    config_bytes = args.config.read_bytes()
    cfg = yaml.safe_load(config_bytes)
    model_path = args.model_path or cfg["model"]["repo_id"]
    revision = args.revision or cfg["model"]["revision"]
    subjects = [part.strip() for part in args.subjects.split(",")] if args.subjects else None
    try:
        engine_overrides = parse_engine_overrides(args.engine_override)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.dry_run and not cfg["prompt"]["verified"]:
        parser.error("prompt source is unverified")
    samples = load_samples(cfg, args.data_root, subjects)
    return run_pipeline(cfg, config_bytes, args.out, samples, model_path, revision, args.data_root,
                        args.dry_run, args.limit, subjects,
                        engine_overrides=engine_overrides,
                        invocation_args={key: str(value) if isinstance(value, Path) else value
                                         for key, value in vars(args).items()})


if __name__ == "__main__":
    raise SystemExit(main())
