"""Metadata builder and validator for evaluation runs according to the 2026-09-21 specification.

This module outputs run_metadata.json containing execution linkage, model,
dataset, prompt template, sampling/generation settings, image processing,
execution environment, code/reproduction details, and output interpretation.
Personal computer absolute paths are strictly sanitized.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import yaml

OUTPUT_SPEC_VERSION = "2026-09-21"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_path(path_val: Any) -> Any:
    """Sanitize paths to remove personal computer absolute paths.

    Preserves repository-relative paths, RunPod workspace standard paths,
    or model hub snapshot identifiers.
    """
    if not isinstance(path_val, str):
        return path_val
    # If path contains common personal directories like /Users/... or /home/... (except /workspace)
    # convert them to relative or standard paths.
    p = path_val.replace("\\", "/")
    # If it refers to snapshots inside huggingface hub
    match = re.search(r"models--([^/]+)--([^/]+)/snapshots/([a-f0-9]+)", p)
    if match:
        repo_owner, repo_name, revision = match.groups()
        return f"{repo_owner}/{repo_name}@{revision}"

    # Match repository paths (promoted layout: <repo>/code/...)
    code_match = re.search(r"(?:^|/)code/(.+)$", p)
    if code_match:
        return "code/" + code_match.group(1)
    # Match repository paths
    for marker in ("baseline/src/", "assignment/src/", "repo/assignment/src/"):
        if marker in p:
            return "src/" + p.split(marker, 1)[1]
    for marker in ("baseline/", "assignment/", "repo/assignment/"):
        if marker in p:
            sub = p.split(marker, 1)[1]
            if sub.startswith("runs/"):
                return sub
            return sub

    # Strip macOS /Users/... prefix
    if p.startswith("/Users/"):
        parts = p.split("/")
        # /Users/<username>/...
        if len(parts) > 3:
            return "/".join(parts[3:])

    return p


def sanitize_dict_paths(data: Any) -> Any:
    """Recursively sanitize string paths inside dictionaries and lists."""
    if isinstance(data, dict):
        return {key: sanitize_dict_paths(value) for key, value in data.items()}
    if isinstance(data, list):
        return [sanitize_dict_paths(item) for item in data]
    if isinstance(data, str):
        return sanitize_path(data)
    return data


def compute_raw_stats(raw_path: Path) -> dict:
    """Read raw.jsonl and compute summary counts and statistics."""
    total_rows = 0
    statuses: Counter = Counter()
    finish_reasons: Counter = Counter()
    question_types: Counter = Counter()
    subjects: Counter = Counter()
    output_tokens_list: list[int] = []
    total_prompt_tokens = 0

    if raw_path.exists():
        with raw_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                total_rows += 1
                try:
                    record = json.loads(line)
                    statuses[record.get("status", "unknown")] += 1
                    fr = record.get("finish_reason")
                    if fr is not None:
                        finish_reasons[str(fr)] += 1
                    qt = record.get("question_type")
                    if qt is not None:
                        question_types[str(qt)] += 1
                    sub = record.get("subject")
                    if sub is not None:
                        subjects[str(sub)] += 1
                    ot = record.get("output_tokens")
                    if ot is not None and isinstance(ot, int):
                        output_tokens_list.append(ot)
                    pt = record.get("num_prompt_tokens")
                    if pt is not None and isinstance(pt, int):
                        total_prompt_tokens += pt
                except json.JSONDecodeError:
                    statuses["malformed_json"] += 1

    output_tokens_list.sort()
    med_tokens = (output_tokens_list[len(output_tokens_list) // 2]
                  if output_tokens_list else 0)

    return {
        "total_rows": total_rows,
        "statuses": dict(statuses),
        "finish_reasons": dict(finish_reasons),
        "question_types": dict(question_types),
        "subject_counts": dict(subjects),
        "output_tokens": {
            "total": sum(output_tokens_list),
            "min": output_tokens_list[0] if output_tokens_list else 0,
            "median": med_tokens,
            "max": output_tokens_list[-1] if output_tokens_list else 0,
            "count": len(output_tokens_list),
        },
        "total_prompt_tokens": total_prompt_tokens,
    }


def build_run_metadata(out_dir: Path, cfg: dict, env: dict | None = None,
                       raw_path: Path | None = None,
                       invocation_args: dict | None = None,
                       command_line: str | None = None) -> dict:
    """Construct complete run_metadata.json conforming to spec version 2026-09-21."""
    out_dir = Path(out_dir)
    raw_path = raw_path or (out_dir / cfg.get("run", {}).get("raw_filename", "raw.jsonl"))

    # Compute raw.jsonl checksum and metrics
    raw_exists = raw_path.exists()
    raw_bytes = raw_path.stat().st_size if raw_exists else 0
    raw_sha = sha256_file(raw_path) if raw_exists else None
    stats = compute_raw_stats(raw_path)

    # Invocation details from env or args
    invocations = (env or {}).get("invocations", [])
    first_invocation = invocations[0] if invocations else {}
    last_invocation = invocations[-1] if invocations else {}

    started_at = first_invocation.get("started_at") or utc_now()
    finished_at = last_invocation.get("finished_at") or utc_now()
    duration_seconds = sum(item.get("duration_seconds") or 0.0 for item in invocations)
    if not duration_seconds and last_invocation.get("duration_seconds"):
        duration_seconds = last_invocation.get("duration_seconds")

    # Execution command
    if command_line is None:
        cmd_args = last_invocation.get("args") or invocation_args or {}
        cmd_parts = [f"--{k} {v}" for k, v in cmd_args.items() if v is not None]
        command_line = f"python src/generate.py {' '.join(cmd_parts)}" if cmd_parts else "python src/generate.py"

    # Git details
    git_info = last_invocation.get("git") or (env or {}).get("git_first") or {}

    # Environment snapshot
    env_snapshot = (env or {}).get("environment", {})
    packages = env_snapshot.get("packages", cfg.get("environment", {}).get("expected_packages", {}))
    gpu_info = env_snapshot.get("gpu", {})
    python_ver = env_snapshot.get("python", sys.version.split()[0])
    model_snapshot_data = env_snapshot.get("model_snapshot", {})

    # Sanitize snapshot paths if present
    sanitized_snapshot_files = {}
    for rel_file, file_info in model_snapshot_data.get("files", {}).items():
        sanitized_snapshot_files[rel_file] = {
            "bytes": file_info.get("bytes"),
            "sha256": file_info.get("sha256"),
            "blob_id": file_info.get("blob_id"),
        }
    sanitized_model_snapshot = {
        "directory_name": model_snapshot_data.get("directory_name", cfg["model"]["revision"]),
        "files": sanitized_snapshot_files,
    }

    # Config digest
    config_sha256 = (env or {}).get("config_sha256")
    if not config_sha256:
        config_bytes = yaml.safe_dump(cfg, sort_keys=False).encode("utf-8")
        config_sha256 = hashlib.sha256(config_bytes).hexdigest()

    metadata: dict[str, Any] = {
        "output_spec_version": OUTPUT_SPEC_VERSION,
        "run": {
            "run_id": out_dir.name,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": round(duration_seconds, 3) if duration_seconds else None,
            "raw_file": {
                "filename": raw_path.name,
                "sha256": raw_sha,
                "bytes": raw_bytes,
                "row_count": stats["total_rows"],
            },
            "counts": {
                "total": stats["total_rows"],
                "ok": stats["statuses"].get("ok", 0),
                "skip": stats["statuses"].get("skip", 0),
                "error": stats["statuses"].get("error", 0),
            },
            "finish_reasons": stats["finish_reasons"],
            "question_types": stats["question_types"],
            "token_stats": stats["output_tokens"],
            "prompt_tokens_total": stats["total_prompt_tokens"],
        },
        "model": {
            "repo_id": cfg["model"]["repo_id"],
            "revision": cfg["model"]["revision"],
            "tokenizer_processor": {
                "repo_id": cfg["model"]["repo_id"],
                "revision": cfg["model"]["revision"],
            },
            "dtype": cfg["model"]["dtype"],
            "model_snapshot": sanitized_model_snapshot,
        },
        "dataset": {
            "repo_id": cfg["dataset"]["repo_id"],
            "revision": cfg["dataset"]["revision"],
            "split": cfg["dataset"]["split"],
            "expected_rows_per_subject": cfg["dataset"]["expected_rows_per_subject"],
            "expected_total_rows": cfg["dataset"]["expected_total_rows"],
            "subjects": cfg["dataset"]["subjects"],
            "loaded_subjects_count": len(stats["subject_counts"]) or len(cfg["dataset"]["subjects"]),
        },
        "prompt": {
            "source": {
                "repo": cfg["prompt"]["source"]["repo"],
                "path": cfg["prompt"]["source"]["path"],
                "commit": cfg["prompt"]["source"]["commit"],
                "license": cfg["prompt"]["source"]["license"],
            },
            "template": {
                "format_multiple_choice": "Question: {question}\\nOptions:\\n{options}\\nPlease select the correct answer from the options above.",
                "format_open": "Question: {question}",
                "instruction": "Please select the correct answer from the options above.",
            },
            "system_prompt": None,
            "user_prompt_structure": "Single user message with multimodal image objects followed by formatted text prompt.",
            "raw_prompt_stage": "before_chat_template",
            "chat_template_config": {
                "add_generation_prompt": True,
                "applied_by": "vllm / HF processor",
            },
        },
        "generation": {
            "do_sample": cfg["sampling"]["do_sample"],
            "temperature": cfg["sampling"]["temperature"],
            "top_p": cfg["sampling"]["top_p"],
            "top_k": cfg["sampling"]["top_k"],
            "repetition_penalty": cfg["sampling"]["repetition_penalty"],
            "presence_penalty": cfg["sampling"]["presence_penalty"],
            "frequency_penalty": 0.0,
            "seed": {
                "sampling_params_seed": cfg["sampling"]["seed"],
                "engine_seed": cfg["sampling"]["engine_seed"],
            },
            "max_new_tokens": cfg["budget"]["max_new_tokens"],
            "max_model_len": cfg["budget"]["max_model_len"],
            "stop_strings": [],
            "stop_token_ids": cfg["sampling"].get("stop_token_ids", []),
            "eos_handling": "Default tokenizer EOS tokens (<|im_end|>, <|endoftext|>), ignore_eos=False",
        },
        "image_processing": {
            "min_pixels": cfg["image"]["min_pixels"],
            "max_pixels": cfg["image"]["max_pixels"],
            "image_patch_size": cfg["image"]["image_patch_size"],
            "spatial_merge_size": cfg["image"]["spatial_merge_size"],
            "limit_mm_per_prompt": cfg["engine"]["limit_mm_per_prompt"],
            "image_selection_rule": "Each sample owns images corresponding to non-null image_N columns in the dataset row (image_1 to image_N).",
            "image_order_rule": "Images passed in ascending order of slot numbers [1, 2, ...]; matches raw.image_indices.",
            "preprocessing": "Qwen2-VL / Qwen3-VL processor smart_resize with patch_size=16 and spatial_merge_size=2; token grid factor=32.",
        },
        "environment": {
            "backend": {
                "name": "vllm",
                "version": packages.get("vllm", "unknown"),
            },
            "python_version": python_ver,
            "packages": packages,
            "cuda": {
                "cuda_header": gpu_info.get("cuda_header", "unknown"),
                "driver_version": gpu_info.get("driver", "unknown"),
            },
            "gpu": {
                "name": gpu_info.get("name", "unknown"),
                "count": 1,
                "memory_total": gpu_info.get("memory_total", "unknown"),
                "bf16_supported": gpu_info.get("bf16_supported", "unknown"),
            },
            "parallelism": {
                "pipeline_parallel_size": 1,
                "tensor_parallel_size": 1,
                "data_parallel_size": 1,
            },
            "batching": {
                "gpu_memory_utilization": cfg["engine"]["gpu_memory_utilization"],
                "max_num_batched_tokens": cfg["engine"]["max_num_batched_tokens"],
                "max_num_seqs": cfg["engine"]["max_num_seqs"],
                "chunk_size": cfg["run"]["chunk_size"],
                "first_chunk_size": cfg["run"].get("first_chunk_size", cfg["run"]["chunk_size"]),
            },
            "env_vars": sanitize_dict_paths(env_snapshot.get("env_vars", cfg["environment"].get("env_vars", {}))),
        },
        "code_and_reproduction": {
            "git": {
                "commit": git_info.get("commit", "unknown"),
                "dirty": git_info.get("dirty", False),
            },
            "execution_command": sanitize_path(command_line),
            "config": {
                "relative_path": "src/config.yaml",
                "sha256": config_sha256,
                "values": sanitize_dict_paths(cfg),
            },
        },
        "output_interpretation": {
            "decoding": {
                "skip_special_tokens": cfg["sampling"].get("skip_special_tokens", True),
            },
            "output_token_ids": {
                "includes_stop_tokens": True,
                "definition": "Generated token IDs returned by vLLM engine (CompletionOutput.token_ids), excluding prompt tokens. When finish_reason is 'stop', the sequence ends with the terminating EOS token (<|im_end|>, id 151645 for this model), which is counted in output_tokens; when finish_reason is 'length', no EOS token is present.",
            },
            "num_prompt_tokens": {
                "measurement": "Engine input token count measured by vLLM, including expanded multimodal image tokens.",
            },
            "finish_reason_mapping": {
                "stop": "Natural termination by EOS or stop token.",
                "length": "Generation reached max_new_tokens budget limit.",
                "null": "Generation failed, skipped, or undetermined.",
            },
        },
    }

    return metadata


def write_run_metadata(out_dir: Path, cfg: dict, records: list[dict] | None = None,
                       env: dict | None = None,
                       invocation_args: dict | None = None,
                       command_line: str | None = None) -> Path:
    """Build and write run_metadata.json to out_dir atomically."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = build_run_metadata(out_dir, cfg, env=env,
                                  invocation_args=invocation_args,
                                  command_line=command_line)

    dest = out_dir / "run_metadata.json"
    tmp = dest.with_name("run_metadata.json.tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(dest)
    return dest


def validate_run_metadata(metadata_path: Path, raw_path: Path | None = None) -> tuple[int, list[str]]:
    """Validate run_metadata.json conformance against spec and against raw.jsonl."""
    errors = []
    if not metadata_path.exists():
        return 1, [f"missing metadata file: {metadata_path}"]

    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return 1, [f"invalid json in metadata: {exc}"]

    # Check specification version
    spec_ver = data.get("output_spec_version")
    if spec_ver != OUTPUT_SPEC_VERSION:
        errors.append(f"output_spec_version {spec_ver!r} != {OUTPUT_SPEC_VERSION!r}")

    # Required top-level sections
    required_sections = ("run", "model", "dataset", "prompt", "generation",
                         "image_processing", "environment", "code_and_reproduction",
                         "output_interpretation")
    for sec in required_sections:
        if sec not in data:
            errors.append(f"missing top-level section: {sec}")

    # Check model info
    model = data.get("model", {})
    if model.get("repo_id") != "Qwen/Qwen3-VL-4B-Instruct":
        errors.append(f"unexpected model repo_id: {model.get('repo_id')}")
    if model.get("revision") != "ebb281ec70b05090aa6165b016eac8ec08e71b17":
        errors.append(f"unexpected model revision: {model.get('revision')}")

    # Check dataset info
    ds = data.get("dataset", {})
    if ds.get("repo_id") != "MMMU/MMMU":
        errors.append(f"unexpected dataset repo_id: {ds.get('repo_id')}")
    if ds.get("revision") != "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68":
        errors.append(f"unexpected dataset revision: {ds.get('revision')}")

    # Check raw_file connection if raw_path is available
    if raw_path is None and "raw_file" in data.get("run", {}):
        raw_path = metadata_path.parent / data["run"]["raw_file"].get("filename", "raw.jsonl")

    if raw_path and raw_path.exists():
        actual_sha = sha256_file(raw_path)
        meta_sha = data.get("run", {}).get("raw_file", {}).get("sha256")
        if actual_sha != meta_sha:
            errors.append(f"raw_file sha256 mismatch: actual={actual_sha} metadata={meta_sha}")
        raw_bytes = raw_path.stat().st_size
        meta_bytes = data.get("run", {}).get("raw_file", {}).get("bytes")
        if raw_bytes != meta_bytes:
            errors.append(f"raw_file bytes mismatch: actual={raw_bytes} metadata={meta_bytes}")

    # Check for personal computer paths in serialized text
    text = metadata_path.read_text(encoding="utf-8")
    if "/Users/" in text:
        errors.append("metadata contains personal path prefix '/Users/'")

    return (1 if errors else 0), errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or validate run_metadata.json")
    parser.add_argument("--run", type=Path, help="Run directory containing raw.jsonl and optionally env.json")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--validate_only", action="store_true", help="Only validate existing run_metadata.json")
    args = parser.parse_args(argv)

    if not args.run or not args.run.exists():
        parser.error("--run must specify an existing run directory")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    meta_path = args.run / "run_metadata.json"

    if not args.validate_only:
        env_path = args.run / "env.json"
        env = json.loads(env_path.read_text(encoding="utf-8")) if env_path.exists() else None
        print(f"Building {meta_path}...")
        write_run_metadata(args.run, cfg, env=env)
        print(f"Successfully generated {meta_path}")

    code, errors = validate_run_metadata(meta_path)
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return code
    print(f"VALIDATION PASSED: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
