"""Read-only environment and dataset checks for a pinned evaluation run."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import importlib
from importlib import metadata
import inspect
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time
import traceback

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import yaml


SECTION_NAMES = ("system", "packages", "vllm_import", "gpu", "qwen_vl_utils", "model", "data")
MARKER = re.compile(r"<image\s*(\d+)>")
IMAGE_COLUMN = re.compile(r"image_(\d+)$")
NETWORK_ERRORS = ("connection", "connecterror", "timeout", "offline", "network", "dns", "name resolution")


def check(status, name, **details):
    if "status" in details or "name" in details:
        raise ValueError("check details cannot contain reserved keys: status/name")
    return {"status": status, "name": name, **details}


def installed_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def parse_requirements(path):
    """Return active, exact pins; Requirement handles comments and PEP 508 markers."""
    pins = []
    for number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith(("#", "-")):
            continue
        requirement = Requirement(re.split(r"\s+#", line, maxsplit=1)[0].strip())
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        if any(spec.operator == "==" for spec in requirement.specifier):
            pins.append({"line": number, "raw": line, "requirement": requirement})
    return pins


def version_satisfies(requirement, version):
    return version is not None and requirement.specifier.contains(version, prereleases=True)


def run_command(argv, timeout, env=None):
    started = time.monotonic()
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env, check=False)
        return {"returncode": result.returncode, "elapsed_seconds": round(time.monotonic() - started, 3),
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "elapsed_seconds": round(time.monotonic() - started, 3),
                "stdout": "", "stderr": str(exc), "error_type": type(exc).__name__}


def system_section(config, args):
    expected = config["environment"]["env_vars"]
    names = sorted(set(expected) | {"HF_HOME", "HF_HUB_CACHE", "CUDA_VISIBLE_DEVICES"})
    variables = {name: os.environ.get(name, "unset") for name in names}
    pretty_name = "unknown"
    try:
        source = Path("/etc/os-release").read_text(encoding="utf-8")
        match = re.search(r'^PRETTY_NAME=(.*)$', source, re.MULTILINE)
        if match:
            pretty_name = match.group(1).strip().strip('"')
    except OSError:
        pass
    checks = [check("INFO", "system_info", python=platform.python_version(), platform=platform.platform(),
                    os_pretty_name=pretty_name, env_vars=variables)]
    for name, value in expected.items():
        checks.append(check("PASS" if variables[name] == str(value) else "WARN", "env_var",
                            key=name, expected=str(value), actual=variables[name]))
    return checks


def packages_section(config, args):
    checks = []
    expected = config["environment"]["expected_packages"]
    for name, wanted in expected.items():
        actual = platform.python_version() if name == "python" else installed_version(name)
        checks.append(check("PASS" if actual == str(wanted) else "FAIL", "expected_version",
                            package=name, expected=str(wanted), actual=actual or "unknown"))
    pins = parse_requirements(args.requirements)
    mismatches = []
    for pin in pins:
        req = pin["requirement"]
        actual = installed_version(req.name)
        if not version_satisfies(req, actual):
            mismatches.append({"line": pin["line"], "requirement": pin["raw"], "installed": actual or "unknown"})
    checks.append(check("FAIL" if mismatches else "PASS", "requirements_pins", checked=len(pins), mismatches=mismatches))
    names = ("vllm", "torch", "transformers", "tokenizers", "openai", "qwen-vl-utils", "datasets",
             "huggingface_hub", "pillow", "pyyaml", "numpy", "flashinfer", "xformers")
    checks.append(check("INFO", "installed_versions", versions={name: installed_version(name) or "unknown" for name in names}))
    dependencies = []
    try:
        requirements = metadata.requires("vllm") or []
    except metadata.PackageNotFoundError:
        requirements = []
    watched = {"openai", "transformers", "tokenizers", "torch", "qwen-vl-utils"}
    for raw in requirements:
        req = Requirement(raw)
        name = canonicalize_name(req.name)
        if name not in watched and "qwen" not in name:
            continue
        applicable = req.marker is None or req.marker.evaluate({"extra": ""})
        actual = installed_version(req.name)
        satisfied = version_satisfies(req, actual) if applicable else None
        dependencies.append({"raw": raw, "installed": actual or "unknown", "applicable": applicable,
                             "satisfied": satisfied})
    checks.append(check("WARN" if any(d["satisfied"] is False for d in dependencies) else "INFO",
                        "vllm_dependency_specifiers", dependencies=dependencies))
    result = run_command([sys.executable, "-m", "pip", "check"], 120)
    checks.append(check("PASS" if result["returncode"] == 0 else "FAIL", "pip_check", **result))
    return checks


def vllm_import_section(config, args):
    env = dict(os.environ, **{key: str(value) for key, value in config["environment"]["env_vars"].items()})
    result = run_command([sys.executable, "-c", "from vllm import LLM"], 300, env=env)
    result["stderr_tail"] = "\n".join(result.pop("stderr").splitlines()[-60:])
    return [check("PASS" if result["returncode"] == 0 else "FAIL", "vllm_import", **result)]


def gpu_section(config, args):
    query = run_command(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
                         "--format=csv,noheader"], 30)
    if query["returncode"] != 0 or not query["stdout"].strip():
        return [check("SKIP", "gpu_unavailable", reason=query["stderr"] or "No GPU reported by nvidia-smi")]
    checks = [check("INFO", "nvidia_smi_query", **query)]
    header = run_command(["nvidia-smi"], 30)
    match = re.search(r"CUDA Version:\s*([\d.]+)", header["stdout"])
    checks.append(check("INFO", "nvidia_smi_cuda", version=match.group(1) if match else "unknown"))
    for line in query["stdout"].splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            checks.append(check("WARN", "gpu_query_parse", raw=line))
            continue
        try:
            major = int(fields[3].split(".")[0])
        except ValueError:
            major = None
        checks.append(check("FAIL" if major is not None and major < config["environment"]["min_driver_major"]
                            else "WARN" if major is None else "PASS", "gpu_driver", gpu_name=fields[0],
                            memory_total=fields[1], memory_used=fields[2], driver_version=fields[3],
                            min_driver_major=config["environment"]["min_driver_major"]))
    try:
        import torch
        available = torch.cuda.is_available()
        checks.append(check("INFO", "torch_cuda", available=available,
                            device_name=torch.cuda.get_device_name(0) if available else "unknown",
                            cuda_version=torch.version.cuda or "unknown",
                            bf16_supported=torch.cuda.is_bf16_supported() if available else "unknown"))
    except Exception:
        checks.append(check("WARN", "torch_cuda", traceback=traceback.format_exc()))
    return checks


def qwen_vl_utils_section(config, args):
    if installed_version("qwen-vl-utils") is None:
        return [check("FAIL", "qwen_vl_utils", reason="package not installed")]
    module = importlib.import_module("qwen_vl_utils")
    vision = importlib.import_module("qwen_vl_utils.vision_process")
    process_signature = inspect.signature(module.process_vision_info)
    constants = {name: value for name, value in vars(vision).items()
                 if type(value) is int and any(part in name for part in
                 ("FACTOR", "PIXELS", "PATCH", "MERGE", "TOKEN", "RATIO", "SIZE"))}
    smart_resize = getattr(module, "smart_resize", None) or getattr(vision, "smart_resize", None)
    return [check("PASS" if "image_patch_size" in process_signature.parameters else "FAIL", "qwen_vl_utils",
                  process_vision_info=str(process_signature),
                  smart_resize=str(inspect.signature(smart_resize)) if smart_resize else "unknown",
                  constants=constants)]


def snapshot_details(path, config):
    path = Path(path)
    checks = []
    model_config = path / "config.json"
    if model_config.exists():
        data = json.loads(model_config.read_text(encoding="utf-8"))
        vision = data.get("vision_config") or {}
        text = data.get("text_config") or {}
        dtype_source = next((source for source, value in
                             (("torch_dtype", data.get("torch_dtype")), ("dtype", data.get("dtype")),
                              ("text_config.torch_dtype", text.get("torch_dtype")),
                              ("text_config.dtype", text.get("dtype"))) if value is not None), "unknown")
        dtype = data.get("torch_dtype") or data.get("dtype") or text.get("torch_dtype") or text.get("dtype") or "unknown"
        checks.append(check("PASS" if dtype == config["model"]["dtype"] else "FAIL", "model_config",
                            dtype=dtype, dtype_source=dtype_source,
                            max_position_embeddings=text.get("max_position_embeddings", "unknown"),
                            vision={key: vision[key] for key in vision if any(term in key for term in
                                    ("patch_size", "spatial_merge_size"))}))
    else:
        checks.append(check("FAIL", "model_config", reason="config.json missing"))
    for name in ("generation_config.json", "preprocessor_config.json"):
        file = path / name
        checks.append(check("INFO" if file.exists() else "WARN", name,
                            content=json.loads(file.read_text(encoding="utf-8")) if file.exists() else "unknown"))
    video = path / "video_preprocessor_config.json"
    checks.append(check("INFO" if video.exists() else "WARN", "video_preprocessor_config.json",
                        keys=list(json.loads(video.read_text(encoding="utf-8"))) if video.exists() else "unknown"))
    tokenizer = path / "tokenizer_config.json"
    has_tokenizer_template = tokenizer.exists() and bool(json.loads(tokenizer.read_text(encoding="utf-8")).get("chat_template"))
    checks.append(check("INFO", "chat_template", tokenizer_config=has_tokenizer_template,
                        chat_template_json=(path / "chat_template.json").exists(),
                        chat_template_jinja=(path / "chat_template.jinja").exists()))
    return checks


def model_section(config, args):
    path = Path(args.model_path)
    revision = args.revision
    checks = []
    if revision != config["model"]["revision"]:
        checks.append(check("WARN", "revision_override", configured=config["model"]["revision"], actual=revision))
    if path.is_dir():
        checks.extend(snapshot_details(path, config))
        return checks
    if args.download:
        from huggingface_hub import snapshot_download
        snapshot = Path(snapshot_download(repo_id=args.model_path, revision=revision))
        checks.append(check("PASS" if snapshot.name == revision else "FAIL", "snapshot_revision",
                            returned_path=str(snapshot), expected=revision))
        files = [{"path": str(file.relative_to(snapshot)), "bytes": file.stat().st_size}
                 for file in snapshot.rglob("*") if file.is_file()]
        checks.append(check("INFO", "snapshot_files", files=files,
                            total_gb=round(sum(file["bytes"] for file in files) / 1e9, 3)))
        checks.extend(snapshot_details(snapshot, config))
    else:
        from huggingface_hub import HfApi
        try:
            sha = HfApi().model_info(args.model_path, revision=revision).sha
        except Exception as exc:
            if is_network_error(exc):
                checks.append(check("SKIP", "model_revision", reason=str(exc)))
                return checks
            raise
        checks.append(check("PASS" if sha == revision else "FAIL", "model_revision", expected=revision, actual=sha))
    return checks


def is_network_error(exc):
    return any(term in (type(exc).__name__ + " " + str(exc)).lower() for term in NETWORK_ERRORS)


def parsed_options(value):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value), True
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            return value, False
    return value, None


def marker_comparison(row, image_columns):
    options, _ = parsed_options(row.get("options"))
    option_markers = {int(number) for number in MARKER.findall(str(options))}
    markers = {int(number) for number in MARKER.findall(str(row.get("question") or ""))} | option_markers
    images = {int(IMAGE_COLUMN.fullmatch(name).group(1)) for name in image_columns if row.get(name) is not None}
    categories = []
    if markers == images:
        categories.append("match")
    if markers - images:
        categories.append("marker_without_image")
    if images - markers:
        categories.append("image_without_marker")
    return categories, bool(option_markers), markers, images


def inspect_subject(dataset, subject):
    columns = list(dataset.column_names)
    image_columns = [name for name in columns if IMAGE_COLUMN.fullmatch(name)]
    option_types = Counter()
    parse_counts = Counter()
    option_counts = Counter()
    question_types = Counter()
    answer_formats = {}
    nonnull_images = Counter()
    comparisons = Counter()
    comparison_examples = {name: [] for name in ("match", "marker_without_image", "image_without_marker")}
    options_with_marker = 0
    ids = []
    pixels = []
    samples = []
    for row in dataset:
        row_id = str(row.get("id", "unknown"))
        ids.append(row_id)
        raw_options = row.get("options")
        option_types[type(raw_options).__name__] += 1
        options, parsed = parsed_options(raw_options)
        if parsed is not None:
            parse_counts["success" if parsed else "failure"] += 1
        if parsed and isinstance(options, (list, tuple, dict)):
            option_counts[len(options)] += 1
        kind = str(row.get("question_type", "unknown"))
        question_types[kind] += 1
        bucket = answer_formats.setdefault(kind, {"single_letter": 0, "other": 0, "examples": []})
        answer = row.get("answer")
        if isinstance(answer, str) and re.fullmatch(r"[A-Z]", answer):
            bucket["single_letter"] += 1
        else:
            bucket["other"] += 1
        if len(bucket["examples"]) < 5:
            bucket["examples"].append(repr(answer))
        present = [name for name in image_columns if row.get(name) is not None]
        nonnull_images[len(present)] += 1
        categories, option_marker, markers, images = marker_comparison(row, image_columns)
        options_with_marker += int(option_marker)
        for category in categories:
            comparisons[category] += 1
            if len(comparison_examples[category]) < 5:
                comparison_examples[category].append({"id": row_id, "markers": sorted(markers), "images": sorted(images)})
        for name in present:
            image = row[name]
            if hasattr(image, "size") and isinstance(image.size, tuple):
                pixels.append(image.size[0] * image.size[1])
            elif isinstance(image, dict) and hasattr(image.get("image"), "size"):
                size = image["image"].size
                pixels.append(size[0] * size[1])
        if subject == "Accounting" and len(samples) < 2:
            samples.append({"id": row_id, "question_type": kind, "options_repr": repr(raw_options),
                            "answer": answer, "question_prefix": str(row.get("question") or "")[:200]})
    for kind, bucket in answer_formats.items():
        if kind.lower() in ("multiple-choice", "multiple_choice", "mcq"):
            bucket["valid_multiple_choice"] = bucket["other"] == 0
        else:
            bucket["open_answer_examples"] = bucket.pop("examples")
    return {"rows": len(dataset), "columns": columns, "options_types": dict(option_types),
            "options_literal_eval": dict(parse_counts), "parsed_option_counts": dict(option_counts),
            "question_types": dict(question_types), "answer_formats": answer_formats,
            "image_columns": image_columns, "non_null_image_counts": dict(nonnull_images),
            "marker_comparison": dict(comparisons), "marker_examples": comparison_examples,
            "options_with_marker": options_with_marker, "ids_unique": len(ids) == len(set(ids)),
            "id_examples": ids[:5], "image_pixels": {"min": min(pixels), "median": statistics.median(pixels),
            "max": max(pixels)} if pixels else "unknown", "samples": samples}


def data_section(config, args):
    from datasets import load_dataset
    setting = config["dataset"]
    subjects = args.subjects or setting["subjects"]
    checks = []
    total = 0
    first_columns = None
    for subject in subjects:
        try:
            dataset = load_dataset(setting["repo_id"], subject, split=setting["split"],
                                   revision=setting["revision"], cache_dir=args.data_root)
            details = inspect_subject(dataset, subject)
            total += details["rows"]
            columns_equal = first_columns is None or details["columns"] == first_columns
            if first_columns is None:
                first_columns = details["columns"]
            status = "PASS" if details["rows"] == setting["expected_rows_per_subject"] and columns_equal and details["ids_unique"] else "FAIL"
            checks.append(check(status, "subject", subject=subject, columns_match=columns_equal, **details))
        except Exception as exc:
            checks.append(check("SKIP" if is_network_error(exc) else "FAIL", "subject", subject=subject,
                                reason=str(exc), traceback=traceback.format_exc()))
    expected = setting["expected_total_rows"] if len(subjects) == len(setting["subjects"]) else setting["expected_rows_per_subject"] * len(subjects)
    skipped = any(item["status"] == "SKIP" for item in checks)
    checks.append(check("SKIP" if skipped else "PASS" if total == expected else "FAIL", "total_rows",
                        expected=expected, actual=total, selected_subjects=len(subjects)))
    return checks


def run_sections(config, args, runners=None):
    runners = runners or {"system": system_section, "packages": packages_section, "vllm_import": vllm_import_section,
                          "gpu": gpu_section, "qwen_vl_utils": qwen_vl_utils_section, "model": model_section, "data": data_section}
    outcomes = {}
    for name in args.sections:
        try:
            outcomes[name] = runners[name](config, args)
        except Exception:
            outcomes[name] = [check("FAIL", "section_exception", traceback=traceback.format_exc())]
    return outcomes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model_path")
    parser.add_argument("--revision")
    parser.add_argument("--data_root")
    parser.add_argument("--requirements", type=Path, default=Path(__file__).with_name("requirements.txt"))
    parser.add_argument("--sections", default="all", help="Comma-separated names or all")
    parser.add_argument("--subjects", help="Comma-separated subject names")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.model_path = args.model_path or config["model"]["repo_id"]
    args.revision = args.revision or config["model"]["revision"]
    args.sections = list(SECTION_NAMES) if args.sections == "all" else [part.strip() for part in args.sections.split(",")]
    invalid = set(args.sections) - set(SECTION_NAMES)
    if invalid:
        parser.error("unknown sections: " + ", ".join(sorted(invalid)))
    args.subjects = [part.strip() for part in args.subjects.split(",")] if args.subjects else None
    if args.subjects and set(args.subjects) - set(config["dataset"]["subjects"]):
        parser.error("unknown subjects: " + ", ".join(sorted(set(args.subjects) - set(config["dataset"]["subjects"]))))
    outcomes = run_sections(config, args)
    counts = Counter(item["status"] for checks in outcomes.values() for item in checks)
    report = {"sections": outcomes, "counts": dict(counts), "config": str(args.config),
              "model_path": args.model_path, "revision": args.revision,
              "selected_subjects": args.subjects or config["dataset"]["subjects"]}
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "env_check.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Report: {destination}")
    for section, checks in outcomes.items():
        summary = Counter(item["status"] for item in checks)
        print(f"{section}: " + ", ".join(f"{key}={summary[key]}" for key in ("PASS", "FAIL", "WARN", "INFO", "SKIP") if summary[key]))
    print("Total: " + ", ".join(f"{key}={counts[key]}" for key in ("PASS", "FAIL", "WARN", "INFO", "SKIP") if counts[key]))
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
