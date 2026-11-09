"""CPU verification of the inputs and token budget for every selected sample."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import statistics

import yaml

if __package__:
    from .common import build_messages, load_samples, select_smoke
    from .inputs import load_processor, prepare_inputs, resolve_model_dir
else:
    from common import build_messages, load_samples, select_smoke
    from inputs import load_processor, prepare_inputs, resolve_model_dir


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _percentile(sorted_values: list[int], fraction: float):
    return sorted_values[max(0, math.ceil(len(sorted_values) * fraction) - 1)]


def _distribution(values: list[int]) -> dict:
    if not values:
        return {name: "unknown" for name in ("min", "median", "p90", "p99", "max")}
    ordered = sorted(values)
    return {"min": ordered[0], "median": statistics.median(ordered),
            "p90": _percentile(ordered, 0.9), "p99": _percentile(ordered, 0.99),
            "max": ordered[-1]}


def inspect_sample(sample, cfg: dict, processor) -> tuple[dict, list[str], dict]:
    messages = build_messages(sample, cfg, sample.images())
    originals = [part["image"].size for part in messages[0]["content"]
                 if part["type"] == "image"]
    prepared, resized = prepare_inputs(messages, processor)
    images = prepared["multi_modal_data"].get("image")
    prompt = prepared["prompt"]
    no_resize = processor(text=[prompt], images=images, do_resize=False)
    default_resize = processor(text=[prompt], images=images, do_resize=True)
    count = len(no_resize["input_ids"][0])
    default_count = len(default_resize["input_ids"][0])
    grids = no_resize.get("image_grid_thw")
    grids = grids.tolist() if grids is not None else []
    patch = processor.image_processor.patch_size
    merge = processor.image_processor.merge_size
    factor = patch * merge
    errors = []
    if patch != cfg["image"]["image_patch_size"]:
        errors.append(f"patch_size: {patch} != {cfg['image']['image_patch_size']}")
    if merge != cfg["image"]["spatial_merge_size"]:
        errors.append(f"merge_size: {merge} != {cfg['image']['spatial_merge_size']}")
    if len(originals) != len(resized) or len(resized) != len(grids):
        errors.append(f"image count: original={len(originals)}, resized={len(resized)}, grid={len(grids)}")
    details = []
    for position, (original, size, grid) in enumerate(zip(originals, resized, grids), 1):
        width, height = size
        grid_tokens = math.prod(grid) // (merge * merge)
        formula_tokens = (height // factor) * (width // factor)
        pixels = width * height
        if width % factor or height % factor:
            errors.append(f"image {position}: dimensions {size} are not divisible by {factor}")
        if grid_tokens != formula_tokens or grid != [1, height // patch, width // patch]:
            errors.append(f"image {position}: grid {grid}, formula {formula_tokens}, grid tokens {grid_tokens}")
        if not cfg["image"]["min_pixels"] <= pixels <= cfg["image"]["max_pixels"]:
            errors.append(f"image {position}: {pixels} pixels outside configured range; rounding needs review")
        details.append({"index": sample.image_indices[position - 1],
                        "original_size": list(original), "resized_size": list(size),
                        "pixels": pixels, "image_grid_thw": grid,
                        "visual_tokens": grid_tokens})
    if count != default_count:
        errors.append(f"token count: do_resize=False {count}, do_resize=True {default_count}")
    max_input = (cfg["budget"]["max_model_len"] - cfg["budget"]["max_new_tokens"]
                 - cfg["budget"]["prompt_safety_margin"])
    row = {"id": sample.id, "subject": sample.subject, "question_type": sample.question_type,
           "images": details, "visual_tokens": sum(item["visual_tokens"] for item in details),
           "total_tokens": count, "default_resize_tokens": default_count,
           "exceeds": count > max_input}
    return row, errors, _plain(prepared["mm_processor_kwargs"])


def run(cfg: dict, config_bytes: bytes, out: Path, model_path: str,
        revision: str, data_root: str | None, subjects: list[str] | None,
        limit: int | None) -> int:
    model_dir = resolve_model_dir(model_path, revision, weights=False)
    processor = load_processor(model_dir, cfg)
    samples = load_samples(cfg, data_root, subjects)
    if limit is not None:
        samples = select_smoke(samples, limit)
    rows = []
    failures = []
    video_observed = {}
    for sample in samples:
        try:
            row, problems, video_kwargs = inspect_sample(sample, cfg, processor)
            rows.append(row)
            video_observed[json.dumps(video_kwargs, ensure_ascii=False, sort_keys=True)] = video_kwargs
            failures.extend(f"{sample.id}: {problem}" for problem in problems)
        except Exception as exc:
            failures.append(f"{sample.id}: {exc!r}")
    budget = cfg["budget"]
    max_input = budget["max_model_len"] - budget["max_new_tokens"] - budget["prompt_safety_margin"]
    counts = [row["total_tokens"] for row in rows]
    over = [{"id": row["id"], "image_count": len(row["images"]),
             "total_tokens": row["total_tokens"], "over_by": row["total_tokens"] - max_input}
            for row in rows if row["exceeds"]]
    by_images = defaultdict(list)
    for row in rows:
        by_images[len(row["images"])].append(row["total_tokens"])
    image_counts = {str(number): {"count": len(values), "mean_total_tokens": statistics.mean(values)}
                    for number, values in sorted(by_images.items())}
    lengths = (12288, 16384, 20480, 24576, 32768)
    outputs = (4096, 8192, 16384)
    summary = {
        "processed": len(rows), "selected": len(samples),
        "partial": len(samples) != cfg["dataset"]["expected_total_rows"],
        "total_tokens": _distribution(counts), "by_image_count": image_counts,
        "exceeds_count": len(over), "exceeds": over,
        "max_model_len_for_zero_skips": (max(counts) + budget["max_new_tokens"]
                                          + budget["prompt_safety_margin"] if counts else "unknown"),
        "skips_by_max_model_len": {
            str(length): sum(value > length - budget["max_new_tokens"] - budget["prompt_safety_margin"]
                             for value in counts) for length in lengths},
        "skips_by_max_new_tokens": {
            str(output): sum(value > budget["max_model_len"] - output - budget["prompt_safety_margin"]
                             for value in counts) for output in outputs},
        "validation_failures": failures,
    }
    result = {
        "meta": {"transformers": _version("transformers"), "tokenizers": _version("tokenizers"),
                 "qwen-vl-utils": _version("qwen-vl-utils"), "torch": _version("torch"),
                 "Pillow": _version("Pillow"),
                 "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                 "model_revision_dirname": model_dir.name, "max_input": max_input},
        "samples": rows, "summary": summary,
        "video_kwargs_observed": [video_observed[key] for key in sorted(video_observed)],
    }
    out.mkdir(parents=True, exist_ok=True)
    destination = out / "preflight.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"processed: {len(rows)}/{len(samples)}; partial: {summary['partial']}")
    print("total_tokens:", summary["total_tokens"])
    print("by_image_count:", image_counts)
    print("exceeds:", len(over))
    for item in over:
        print(f"  {item['id']} images={item['image_count']} tokens={item['total_tokens']} over={item['over_by']}")
    print("max_model_len_for_zero_skips:", summary["max_model_len_for_zero_skips"])
    print("skips_by_max_model_len:", summary["skips_by_max_model_len"])
    print("skips_by_max_new_tokens:", summary["skips_by_max_new_tokens"])
    print("video_kwargs_observed:", result["video_kwargs_observed"])
    print("validation_failures:", len(failures))
    for failure in failures:
        print("  FAIL", failure)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model_path")
    parser.add_argument("--revision")
    parser.add_argument("--data_root")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--subjects")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("limit must be nonnegative")
    config_bytes = args.config.read_bytes()
    cfg = yaml.safe_load(config_bytes)
    subjects = [part.strip() for part in args.subjects.split(",")] if args.subjects else None
    return run(cfg, config_bytes, args.out, args.model_path or cfg["model"]["repo_id"],
               args.revision or cfg["model"]["revision"], args.data_root, subjects, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
