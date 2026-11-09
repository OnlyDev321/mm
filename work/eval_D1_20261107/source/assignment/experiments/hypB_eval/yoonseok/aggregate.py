"""Offline paired summaries from complete 900-item raw and v2 score folders."""
import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import sys
from verify_code import verify

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


SCORER = module("hypb_scorer", "code/scoring/evaluate.py")
REPETITION = module("hypb_repetition", "assignment/experiments/repetition/analyze.py")
CATEGORIES = module("hypb_categories", "assignment/experiments/gap_checks/categories.py").SUBJECT_TO_CATEGORY
MARKERS = re.compile(r"\b(?:wait|perhaps|alternatively|actually|however|reconsider|re-examine|re-evaluate)\b", re.I)
M0_SHA = "ef23f0c49d9b1ae6c62b9625fbd52cce474a0c035c1a644cfa737e0967daa313"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def raw_rows(path):
    path = Path(path)
    data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    rows = [json.loads(line) for line in data.decode("utf-8-sig").splitlines() if line.strip()]
    SCORER.validate_raw(rows)
    require({r["max_new_tokens"] for r in rows} == {8192}, "expected the frozen 8192 output budget")
    return rows, hashlib.sha256(data).hexdigest()


def rule_scores(rows):
    summary, details = SCORER.mmmu_diagnostic(rows, SCORER.load_parsers())
    summary["serialized_gold_adaptation_count"] = len(summary.pop("serialized_gold_adaptations"))
    return summary, {r["id"]: r for r in details}


def score_rows(path, rows, digest):
    path = Path(path)
    compact = not (path / "manifest.json").exists()
    manifest_path = path / ("scoring_manifest.json" if compact else "manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["policy_id"] == SCORER.POLICY, "v2 policy mismatch")
    require(manifest["source_sha256"] == digest, "scored raw hash mismatch")
    scored = SCORER.read_rows(path / ("item_results.jsonl" if compact else "final_results.jsonl"))
    SCORER.unique(scored, "v2 scores")
    by_id = {r["id"]: r for r in scored}
    require(set(by_id) == {r["id"] for r in rows}, "v2 scores must match all 900 IDs")
    parsers = SCORER.load_parsers()
    for row in rows:
        result = by_id[row["id"]]
        require(result["final_correct"] is None or type(result["final_correct"]) is bool, "invalid correctness")
        expected = SCORER.derive_item(row, parsers)
        if compact:
            for field in ("subject", "question_type", "finish_reason", "output_tokens"):
                require(result[field] == row[field], f"{row['id']}: compact score identity mismatch")
            result = {**expected, **result}
            by_id[row["id"]] = result
        for field in ("subject", "question_type", "finish_reason", "route", "raw_sha256", "gt", "choices"):
            require(result[field] == expected[field], f"{row['id']}: score identity/route mismatch ({field})")
        if result["route"] == "auto":
            require(result["final_correct"] == (expected["auto_answer"] == expected["gt"]), "automatic score mismatch")
    return manifest, by_id


def summary(rows, digest, scored, diagnostic):
    v2 = SCORER.aggregate(list(scored.values()))
    v2["by_category"] = {
        category: SCORER.score_group([scored[r["id"]] for r in rows if CATEGORIES[r["subject"]] == category])
        for category in sorted(set(CATEGORIES.values()))
    }
    directions = sum(len(MARKERS.findall(r["raw_text"])) >= 21 for r in rows)
    repeated = sum(REPETITION.flagged(REPETITION.repetition(r["raw_text"])) for r in rows)
    tokens = [r["output_tokens"] for r in rows]
    return {
        "raw_sha256": digest, "v2": v2, "no_judge": diagnostic,
        "processing_routes": dict(Counter(r["route"] for r in scored.values())),
        "length": {"count": sum(r["finish_reason"] == "length" for r in rows), "total": 900},
        "direction_overuse": {"count": directions, "total": 900, "rate": directions / 900},
        "repetition_candidates": {"count": repeated, "total": 900, "rate": repeated / 900},
        "output_tokens": {"total": sum(tokens), "mean": statistics.mean(tokens),
                          "median": statistics.median(tokens), "min": min(tokens), "max": max(tokens)},
    }


def paired(left, right, ids):
    pending = sum(left[i]["final_correct"] is None or right[i]["final_correct"] is None for i in ids)
    if pending:
        return {"total": len(ids), "pending_pairs": pending, "gains": None, "losses": None, "net_gain": None}
    gains = sum(not left[i]["final_correct"] and right[i]["final_correct"] for i in ids)
    losses = sum(left[i]["final_correct"] and not right[i]["final_correct"] for i in ids)
    return {"total": len(ids), "pending_pairs": 0, "gains": gains, "losses": losses, "net_gain": gains - losses}


def compare(left_raw, left_scores, right_raw, right_scores, length_ids, k0):
    verify()
    left, left_sha = raw_rows(left_raw)
    right, right_sha = raw_rows(right_raw)
    indexed = {r["id"]: r for r in right}
    require(set(indexed) == {r["id"] for r in left}, "raw IDs differ")
    for row in left:
        for key in ("subject", "question_type", "gold", "options", "image_indices", "prompt", "num_prompt_tokens"):
            require(row[key] == indexed[row["id"]][key], f"{row['id']}: paired input mismatch ({key})")
    lm, ls = score_rows(left_scores, left, left_sha)
    rm, rs = score_rows(right_scores, right, right_sha)
    require(lm["judge_settings"] == rm["judge_settings"], "Judge settings differ")
    ld, lr = rule_scores(left)
    rd, rr = rule_scores(right)
    ids = Path(length_ids).read_text(encoding="utf-8").splitlines()
    require(len(ids) == len(set(ids)) == 128 and set(ids) <= set(indexed), "expected the frozen 128 unique M0 IDs")
    primary = paired(ls, rs, ids)
    subset_no_judge = paired(lr, rr, ids)
    all_v2 = paired(ls, rs, list(indexed))
    lsum, rsum = summary(left, left_sha, ls, ld), summary(right, right_sha, rs, rd)
    complete = all_v2["pending_pairs"] == 0
    delta = 100 * all_v2["net_gain"] / 900 if complete else None
    mechanism = {
        key: {"left_rate": lsum[key]["count"] / 900, "right_rate": rsum[key]["count"] / 900,
              "delta_pp": 100 * (rsum[key]["count"] - lsum[key]["count"]) / 900,
              "decreased": rsum[key]["count"] < lsum[key]["count"]}
        for key in ("length", "direction_overuse", "repetition_candidates")
    }
    return {
        "left": lsum, "right": rsum, "frozen_length_ids_sha256": hashlib.sha256(Path(length_ids).read_bytes()).hexdigest(),
        "primary_128_v2": primary, "reference_128_no_judge": subset_no_judge,
        "safety_900_v2": {**all_v2, "accuracy_delta_pp": delta},
        "mechanism": mechanism,
        "decision": {"k0_argument": k0, "primary_threshold": k0 + 4,
                     "primary_pass": primary["net_gain"] >= k0 + 4 if primary["net_gain"] is not None else None,
                     "safety_pass": delta >= -1.7 and all_v2["losses"] <= all_v2["gains"] if complete else None,
                     "mechanism_length_and_direction_decreased": all(mechanism[k]["decreased"] for k in ("length", "direction_overuse"))},
        "api_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for side in ("left", "right"):
        parser.add_argument(f"--{side}-raw", required=True, type=Path)
        parser.add_argument(f"--{side}-scores", required=True, type=Path)
    parser.add_argument("--length-ids", type=Path, default=HERE / "m0_length_ids.txt")
    parser.add_argument("--k0", required=True, type=int, help="Explicit agreed value, no default")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.left_raw, args.left_scores, args.right_raw, args.right_scores, args.length_ids, args.k0)
    write_json(args.out, result)
    print(json.dumps({key: result[key] for key in ("primary_128_v2", "safety_900_v2", "decision")}, indent=2))


if __name__ == "__main__":
    main()
