"""Portable MMMU scoring with YAML configuration; prepare is always offline."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import getpass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request
import uuid
import warnings


ROOT = Path(__file__).resolve().parent
ENDPOINT = "https://api.openai.com/v1/chat/completions"
POLICY = "hybrid100_mc_qwen_ab_open_no_length_gate_v2"
DEFAULT_MODEL = "gpt-4.1-mini-2025-04-14"
SUPPORTED_MODELS = (DEFAULT_MODEL, "gpt-3.5-turbo-0125", "gpt-4o-mini-2024-07-18")

SUBJECTS = "Accounting Agriculture Architecture_and_Engineering Art Art_Theory Basic_Medical_Science Biology Chemistry Clinical_Medicine Computer_Science Design Diagnostics_and_Laboratory_Medicine Economics Electronics Energy_and_Power Finance Geography History Literature Manage Marketing Materials Math Mechanical_Engineering Music Pharmacy Physics Psychology Public_Health Sociology".split()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_rows(path, optional=False):
    path = Path(path)
    if optional and not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def unique(rows, label):
    require(all(isinstance(row, dict) and isinstance(row.get("id"), str) for row in rows), label + ": missing ID")
    result = {row["id"]: row for row in rows}
    require(len(result) == len(rows), label + ": duplicate IDs")
    return result


def atomic(path, text):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def write_json(path, value):
    atomic(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_rows(path, values):
    atomic(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values))


def append(path, value):
    with Path(path).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def execution_lock(out):
    lock = Path(out) / ".evaluation.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError("Run is locked. Confirm no evaluator is active before manually removing .evaluation.lock.") from None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "created_at": now()}, handle)
        yield
    finally:
        lock.unlink()


def load_scoring_config(path, judge_model=None):
    try:
        import yaml
    except ImportError:
        raise ValueError("Install requirements.txt (PyYAML is required to read --config)") from None
    data = Path(path).read_bytes()
    try:
        config = yaml.safe_load(data.decode("utf-8-sig"))
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise ValueError("Invalid YAML configuration: " + type(exc).__name__) from None
    require(isinstance(config, dict) and isinstance(config.get("scoring"), dict), "Configuration must contain a scoring mapping")
    scoring = config["scoring"]
    require(scoring.get("policy") == POLICY, "Unsupported scoring.policy")
    require(isinstance(scoring.get("judge"), dict), "Missing scoring.judge")
    judge = dict(scoring["judge"])
    require(set(judge) == {"model", "temperature", "top_p", "seed", "max_tokens"}, "scoring.judge must contain model, temperature, top_p, seed, max_tokens only")
    if judge_model is not None:
        judge["model"] = judge_model
    require(judge["model"] in SUPPORTED_MODELS, "Unsupported Judge model snapshot")
    for key in ("temperature", "top_p"):
        require(type(judge[key]) in (int, float) and math.isfinite(judge[key]), "Invalid Judge " + key)
    require(0 <= judge["temperature"] <= 2 and 0 < judge["top_p"] <= 1, "Judge temperature/top_p out of range")
    require(type(judge["seed"]) is int and judge["seed"] >= 0, "Invalid Judge seed")
    require(type(judge["max_tokens"]) is int and judge["max_tokens"] > 0, "Invalid Judge max_tokens")
    pricing = scoring.get("prices_usd_per_million")
    require(isinstance(pricing, dict) and isinstance(pricing.get(judge["model"]), dict), "Missing scoring.prices_usd_per_million for selected model")
    prices = dict(pricing[judge["model"]])
    require(set(prices) == {"input", "cached_input", "output"}, "Prices must contain input, cached_input, output")
    require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in prices.values()), "Invalid token price")
    return data, judge, prices


def code_hashes():
    paths = [ROOT / "evaluate.py", ROOT / "frozen" / "parsers.py", *sorted((ROOT / "frozen" / "vendor").glob("*.py"))]
    require(len(paths) >= 5, "Missing frozen parser/vendor files")
    return {path.relative_to(ROOT).as_posix(): sha(path.read_bytes()) for path in paths}


def load_parsers():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("portable_frozen_parsers", ROOT / "frozen" / "parsers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_raw(rows):
    unique(rows, "raw")
    require(len(rows) == 900, "Expected all 900 validation items")
    require(Counter(row.get("subject") for row in rows) == Counter({subject: 30 for subject in SUBJECTS}), "Expected the fixed 30 subjects x 30 items")
    require(Counter(row.get("question_type") for row in rows) == Counter({"multiple-choice": 847, "open": 53}), "Expected MC 847 / open 53")
    fields = {"schema_version", "synthetic", "id", "subject", "subfield", "topic_difficulty", "img_type", "question_type", "gold", "options", "options_raw", "image_indices", "prompt", "status", "reason", "error", "raw_text", "output_token_ids", "output_tokens", "finish_reason", "stop_reason", "max_new_tokens", "num_prompt_tokens", "precomputed_prompt_tokens"}
    for row in rows:
        label = row["id"] + ": "
        require(fields == set(row), label + "raw schema fields differ; missing=" + ", ".join(sorted(fields - set(row))) + "; extra=" + ", ".join(sorted(set(row) - fields)))
        for key in ("schema_version", "id", "subject", "question_type", "gold", "options_raw", "prompt", "status", "raw_text", "finish_reason"):
            require(isinstance(row[key], str), label + key + " must be a string")
        for key in ("subfield", "topic_difficulty", "img_type"):
            require(row[key] is None or isinstance(row[key], str), label + key + " must be a string or null")
        require(row["schema_version"] == "0", label + "unsupported raw schema version")
        require(row["id"] in {f"validation_{row['subject']}_{number}" for number in range(1, 31)}, label + "ID does not belong to the fixed subject validation split")
        require(row["synthetic"] is False, label + "synthetic data not accepted")
        require(row["status"] == "ok" and row["error"] is None and row["reason"] is None, label + "inference error or unfinished row")
        require(row["raw_text"].strip() and row["prompt"].strip() and row["gold"].strip(), label + "empty input/output/reference")
        require(row["finish_reason"] in ("stop", "length"), label + "unknown finish_reason")
        require(row["stop_reason"] is None or type(row["stop_reason"]) in (str, int), label + "invalid stop_reason")
        require(isinstance(row["options"], list) and all(isinstance(text, str) and text.strip() for text in row["options"]), label + "invalid options list")
        if row["question_type"] == "multiple-choice":
            require(2 <= len(row["options"]) <= 26 and row["gold"] in [chr(65 + index) for index in range(len(row["options"]))], label + "invalid MC gold/options")
        else:
            require(row["options"] == [], label + "open options must be empty")
        require(isinstance(row["image_indices"], list) and all(type(index) is int and 1 <= index <= 7 for index in row["image_indices"]), label + "invalid image_indices")
        require(row["image_indices"] == sorted(set(row["image_indices"])), label + "image indices must be unique and ascending")
        for key in ("output_tokens", "max_new_tokens", "num_prompt_tokens", "precomputed_prompt_tokens"):
            require(type(row[key]) is int and row[key] > 0, label + "invalid " + key)
        tokens = row["output_token_ids"]
        require(isinstance(tokens, list) and all(type(token) is int and token >= 0 for token in tokens), label + "invalid output_token_ids")
        require(len(tokens) == row["output_tokens"] <= row["max_new_tokens"], label + "token count/budget mismatch")
        require(row["precomputed_prompt_tokens"] == row["num_prompt_tokens"], label + "prompt token counts differ")
        require(row["finish_reason"] != "length" or row["output_tokens"] == row["max_new_tokens"], label + "length output count must equal the recorded cap")


def derive_item(row, parsers):
    is_open = row["question_type"] == "open"
    choices = {"A": row["gold"], "B": "Other Answers"} if is_open else {chr(65 + index): value for index, value in enumerate(row["options"])}
    answer = parsers.qwen_parse(row["raw_text"], choices)
    conflict = False
    if not is_open:
        marker_answer = parsers.marker(row["raw_text"], choices)["answer"]
        conflict = bool(answer and marker_answer and answer != marker_answer)
        answer = None if conflict else (answer or marker_answer)
    route = "judge_conflict" if conflict else "auto" if answer is not None else "judge_rule_failure"
    return {**{key: row[key] for key in ("id", "subject", "question_type", "finish_reason", "output_tokens")},
            "choices": choices, "gt": "A" if is_open else row["gold"], "route": route,
            "auto_answer": answer if route == "auto" else None,
            "raw_sha256": sha(row["raw_text"].encode("utf-8")), "reference_answer_visible": is_open}


def make_request(row, item, judge_settings, parsers, source_hash):
    payload = {**judge_settings, "messages": [{"role": "user", "content": parsers.judge_prompt(row["prompt"], item["choices"], row["raw_text"])}]}
    return {"id": row["id"], "request_sha256": sha(canonical(payload)), "request": payload,
            "raw_sha256": item["raw_sha256"], "source_sha256": source_hash, "route": item["route"],
            "question_type": row["question_type"], "reference_answer_visible": item["reference_answer_visible"]}


def score_group(rows):
    pending = sum(row["final_correct"] is None for row in rows)
    correct = sum(row["final_correct"] is True for row in rows)
    incorrect = sum(row["final_correct"] is False for row in rows)
    return {"total": len(rows), "complete": correct + incorrect, "pending": pending, "correct": correct,
            "incorrect": incorrect, "extraction_failures": sum(row["final_status"] == "extraction_failure" for row in rows),
            "accuracy": correct / len(rows) if rows and not pending else None}


def aggregate(rows):
    result = {"overall": score_group(rows)}
    for label, field in (("by_subject", "subject"), ("by_type", "question_type"), ("by_finish", "finish_reason"), ("by_route", "route")):
        result[label] = {name: score_group([row for row in rows if row[field] == name]) for name in sorted({row[field] for row in rows})}
    result["macro_accuracy"] = None if result["overall"]["pending"] else sum(group["accuracy"] for group in result["by_subject"].values()) / len(result["by_subject"])
    require(result["macro_accuracy"] is None or abs(result["macro_accuracy"] - result["overall"]["accuracy"]) < 1e-12, "Macro / overall mismatch")
    return result


def mmmu_diagnostic(rows, parsers):
    details, adaptations = [], []
    for row in rows:
        gold = row["gold"]
        if row["question_type"] == "multiple-choice":
            choices = {chr(65 + index): text for index, text in enumerate(row["options"])}
            answer, blocked = parsers.mmmu_mc_parse(row["raw_text"], choices)
            correct, failure = answer == gold, answer is None
        else:
            if gold.strip().startswith("[") and gold.strip().endswith("]"):
                alternatives = ast.literal_eval(gold)
                require(isinstance(alternatives, list) and alternatives and all(isinstance(text, str) and text.strip() for text in alternatives), row["id"] + ": malformed alternative gold")
                adaptations.append({"id": row["id"], "original": gold, "alternatives": alternatives})
                gold = alternatives
            answer, blocked = parsers.open_candidates(row["raw_text"]), False
            correct, failure = bool(parsers.MMMU["eval_open"](gold, answer)), not answer
        details.append({**{key: row[key] for key in ("id", "subject", "question_type", "finish_reason")},
                        "route": "mmmu_rule_no_length_gate", "parsed": answer, "gold_for_eval": gold,
                        "random_fallback_blocked": blocked, "final_correct": bool(correct),
                        "final_status": "extraction_failure" if failure else "complete_rule"})
    return {**aggregate(details), "policy": "MMMU rules; random fallback disabled; failures incorrect; no Judge; no length gate",
            "serialized_gold_adaptations": adaptations, "random_fallback_blocked": sum(row["random_fallback_blocked"] for row in details), "api_calls": 0}, details


def prepare(raw, out, config, judge_model=None):
    raw, out = Path(raw).resolve(), Path(out).resolve()
    config_data, judge_settings, prices = load_scoring_config(config, judge_model)
    data = raw.read_bytes()
    source_hash = sha(data)
    rows = [json.loads(line) for line in data.decode("utf-8-sig").splitlines() if line.strip()]
    validate_raw(rows)
    parsers = load_parsers()
    items = [derive_item(row, parsers) for row in rows]
    requests = [make_request(row, item, judge_settings, parsers, source_hash) for row, item in zip(rows, items) if item["route"] != "auto"]
    diagnostic, diagnostic_rows = mmmu_diagnostic(rows, parsers)
    out.mkdir(parents=True, exist_ok=True)
    require(not any(out.iterdir()), "Output directory is not empty; use run/summarize for an existing evaluation, or choose a new directory")
    with execution_lock(out):
        (out / "input").mkdir()
        (out / "input" / "raw.jsonl").write_bytes(data)
        (out / "input" / "config.yaml").write_bytes(config_data)
        write_rows(out / "items.jsonl", items)
        write_rows(out / "requests.jsonl", requests)
        manifest = {"schema_version": "portable-evaluation-1", "created_at": now(), "source_path": "input/raw.jsonl",
                    "source_sha256": source_hash, "policy_id": POLICY, "judge_settings": judge_settings, "endpoint": ENDPOINT,
                    "config_path": "input/config.yaml", "config_sha256": sha(config_data), "judge_model_override": judge_model, "price_per_million_usd": prices,
                    "total": len(rows), "judge_count": len(requests), "automatic_count": len(items) - len(requests),
                    "items_sha256": sha((out / "items.jsonl").read_bytes()), "requests_sha256": sha((out / "requests.jsonl").read_bytes()),
                    "code_sha256": code_hashes(), "max_new_tokens_values": sorted({row["max_new_tokens"] for row in rows}),
                    "reproducibility_metadata": {"model_revision": "not_attested_by_raw", "dataset_revision": "not_attested_by_raw", "generation_seed": "not_attested_by_raw", "inference_code_commit": "not_attested_by_raw"},
                    "policy_notes": ["MC gold hidden from Judge; open A contains verbatim gold and B is Other Answers.", "Only rule failures and MC rule conflicts go to Judge, regardless of backbone finish_reason; no random fallback or retries of completed Z.", "Pinned prompt lists A-D/Z; parsing permits all actual option labels.", "Raw snapshot is byte-identical; original input was not modified."]}
        write_json(out / "manifest.json", manifest)
        write_json(out / "mmmu_summary.json", {"source_sha256": source_hash, **diagnostic})
        write_rows(out / "mmmu_results.jsonl", diagnostic_rows)
        require(sha(raw.read_bytes()) == source_hash, "Original input changed during preparation")
        result = summarize(out)
    return result


def load_inputs(out):
    out = Path(out).resolve()
    manifest = read_json(out / "manifest.json")
    require(manifest.get("schema_version") == "portable-evaluation-1" and manifest["policy_id"] == POLICY, "Unsupported manifest/policy")
    require(manifest["endpoint"] == ENDPOINT, "Endpoint changed")
    config_rel = Path(manifest["config_path"])
    require(not config_rel.is_absolute() and (out / config_rel).resolve().is_relative_to(out), "Config snapshot must remain inside the run directory")
    config_data, expected_settings, expected_prices = load_scoring_config(out / config_rel, manifest.get("judge_model_override"))
    require(sha(config_data) == manifest["config_sha256"], "Config snapshot hash mismatch")
    require(manifest["judge_settings"] == expected_settings and manifest["price_per_million_usd"] == expected_prices, "Effective Judge settings/prices differ from saved scoring configuration")
    source_rel = Path(manifest["source_path"])
    require(not source_rel.is_absolute(), "source_path must be relative to the run directory")
    source = (out / source_rel).resolve()
    require(source.is_relative_to(out), "Source snapshot must remain inside the run directory")
    require(sha(source.read_bytes()) == manifest["source_sha256"], "Raw snapshot hash mismatch")
    require(manifest["code_sha256"] == code_hashes(), "Evaluator/frozen parser hashes changed; use the original package")
    for filename, key in (("items.jsonl", "items_sha256"), ("requests.jsonl", "requests_sha256")):
        require(sha((out / filename).read_bytes()) == manifest[key], filename + " hash mismatch")
    rows = read_rows(source)
    validate_raw(rows)
    parsers = load_parsers()
    items = read_rows(out / "items.jsonl")
    requests = read_rows(out / "requests.jsonl")
    unique(items, "items")
    unique(requests, "requests")
    expected_items = [derive_item(row, parsers) for row in rows]
    require(items == expected_items, "Saved routes/automatic answers differ from frozen rules")
    expected_requests = [make_request(row, item, manifest["judge_settings"], parsers, manifest["source_sha256"]) for row, item in zip(rows, items) if item["route"] != "auto"]
    require(requests == expected_requests, "Saved request bodies differ from the raw/policy")
    require(len(items) == manifest["total"] and len(requests) == manifest["judge_count"] and len(items) - len(requests) == manifest["automatic_count"], "Manifest count mismatch")
    return out, manifest, rows, items, requests, parsers


def response_usage(response, model):
    require(response.get("model") == model, "API response model does not match the pinned snapshot")
    u = response.get("usage")
    require(isinstance(u, dict), "API response missing usage")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        require(type(u.get(key)) is int and u[key] >= 0, "Invalid usage: " + key)
    require(u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"], "Total usage mismatch")
    details = u.get("prompt_tokens_details") or {}
    require(isinstance(details, dict), "Invalid prompt token details")
    cached = details.get("cached_tokens", 0)
    require(type(cached) is int and 0 <= cached <= u["prompt_tokens"], "Invalid cached token usage")
    return {**{key: u[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")}, "cached_tokens": cached, "uncached_prompt_tokens": u["prompt_tokens"] - cached}


def calculate(out):
    out, manifest, rows, items, request_rows, parsers = load_inputs(out)
    requests = unique(request_rows, "requests")
    responses = unique(read_rows(out / "responses.jsonl", optional=True), "responses")
    require(set(responses) <= set(requests), "Response has no matching request")
    attempts = read_rows(out / "api_attempts.jsonl", optional=True)
    for event in attempts:
        require(event.get("id") in requests and event.get("request_sha256") == requests[event["id"]]["request_sha256"], "Attempt provenance mismatch")
        require(event.get("event") in ("started", "received", "failed") and bool(event.get("attempt_id")), "Malformed attempt event")
    totals = Counter({key: 0 for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "uncached_prompt_tokens")})
    finals, raw_outputs, response_finishes = [], Counter(), Counter()
    for row, item in zip(rows, items):
        result = {**item, "source_path": manifest["source_path"], "source_sha256": manifest["source_sha256"],
                  "policy_id": POLICY, "raw_text": row["raw_text"], "input_prompt": row["prompt"],
                  "reference_gold_original": row["gold"], "reference_answer_visible_to_judge": item["reference_answer_visible"],
                  "final_answer": None, "final_correct": None, "final_status": "pending_api",
                  "final_source": "rule" if item["route"] == "auto" else "judge", "request_sha256": None,
                  "judge_response_text": None, "judge_response": None, "judge_finish_reason": None, "judge_usage": None, "judge_received_at": None}
        if item["route"] == "auto":
            result.update(final_answer=item["auto_answer"], final_correct=item["auto_answer"] == item["gt"], final_status="complete_rule")
        else:
            request = requests[item["id"]]
            result["request_sha256"] = request["request_sha256"]
            if item["id"] in responses:
                saved = responses[item["id"]]
                require(saved.get("request_sha256") == request["request_sha256"] and bool(saved.get("received_at")), "Response provenance mismatch: " + item["id"])
                response = saved["response"]
                usage = response_usage(response, manifest["judge_settings"]["model"])
                totals.update(usage)
                choices = response.get("choices")
                require(isinstance(choices, list) and len(choices) == 1, "Expected exactly one Judge response choice")
                choice = choices[0]
                require(isinstance(choice.get("message"), dict), "Missing response message")
                text = choice["message"].get("content")
                require(text is None or isinstance(text, str), "Judge response content must be text")
                finish = choice.get("finish_reason")
                require(isinstance(finish, str) and bool(finish), "Missing Judge finish_reason")
                answer = parsers.qwen_parse(text or "", item["choices"]) if finish == "stop" else None
                result.update(final_answer=answer, final_correct=answer == item["gt"] if answer is not None else False,
                              final_status="complete_judge" if answer is not None else "extraction_failure",
                              judge_response_text=text, judge_response=response, judge_finish_reason=finish,
                              judge_usage=usage, judge_received_at=saved["received_at"])
                raw_outputs[text or ""] += 1
                response_finishes[finish] += 1
        finals.append(result)
    prices = dict(manifest["price_per_million_usd"])
    cost = (totals["uncached_prompt_tokens"] * prices["input"] + totals["cached_tokens"] * prices["cached_input"] + totals["completion_tokens"] * prices["output"]) / 1_000_000
    diagnostic, diagnostic_rows = mmmu_diagnostic(rows, parsers)
    summary = {"source_path": manifest["source_path"], "source_sha256": manifest["source_sha256"], "policy_id": POLICY,
               "judge_settings": manifest["judge_settings"], **aggregate(finals), "automatic_count": manifest["automatic_count"],
               "judge_count": len(requests), "response_count": len(responses), "judge_usage": dict(totals),
               "price_per_million_usd": prices, "estimated_recorded_response_cost_usd": round(cost, 10),
               "judge_raw_outputs": dict(raw_outputs), "raw_z_outputs": raw_outputs.get("Z", 0),
               "judge_response_finish_reasons": dict(response_finishes), "api_attempt_events": dict(Counter(event["event"] for event in attempts)),
               "max_new_tokens_values": sorted({row["max_new_tokens"] for row in rows}),
               "judge_mc_more_than_four_choices": sum(item["question_type"] == "multiple-choice" and item["route"] != "auto" and len(item["choices"]) > 4 for item in items),
               "input_file_sha256": {name: sha((out / name).read_bytes()) for name in ("manifest.json", "items.jsonl", "requests.jsonl", "responses.jsonl", "api_attempts.jsonl", "errors.jsonl") if (out / name).exists()},
               "mmmu_diagnostic": {"overall": diagnostic["overall"], "policy": diagnostic["policy"]},
               "aggregation_api_calls": 0, "raw_unchanged": True}
    require(sha((out / manifest["source_path"]).read_bytes()) == manifest["source_sha256"], "Raw snapshot changed while scoring")
    return summary, finals, diagnostic, diagnostic_rows


def format_score(group):
    return "pending" if group["accuracy"] is None else f"{group['accuracy'] * 100:.2f}% ({group['correct']}/{group['total']})"


def scoring_report(s):
    lines = ["# 저장 응답 평가 결과", "", f"전체: **{format_score(s['overall'])}**. API 미완료 {s['overall']['pending']}건.",
             f"자동 {s['automatic_count']}건 / Judge 대상 {s['judge_count']}건 / 저장 응답 {s['response_count']}건.",
             f"Judge: `{s['judge_settings']['model']}`. Raw SHA-256: `{s['source_sha256']}`.", "",
             "| 구분 | 전체 | 정답 | 오답 | 추출 실패 | 미완료 | 정확도 |", "|---|---:|---:|---:|---:|---:|---|"]
    for label in ("by_type", "by_finish", "by_route", "by_subject"):
        for name, group in s[label].items():
            lines.append(f"| {name} | {group['total']} | {group['correct']} | {group['incorrect']} | {group['extraction_failures']} | {group['pending']} | {format_score(group)} |")
    usage, prices = s["judge_usage"], s["price_per_million_usd"]
    lines += ["", "과목당 30문항이므로 종합 정확도는 30과목 정확도의 단순 평균과 같다. 미완료가 있는 집단은 정확도를 계산하지 않는다.",
              f"MMMU 규칙 단독: **{format_score(s['mmmu_diagnostic']['overall'])}**. 랜덤 폴백 제거·실패 오답·Judge 없음·length 게이트 없음·open 복수 정답 목록 어댑터 적용이다.", "",
              f"Judge 입력 {usage['prompt_tokens']:,} (캐시 {usage['cached_tokens']:,}), 출력 {usage['completion_tokens']:,} tokens. 기록 응답 비용 추정 **${s['estimated_recorded_response_cost_usd']:.6f}**.",
              f"100만 토큰당 입력 ${prices['input']:g}, 캐시 입력 ${prices['cached_input']:g}, 출력 ${prices['output']:g}. 미반환 실패 요청의 청구는 포함하지 않는다.", "",
              "- MC는 Qwen + 기존 Final Answer 보완, open은 A=참조답 원문/B=Other Answers다. Final Answer 문자 보완은 MC에만 적용한다.",
              "- Final Answer 100자는 마지막 marker 끝부터 응답 끝까지의 문자 수다. 모델 출력 길이 제한이나 프롬프트 변경 지시가 아니다.",
              "- 추론 종료 사유와 관계없이 규칙 추출 성공은 자동 처리하며, 규칙 미추출·MC 규칙 충돌만 Judge 대상이다. MC Judge에는 정답을 따로 주지 않으며 open Judge에는 참조답을 준다.",
              "- Judge stop 응답만 고정 Qwen 파서로 읽는다. Z·유효 답 없음·Judge non-stop은 완료된 추출 실패로 오답, API 미완료는 pending이다. 완료된 Z를 재호출하지 않는다.",
              f"- 고정 Judge 프롬프트는 A–D/Z를 열거한다. 실제 선택지가 5개 이상인 Judge MC {s['judge_mc_more_than_four_choices']}건도 실제 선택지 전체를 유효하게 처리한다.",
              "- 출력 점수는 채점 정책의 판정이며 사람이 검증한 원문 추출 정확도가 아니다. 높은 점수만으로 원문 답 추출의 충실도를 판단하지 않는다.",
              "- 원본을 보존하고 input/raw.jsonl에 동일 바이트 사본을 저장했다. 모델·데이터 revision, 생성 seed, 추론 코드 commit의 일치는 raw만으로 증명되지 않는다.",
              "- final_results.jsonl은 원문·참조답·Judge 전체 응답과 판정을 보존한다. summarize는 저장 파일만 사용하며 API를 호출하지 않는다.", ""]
    return "\n".join(lines)


def summarize(out):
    out = Path(out).resolve()
    summary, finals, diagnostic, diagnostic_rows = calculate(out)
    write_rows(out / "final_results.jsonl", finals)
    write_json(out / "summary.json", summary)
    write_json(out / "mmmu_summary.json", {"source_sha256": summary["source_sha256"], **diagnostic})
    write_rows(out / "mmmu_results.jsonl", diagnostic_rows)
    atomic(out / "REPORT.md", scoring_report(summary))
    return summary


def run(out, ask_key=False, limit=None):
    out = Path(out).resolve()
    require(limit is None or type(limit) is int and limit >= 0, "limit must be a nonnegative integer")
    require((out / "manifest.json").exists(), "Run prepare first")
    with execution_lock(out):
        # Both cached responses and unresolved attempts are checked under the lock.
        current = summarize(out)
        _, manifest, _, _, requests, _ = load_inputs(out)
        cached = unique(read_rows(out / "responses.jsonl", optional=True), "responses")
        attempts = read_rows(out / "api_attempts.jsonl", optional=True)
        started = {event["id"] for event in attempts if event["event"] == "started"}
        unresolved = started - set(cached)
        require(not unresolved, "Unresolved previous API attempt(s): " + ", ".join(sorted(unresolved)) + ". No automatic retry; inspect the saved receipt/journal first.")
        todo = [request for request in requests if request["id"] not in cached]
        if limit is not None:
            todo = todo[:limit]
        if not todo:
            return current
        if ask_key:
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                try:
                    key = getpass.getpass("OpenAI API key (hidden; not saved): ")
                except getpass.GetPassWarning:
                    raise ValueError("Hidden key entry is unavailable in this terminal; use an interactive terminal or OPENAI_API_KEY") from None
        else:
            key = os.environ.get("OPENAI_API_KEY", "")
        key = key.strip()
        require(bool(key), "Set OPENAI_API_KEY, or use run --ask-key")
        try:
            for request in todo:
                attempt = {"id": request["id"], "request_sha256": request["request_sha256"], "attempt_id": str(uuid.uuid4())}
                append(out / "api_attempts.jsonl", {**attempt, "event": "started", "at": now()})
                try:
                    req = urllib.request.Request(ENDPOINT, data=canonical(request["request"]), headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}, method="POST")
                    with urllib.request.urlopen(req, timeout=60) as response:
                        body = response.read().decode("utf-8")
                        http_status = response.status
                        http_request_id = response.headers.get("x-request-id")
                    # Authorization headers are never persisted; a successful body is retained verbatim.
                    append(out / "http_receipts.jsonl", {**attempt, "received_at": now(), "http_status": http_status, "http_request_id": http_request_id, "body": body})
                    payload = json.loads(body)
                    append(out / "responses.jsonl", {**attempt, "received_at": now(), "response": payload})
                    append(out / "api_attempts.jsonl", {**attempt, "event": "received", "at": now()})
                except Exception as exc:
                    # HTTP error bodies may repeat a credential. Keep only safe structural error data.
                    error = {**attempt, "event": "failed", "at": now(), "error_type": type(exc).__name__,
                             "http_status": exc.code if isinstance(exc, urllib.error.HTTPError) else None,
                             "message": "API attempt failed; no automatic retry. Error body omitted to protect credentials."}
                    append(out / "errors.jsonl", error)
                    append(out / "api_attempts.jsonl", error)
                    summarize(out)
                    raise RuntimeError("API attempt failed for " + request["id"] + "; results remain pending. Inspect errors.jsonl before taking further action.") from None
                current = summarize(out)
                print(f"received {current['response_count']}/{current['judge_count']}; pending {current['overall']['pending']}", flush=True)
        finally:
            key = ""
        return current


def compare(left, right, out):
    left, right, out = Path(left).resolve(), Path(right).resolve(), Path(out).resolve()
    require(not out.is_relative_to(left) and not out.is_relative_to(right), "Comparison output must be outside both source runs")
    linputs, rinputs = load_inputs(left), load_inputs(right)
    lm, rm = linputs[1], rinputs[1]
    require(lm["judge_settings"] == rm["judge_settings"] and lm["policy_id"] == rm["policy_id"] and lm["endpoint"] == rm["endpoint"], "Judge model/settings/policy differ; this is not a matched budget comparison")
    lraw, rraw = unique(linputs[2], "left raw"), unique(rinputs[2], "right raw")
    require(set(lraw) == set(rraw), "Question ID sets differ")
    input_fields = ("subject", "question_type", "gold", "options", "options_raw", "prompt", "image_indices", "img_type", "num_prompt_tokens", "precomputed_prompt_tokens")
    for key in lraw:
        for field in input_fields:
            require(lraw[key][field] == rraw[key][field], key + ": comparison input differs: " + field)
        if "question" in lraw[key] or "question" in rraw[key]:
            require(lraw[key].get("question") == rraw[key].get("question"), key + ": question text differs")
    ls, lf, lmm, _ = calculate(left)
    rs, rf, rmm, _ = calculate(right)
    lfinal, rfinal = unique(lf, "left result"), unique(rf, "right result")
    common = [key for key in lfinal if lfinal[key]["final_correct"] is not None and rfinal[key]["final_correct"] is not None]
    gains = sum(not lfinal[key]["final_correct"] and rfinal[key]["final_correct"] for key in common)
    losses = sum(lfinal[key]["final_correct"] and not rfinal[key]["final_correct"] for key in common)
    complete = len(common) == len(lfinal)
    if complete:
        require(gains - losses == rs["overall"]["correct"] - ls["overall"]["correct"], "Paired correctness mismatch")
    change_rows = [{"id": key, "subject": lfinal[key]["subject"], "question_type": lfinal[key]["question_type"],
                    "left_finish": lfinal[key]["finish_reason"], "right_finish": rfinal[key]["finish_reason"],
                    "left_answer": lfinal[key]["final_answer"], "right_answer": rfinal[key]["final_answer"],
                    "left_status": lfinal[key]["final_status"], "right_status": rfinal[key]["final_status"],
                    "left_correct": lfinal[key]["final_correct"], "right_correct": rfinal[key]["final_correct"]} for key in lfinal]
    result = {"state": "complete" if complete else "partial", "matched_question_inputs_verified": True,
              "judge_model_settings_policy_verified": True, "cap_only_causal_effect_verified": False,
              "unverified_conditions": ["backbone model revision", "dataset revision", "generation seed/sampling settings", "inference code commit", "image preprocessing and actual image bytes", "backend/environment"],
              "left": {"source_sha256": lm["source_sha256"], "max_new_tokens_values": ls["max_new_tokens_values"], "scores": ls, "mmmu_rule_scores": lmm},
              "right": {"source_sha256": rm["source_sha256"], "max_new_tokens_values": rs["max_new_tokens_values"], "scores": rs, "mmmu_rule_scores": rmm},
              "paired_completed": len(common), "paired_pending": len(lfinal) - len(common),
              "right_correctness_gains": gains, "right_correctness_losses": losses,
              "right_minus_left_accuracy_pp": (gains - losses) / len(lfinal) * 100 if complete else None,
              "answer_or_status_disagreements_on_completed": sum(lfinal[key]["final_answer"] != rfinal[key]["final_answer"] or lfinal[key]["final_status"] != rfinal[key]["final_status"] for key in common),
              "inference_finish_transitions": dict(Counter(lraw[key]["finish_reason"] + " -> " + rraw[key]["finish_reason"] for key in lraw)), "api_calls": 0}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "comparison.json", result)
    write_rows(out / "paired_results.jsonl", change_rows)
    report = ["# 두 추론 출력의 동일 채점 정책 비교", "", "900개 ID·질문 문맥·정답·선택지·기록된 이미지 정보 및 Judge 모델/설정/채점 정책이 같은지 확인했다. API를 호출하지 않았다.", "",
              "| 항목 | left | right |", "|---|---|---|",
              f"| 기록된 max_new_tokens | {ls['max_new_tokens_values']} | {rs['max_new_tokens_values']} |",
              f"| 전체 | {format_score(ls['overall'])} | {format_score(rs['overall'])} |",
              f"| MC | {format_score(ls['by_type']['multiple-choice'])} | {format_score(rs['by_type']['multiple-choice'])} |",
              f"| open | {format_score(ls['by_type']['open'])} | {format_score(rs['by_type']['open'])} |",
              f"| stop | {format_score(ls['by_finish'].get('stop', {'accuracy': None}))} | {format_score(rs['by_finish'].get('stop', {'accuracy': None}))} |",
              f"| length 수 | {ls['by_finish'].get('length', {}).get('total', 0)} | {rs['by_finish'].get('length', {}).get('total', 0)} |",
              f"| length 점수 | {format_score(ls['by_finish'].get('length', {'accuracy': None}))} | {format_score(rs['by_finish'].get('length', {'accuracy': None}))} |",
              f"| MMMU 규칙 단독 | {format_score(lmm['overall'])} | {format_score(rmm['overall'])} |",
              f"| Judge 비용 추정 | ${ls['estimated_recorded_response_cost_usd']:.6f} | ${rs['estimated_recorded_response_cost_usd']:.6f} |", "",
              f"완료 쌍 {len(common)}/900. 오른쪽 정답 증가 {gains}, 감소 {losses}. 전체 점수 차이: " + ("pending" if not complete else f"{result['right_minus_left_accuracy_pp']:+.2f}%p") + ".", "",
              "미완료가 있는 집단의 정확도는 pending이다. 두 파일의 length/stop 집단은 서로 다른 문항일 수 있으므로 조건부 점수를 동일 문항 비교로 해석하지 않는다.",
              "**cap만 바꾼 실험이라는 사실은 검증되지 않았다.** raw만으로 모델·데이터 revision, 생성 seed/샘플링 설정, 추론 코드 commit, 이미지 전처리·실제 이미지, 실행 환경의 일치를 증명할 수 없다. 따라서 점수 차이를 cap의 인과 효과로 단정하지 않는다. 별도 실행 메타데이터가 필요하다.",
              "Final Answer 100자는 채점 규칙이며 출력 길이 제한이 아니다. 높은 점수만으로 원문 답 추출의 충실도가 검증되는 것은 아니다.", ""]
    atomic(out / "REPORT.md", "\n".join(report))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Validate/copy raw and prepare requests offline; no API calls")
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--judge-model", choices=SUPPORTED_MODELS)
    p = sub.add_parser("run", help="Send each outstanding request once using your API key")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ask-key", action="store_true")
    p.add_argument("--limit", type=int)
    p = sub.add_parser("summarize", help="Replay saved responses offline")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("compare", help="Compare two matched-question runs offline")
    p.add_argument("--left", type=Path, required=True)
    p.add_argument("--right", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.raw, args.out, args.config, args.judge_model)
        elif args.command == "run":
            result = run(args.out, args.ask_key, args.limit)
        elif args.command == "summarize":
            result = summarize(args.out)
        else:
            result = compare(args.left, args.right, args.out)
        compact = {key: result[key] for key in ("overall", "automatic_count", "judge_count", "response_count", "estimated_recorded_response_cost_usd", "state", "paired_completed", "right_minus_left_accuracy_pp") if key in result}
        print(json.dumps(compact, ensure_ascii=False))
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
