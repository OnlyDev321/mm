"""Build D1 generation and offline v2 preparation reports from saved artifacts."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
REPORTS = ROOT / "reports"
RAW = ROOT / "runs/D1/generation/raw.jsonl"
M0 = WORKSPACE / "assignment/experiments/received_20260922/results/input/raw.jsonl"
E3 = ROOT.parent / "eval_E3_20261101/runs/E3/generation/raw.jsonl"


def load(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def measure(path, expected_sha):
    assert sha(path) == expected_sha
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == 900 and len({r["id"] for r in rows}) == 900
    assert set(Counter(r["subject"] for r in rows).values()) == {30}
    assert all(r["status"] == "ok" and r["finish_reason"] in {"stop", "length"} for r in rows)
    length = [r for r in rows if r["finish_reason"] == "length"]
    closes = sum("</think>" in r["raw_text"] for r in length)
    unclosed = sum("<think>" in r["raw_text"] and "</think>" not in r["raw_text"] for r in length)
    tokens = [r["output_tokens"] for r in rows]
    return {"length": len(length), "closed": closes, "unclosed": unclosed, "no_markers": len(length) - closes - unclosed,
            "total": sum(tokens), "mean": statistics.mean(tokens), "median": statistics.median(tokens)}


def generation():
    proof = load(REPORTS / "TRANSFER_CHECK.json")
    run = load(RAW.parents[1] / "generation_summary.json")
    smoke = load(ROOT / "runs/smoke/generation_summary.json")
    model = load(ROOT / "records/model_receipt.json")
    source = load(REPORTS / "d1_source_receipt.json")
    gate = load(ROOT / "records/gates/env_check.json")["counts"]
    preflight = load(ROOT / "records/preflight/preflight.json")["summary"]
    assert model["a100_hashes_equal"] and model["safetensors_sha256"] == model["a100_safetensors_sha256"]
    assert run["rows"] == run["unique_ids"] == 900 and run["errors"] == run["m0_input_and_prompt_token_mismatches"] == 0
    assert smoke["rows"] == 3 and smoke["errors"] == smoke["m0_input_and_prompt_token_mismatches"] == 0
    assert preflight["processed"] == preflight["selected"] == 900 and not preflight["partial"]
    assert not preflight["validation_failures"] and preflight["exceeds_count"] == 0
    d1 = measure(RAW, proof["raw_sha256"])
    assert d1["total"] == run["output_tokens_total"] and d1["length"] == run["length_count"]
    m0 = measure(M0, run["m0_sha256"])
    e3_run = load(E3.parents[1] / "generation_summary.json")
    e3 = measure(E3, e3_run["raw_sha256"])
    models = (("M0", m0, load(M0.with_name("run_metadata.json"))["run"]["duration_seconds"]),
              ("E3", e3, e3_run["generation_seconds"]), ("D1", d1, run["generation_seconds"]))
    comparison = "\n".join(f"| {name} | {secs:,.3f} | {row['length']} | {row['closed']} | {row['unclosed']} | {row['no_markers']} | "
                           f"{row['total']:,} | {row['mean']:,.2f} | {row['median']:,.1f} |" for name, row, secs in models)
    hashes = "\n".join(f"| `{name}` | `{digest}` |" for name, digest in model["safetensors_sha256"].items())
    text = f"""# D1 평가 1단계 생성 결과

MMMU validation **900문항** 생성과 검증을 완료했다. 파드·로컬 raw 및 실행 기록의 SHA-256이 일치했다.

## 조건과 검증

- E3와 동일한 소스 압축 SHA-256 `{source['source_archive_sha256']}`를 사용했다. 생성 소스 `{source['source_commit']}`, 평가 기준 main `{source['evaluation_main_commit']}`이며 원본 코드·config·게이트·requirements는 그대로다. config SHA-256은 `{run['generation_config_sha256']}`다.
- NVIDIA RTX 4090 1장, 이미지 기본 Python 3.12.3, torch 2.8.0+cu128, vLLM 0.11.0을 사용했다. 설치는 고정 `run_generate.sh`와 같은 경로였고, 병합만 system-site-packages venv와 PEFT 0.17.1로 실행했다.
- 원본 모델 `{model['model']} @ {model['revision']}`에 adapter `{model['adapter_sha256']}`를 bf16으로 병합했다. processor/tokenizer는 원본 바이트와 일치했고, 아래 두 가중치 샤드는 A100 학습 후 병합 기록과 정확히 일치했다.
- 전체 환경 게이트 집계는 `{json.dumps(gate, ensure_ascii=False)}`다. preflight 900/900, 입력 길이 초과·검증 오류 0건이다. 별도 smoke 3문항은 오류·입력 불일치 0건으로 완료했고, 본 실행은 새 폴더에서 시작했다.
- 900개 고유 ID, 30과목 × 30문항, skip·오류 0건이다. M0 입력·프롬프트 토큰 불일치는 0건이며, D1 입력 토큰 합계는 {run['prompt_tokens_total']:,}이다. max_model_len 16,384, max_new_tokens 8,192, 샘플링·seed·이미지·프롬프트는 E3와 같다.

| 병합 샤드 | SHA-256 |
|---|---|
{hashes}

## 생성 비교

| 모델 | 생성 시간(초) | length | `</think>` 있음 | `<think>`만 있음 | 두 태그 없음 | 출력 토큰 합계 | 평균 | 중앙값 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{comparison}

사고 과정 표시는 length 응답에서 문자 그대로의 태그를 센 보고용 분류다. `</think>`가 없고 `<think>`가 있으면 미종료로 분류했다. 두 태그가 없는 응답은 별도로 표시했다. D1 최대 VRAM은 `nvidia-smi` 기준 **{run['max_vram_mib']:,} MiB**다. D1 smoke는 {smoke['generation_seconds']:.3f}초였다. 생성 시간은 각 고정 실행의 `run_metadata.json` 값이다.

## 로컬 전송

- D1 raw SHA-256 `{proof['raw_sha256']}`. 압축 SHA-256 `{proof['archive_sha256']}`, 개별 파일 {proof['files_verified']}개의 해시가 일치했다.
- 본 실행은 `runs/D1/`, 실행 증거는 `records/`에 보관했다. MMMU test·Pro와 재생성은 실행하지 않았다. 파드는 실행 상태로 두었다.
"""
    (REPORTS / "GENERATION_REPORT.md").write_text(text, encoding="utf-8", newline="\n")


def scoring():
    result = load(REPORTS / "SCORING_PREPARATION_SUMMARY.json")
    groups = result["groups"]
    total = groups["all"]
    budget = result["budget"]
    labels = {"all": "전체", "stop": "stop", "length": "length 전체", "length_think_closed": "length · 종료 태그 있음",
              "length_think_unclosed": "length · 시작 태그만 있음", "length_no_think_marker": "length · 두 태그 없음"}
    keys = ("total", "automatic", "judge_total", "rule_extraction_failure", "conflict")
    table = "\n".join(f"| {name} | " + " | ".join(str(groups[key][field]) for field in keys) + " |" for key, name in labels.items())
    text = f"""# D1 평가 2단계 v2 채점 준비

고정 v2로 자동 **{total['automatic']}건**을 처리했다. Judge 대상 **{total['judge_total']}건**은 전부 pending이며 전체 정확도는 `null`이다. Judge 호출과 API 비용은 0이다.

## 입력과 처리 경로

- `source/code/scoring/evaluate.py prepare`를 네트워크 연결을 막은 로컬 CPU에서 실행했다. 정책 `{result['policy_id']}`, 원본·준비 사본 공통 raw SHA-256 `{result['source_sha256']}`다.
- Judge 설정은 `gpt-4.1-mini-2025-04-14`, temperature 0, top_p 1, seed 3407, max_tokens 8이다. 코드·설정·라우팅은 E3와 동일하다.

| 집단 | 문항 | 자동 | Judge 대상 | 규칙 미추출 | 충돌 |
|---|---:|---:|---:|---:|---:|
{table}

M0의 자동/Judge 대상은 555/345건, E3는 737/163건이었다. 규칙 미추출은 Judge로 보내는 사유다. 최종 추출 실패 수와 정확도는 Judge 완료 전까지 미확정이다.

## 보수적 Judge 예산

예상 호출 **{budget['expected_calls']}회**, 예상 비용 **${float(budget['conservative_cost_usd']):.2f}**(계산값 ${budget['conservative_cost_usd']})다. E3와 같은 계산으로 요청 JSON UTF-8 {budget['request_json_utf8_bytes']:,}바이트에 요청당 128토큰을 더한 입력 {budget['input_tokens_conservative_bound']:,}토큰과 최대 출력 {budget['output_tokens_max']:,}토큰을 적용했다. 캐시 할인은 적용하지 않았다. 고정 단가와 일치하는 공개 가격표에서 해당 스냅샷은 Deprecated로 표시된다. [OpenAI GPT-4.1 mini 가격표](https://developers.openai.com/api/docs/models/gpt-4.1-mini)

준비 결과는 `scoring/D1/`, 경로·예산·파일 해시는 `reports/SCORING_PREPARATION_SUMMARY.json`에 보관했다. Judge와 최종 판정은 사용자 확인 후 별도로 진행한다.
"""
    (REPORTS / "SCORING_PREPARATION_REPORT.md").write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generation", "scoring"))
    {"generation": generation, "scoring": scoring}[parser.parse_args().stage]()
