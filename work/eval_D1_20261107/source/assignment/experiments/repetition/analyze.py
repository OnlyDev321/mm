"""Exact-repetition diagnostics for saved model responses."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from statistics import median

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT.parent / "submission"
THRESHOLDS = (0.10, 0.20, 0.30)
MIN_CHARS, MIN_WORDS, NGRAM = 40, 8, 16


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    require(len({r["id"] for r in rows}) == len(rows), f"Duplicate IDs: {path.name}")
    return rows


def units(text):
    """Sentence/line heuristic, with offsets into the unchanged original text."""
    start = 0
    for boundary in re.finditer(r"(?<=[.!?。！？])\s+|\n+", text):
        piece = text[start:boundary.start()]
        if piece.strip():
            yield " ".join(piece.split()), start + len(piece) - len(piece.lstrip())
        start = boundary.end()
    piece = text[start:]
    if piece.strip():
        yield " ".join(piece.split()), start + len(piece) - len(piece.lstrip())


def covered_length(intervals):
    end, count = 0, 0
    for left, right in sorted(intervals):
        count += max(0, right - max(left, end))
        end = max(end, right)
    return count


def repetition(text):
    segments = list(units(text))
    denominator = sum(len(s) for s, _ in segments)
    occurrences = defaultdict(list)
    for segment, offset in segments:
        if len(segment) >= MIN_CHARS and len(re.findall(r"\w+", segment)) >= MIN_WORDS:
            occurrences[segment].append(offset)
    repeated = [(s, offsets) for s, offsets in occurrences.items() if len(offsets) >= 3]
    duplicate_chars = sum(len(s) * (len(offsets) - 1) for s, offsets in repeated)
    dominant = max(repeated, key=lambda x: len(x[0]) * (len(x[1]) - 1), default=None)
    sentence_start = min((offsets[1] for _, offsets in repeated), default=None)

    # Whitespace units include symbols: word-only units miss ASCII/arrow loops.
    matches = list(re.finditer(r"\S+", text))
    tokens = [m.group() for m in matches]
    seen, intervals = {}, []
    ngram_start, max_copies = None, 0
    for i in range(max(0, len(tokens) - NGRAM + 1)):
        key = tuple(tokens[i:i + NGRAM])
        if key not in seen:
            seen[key] = [i, None, i, 1]  # first, second, latest non-overlap, count
            continue
        state = seen[key]
        if i < state[2] + NGRAM:
            continue
        state[2], state[3] = i, state[3] + 1
        max_copies = max(max_copies, state[3])
        if state[3] == 2:
            state[1] = i
        elif state[3] >= 3:
            if state[3] == 3:
                intervals.append((state[1], state[1] + NGRAM))
                onset = matches[state[1]].start()
                ngram_start = onset if ngram_start is None else min(ngram_start, onset)
            intervals.append((i, i + NGRAM))
    duplicate_tokens = covered_length(intervals)
    starts = [x for x in (sentence_start, ngram_start) if x is not None]
    onset = min(starts, default=None)
    return {
        "normalized_unit_chars": denominator,
        "eligible_sentence_occurrences": sum(map(len, occurrences.values())),
        "sentence_repeated_extra_chars": duplicate_chars,
        "sentence_repeat_ratio": duplicate_chars / denominator if denominator else 0,
        "sentence_max_occurrences": max(map(len, occurrences.values()), default=0),
        "sentence_repeat_onset_char": sentence_start,
        "whitespace_units": len(tokens),
        "ngram_repeated_extra_units": duplicate_tokens,
        "ngram_repeat_ratio": duplicate_tokens / len(tokens) if tokens else 0,
        "ngram_max_nonoverlap_copies": max_copies,
        "ngram_repeat_onset_char": ngram_start,
        "repeat_onset_char": onset,
        "repeat_onset_fraction": onset / len(text) if onset is not None else None,
        "dominant_sentence": dominant[0] if dominant else None,
        "dominant_sentence_occurrences": len(dominant[1]) if dominant else 0,
        "dominant_sentence_first_char": dominant[1][0] if dominant else None,
    }


def flagged(row, threshold=0.20):
    return max(row["sentence_repeat_ratio"], row["ngram_repeat_ratio"]) >= threshold


def aggregate(rows):
    count = len(rows)
    ratio = lambda n: n / count if count else None
    onsets = [r["repeat_onset_fraction"] for r in rows if r["repeat_onset_fraction"] is not None]
    correct = sum(r["correct"] for r in rows)
    length = sum(r["finish_reason"] == "length" for r in rows)
    failure = sum(r["status"] == "extraction_failure" for r in rows)
    return {"total": count, "correct": correct, "incorrect": count - correct,
            "accuracy": ratio(correct), "length": length, "length_rate": ratio(length),
            "extraction_failures": failure, "extraction_failure_rate": ratio(failure),
            "output_tokens_total": sum(r["output_tokens"] for r in rows),
            "output_tokens_median": median(r["output_tokens"] for r in rows) if rows else None,
            "sentence_repeat_ratio_median": median(r["sentence_repeat_ratio"] for r in rows) if rows else None,
            "ngram_repeat_ratio_median": median(r["ngram_repeat_ratio"] for r in rows) if rows else None,
            "repeat_onset_fraction_median": median(onsets) if onsets else None}


def analyze(source):
    paths = {"raw": source / "input/raw.jsonl", "metadata": source / "input/run_metadata.json",
             "scores": source / "results/scores.json", "items": source / "results/item_results.jsonl"}
    hashes = {name: sha(path) for name, path in paths.items()}
    scores = json.loads(paths["scores"].read_text(encoding="utf-8"))
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    require(scores["source_sha256"] == hashes["raw"] == metadata["run"]["raw_file"]["sha256"],
            "Raw, score and metadata provenance mismatch")
    raw, saved = read_rows(paths["raw"]), read_rows(paths["items"])
    saved = {r["id"]: r for r in saved}
    require(len(raw) == 900 and {r["id"] for r in raw} == set(saved), "Expected identical 900 ID sets")
    require(set(Counter(r["subject"] for r in raw).values()) == {30}, "Expected 30 items per subject")
    rows = []
    for r in raw:
        result = saved[r["id"]]
        require(all(r[key] == result[key] for key in ("subject", "question_type", "finish_reason", "output_tokens", "gold")),
                "Raw/score item mismatch: " + r["id"])
        require(type(result["correct"]) is bool and result["status"] in
                ("complete_rule", "complete_judge", "extraction_failure"), "Pending/invalid score")
        require(r["status"] == "ok" and isinstance(r["raw_text"], str) and bool(r["raw_text"].strip()), "Invalid raw")
        require(len(r["output_token_ids"]) == r["output_tokens"], "Token count mismatch")
        measured = repetition(r["raw_text"])
        row = {**{k: result[k] for k in ("id", "subject", "question_type", "finish_reason", "output_tokens", "correct", "status", "route")},
               "raw_text_sha256": hashlib.sha256(r["raw_text"].encode()).hexdigest(), **measured}
        row["repetition_flag_20pct"] = flagged(row)
        onset = measured["repeat_onset_char"]
        row["onset_excerpt"] = r["raw_text"][max(0, onset - 100):onset + 260] if onset is not None else None
        rows.append(row)
    rows.sort(key=lambda r: r["id"])
    overall = aggregate(rows)
    require(all(overall[k] == scores["final"]["overall"][k] for k in ("total", "correct", "incorrect", "extraction_failures")),
            "Saved aggregate mismatch")
    require(Counter(r["finish_reason"] for r in rows) == metadata["run"]["finish_reasons"], "Finish counts differ")
    sensitivity = {str(t): {"flagged": aggregate([r for r in rows if flagged(r, t)]),
                            "below_threshold": aggregate([r for r in rows if not flagged(r, t)])} for t in THRESHOLDS}
    by_finish = {finish: {"flagged": aggregate([r for r in rows if r["finish_reason"] == finish and flagged(r)]),
                          "below_threshold": aggregate([r for r in rows if r["finish_reason"] == finish and not flagged(r)])}
                 for finish in ("length", "stop")}
    summary = {"analysis_version": "exact_repetition_v1", "source_sha256": hashes["raw"], "input_sha256": hashes,
               "script_sha256_lf": hashlib.sha256(Path(__file__).read_text(encoding="utf-8").encode()).hexdigest(),
               "policy_id": scores["policy_id"], "judge_settings": scores["judge_settings"],
               "additional_api_calls": 0, "new_inference_runs": 0,
               "metrics": {"sentence_min_chars": MIN_CHARS, "sentence_min_word_runs": MIN_WORDS,
                           "ngram_whitespace_units": NGRAM, "minimum_occurrences": 3,
                           "primary_threshold": 0.20, "thresholds": THRESHOLDS,
                           "flag": "sentence_repeat_ratio >= threshold OR ngram_repeat_ratio >= threshold",
                           "onset": "Second occurrence of a unit/window observed at least three times; original character offset / text length; retrospective, not generation token index."},
               "overall": overall, "threshold_sensitivity": sensitivity, "by_finish": by_finish,
               "detectors": {"sentence": sum(r["sentence_repeat_ratio"] >= .2 for r in rows),
                             "ngram": sum(r["ngram_repeat_ratio"] >= .2 for r in rows),
                             "both": sum(min(r["sentence_repeat_ratio"], r["ngram_repeat_ratio"]) >= .2 for r in rows)},
               "by_subject": {s: {"all": aggregate([r for r in rows if r["subject"] == s]),
                                    "flagged": aggregate([r for r in rows if r["subject"] == s and flagged(r)])}
                              for s in sorted({r["subject"] for r in rows})},
               "by_type": {s: {"flagged": aggregate([r for r in rows if r["question_type"] == s and flagged(r)]),
                                 "below_threshold": aggregate([r for r in rows if r["question_type"] == s and not flagged(r)])}
                           for s in sorted({r["question_type"] for r in rows})}}
    require(all(sha(path) == hashes[name] for name, path in paths.items()), "Input changed during analysis")
    return summary, rows


def pct(value):
    return "—" if value is None else f"{value * 100:.2f}%"


def report(s, rows):
    groups = s["threshold_sensitivity"]["0.2"]
    lines = ["# 최신 raw 900문항 반복·중단 분석 — 2026-09-24", "",
             f"최신 900개 응답의 반복 패턴과 v2 채점 결과를 분석했다. 전체 정답률은 {s['overall']['correct']}/900 ({pct(s['overall']['accuracy'])})다.", "",
             f"원본 SHA-256: `{s['source_sha256']}`.", "",
             "## 1. 자동 반복 후보와 결과", "",
             "아래 ‘반복 후보’는 두 반복 비율 중 하나가 20% 이상인 응답이다.", "",
             "| 집단 | 문항 | 정답 / 정확도 | length / 비율 | 추출 실패 / 비율 | 생성 토큰 중앙값 |",
             "|---|---:|---:|---:|---:|---:|"]
    def table_row(name, g):
        return f"| {name} | {g['total']} | {g['correct']} / {pct(g['accuracy'])} | {g['length']} / {pct(g['length_rate'])} | {g['extraction_failures']} / {pct(g['extraction_failure_rate'])} | {g['output_tokens_median']} |"
    lines += [table_row("전체", s["overall"]), table_row("반복 후보", groups["flagged"]), table_row("기준 미만", groups["below_threshold"])]
    f = groups["flagged"]
    lines += ["", f"반복 후보가 전체 오답에서 차지하는 비율은 {f['incorrect']}/{s['overall']['incorrect']} ({pct(f['incorrect']/s['overall']['incorrect'])}), 추출 실패에서는 {f['extraction_failures']}/{s['overall']['extraction_failures']} ({pct(f['extraction_failures']/s['overall']['extraction_failures'])})다.",
              f"긴 문장 지표 후보 {s['detectors']['sentence']}건, 16단위 지표 후보 {s['detectors']['ngram']}건, 두 지표 공통 {s['detectors']['both']}건이다.",
              f"반복 후보의 반복 시작 위치 중앙값은 응답 문자 길이의 {pct(f['repeat_onset_fraction_median'])}다.", "",
              "## 2. 종료 사유를 나눈 비교", "",
              "| 집단 | 문항 | 정답 / 정확도 | length / 비율 | 추출 실패 / 비율 | 생성 토큰 중앙값 |",
              "|---|---:|---:|---:|---:|---:|"]
    for finish, groups_by_finish in s["by_finish"].items():
        for name, g in groups_by_finish.items():
            lines.append(table_row(f"{finish} · {'반복 후보' if name == 'flagged' else '기준 미만'}", g))
    lines += ["", "## 3. 임계값 민감도", "", "| 기준 | 후보 수 | 후보 정확도 | 기준 미만 정확도 | 후보 추출 실패율 |", "|---|---:|---:|---:|---:|"]
    for threshold, groups_at_threshold in s["threshold_sensitivity"].items():
        a, b = groups_at_threshold["flagged"], groups_at_threshold["below_threshold"]
        lines.append(f"| {float(threshold):.0%} | {a['total']} | {pct(a['accuracy'])} | {pct(b['accuracy'])} | {pct(a['extraction_failure_rate'])} |")
    lines += ["", "## 4. 사례", "", "높은 반복률과 짧은 구절·기호 반복을 보여주는 사례를 선정했다.", "",
              "| 문항 | 문장 비율 | 16단위 비율 | 반복 시작 | 종료 | 정책 정오 |", "|---|---:|---:|---:|---|---|"]
    selected = sorted(rows, key=lambda r: (-max(r["sentence_repeat_ratio"], r["ngram_repeat_ratio"]), r["id"]))[:3]
    examples = {r["id"]: r for r in selected}
    for id_ in ("validation_Accounting_5", "validation_Computer_Science_16", "validation_Music_24", "validation_Music_27"):
        examples[id_] = next(r for r in rows if r["id"] == id_)
    for r in rows:
        if flagged(r) and r["correct"] and r["finish_reason"] == "stop":
            examples[r["id"]] = r
            break
    for r in examples.values():
        lines.append(f"| {r['id']} | {pct(r['sentence_repeat_ratio'])} | {pct(r['ngram_repeat_ratio'])} | {pct(r['repeat_onset_fraction'])} | {r['finish_reason']} | {'정답' if r['correct'] else '오답'} |")
    lines += ["", "## 5. 계산 정의와 해석", "",
              "- 문장/행: 줄바꿈 또는 문장부호(.!?。！？) 뒤 공백에서 나누고 내부 공백만 하나로 정규화한다. 대소문자·숫자·기호는 유지한다. 정규화 후 40자 이상·단어 구성요소(\\w+) 8개 이상인 단위만 후보로 삼는다.",
              "- 문장 반복 비율: 같은 긴 단위가 3회 이상 등장하면 첫 등장을 제외한 중복 단위 문자 수를 합산해, 짧은 단위도 포함한 전체 정규화 단위 문자 수로 나눈다. 동일 문자열의 재등장을 측정한다.",
              "- 16단위 반복 비율: 공백으로 나눈 단위(기호 포함) 16개의 동일한 연속열이 서로 겹치지 않게 3회 이상 등장하면 첫 등장을 제외한 복제 구간을 표시한다. 여러 창의 겹침은 합집합으로 한 번만 세고, 표시된 공백 단위 수 / 전체 공백 단위 수를 사용한다.",
              "- 반복 시작: 완성된 응답에서 위 조건을 만족하는 단위의 두 번째 등장 중 가장 빠른 원문 문자 위치 / 전체 원문 문자 수다.",
              "- 탐지 기준: 두 반복 비율 중 하나가 20% 이상이면 후보로 표시하고, 10%·30% 기준의 결과를 함께 비교한다. 수식·선택지 재인용·도식의 규칙적 패턴도 탐지 대상에 포함된다.",
              "- 정확도·추출 실패는 v2 하이브리드 채점 결과이며, length는 생성 상한 종료 기록이다. 반복과 정답률의 연관을 종료 사유별로 비교했다.", "",
              "## 재현", "", "Python 3.10 이상 표준 라이브러리만 사용한다. 저장소 루트에서 새 출력 경로를 지정한다.", "", "```bash",
              "python assignment/experiments/repetition/analyze.py --out assignment/experiments/repetition/replay",
              "python assignment/experiments/repetition/analyze.py --self-test", "```", "",
              "다른 실행에는 같은 구조의 검증된 제출 묶음을 `--source`로 지정한다. 이번 코드는 동일 900문항 평가용이다.",
              "입력 해시·설정·30과목/문항 유형별 집계는 [summary.json](summary.json), 원문 위치·대표 반복 문장·900건 지표는 [per_item.jsonl](per_item.jsonl)에 있다."]
    return "\n".join(lines) + "\n"


def self_test():
    sentence = "This is one sufficiently long sentence with eight separate words and a decimal 3.14."
    assert repetition(sentence)["sentence_repeat_ratio"] == 0
    assert repetition((sentence + "\n") * 2)["sentence_repeat_ratio"] == 0
    r = repetition((sentence + "\n") * 3)
    assert abs(r["sentence_repeat_ratio"] - 2/3) < 1e-12
    assert r["sentence_repeat_onset_char"] == len(sentence) + 1
    assert repetition("Yes.\n" * 8)["sentence_repeat_ratio"] == 0
    assert repetition(" ")["repeat_onset_char"] is None
    assert repetition(sentence + "\n" + sentence.replace("3.14", "3.15") + "\n" + sentence.replace("3.14", "3.16"))["sentence_repeat_ratio"] == 0
    symbols = "| " * 128
    r = repetition(symbols)
    assert r["sentence_repeat_ratio"] == 0 and r["ngram_repeat_ratio"] > .8
    assert 0 <= r["ngram_repeat_ratio"] <= 1
    unique = " ".join(f"word{i}" for i in range(16))
    assert repetition(unique + " " + unique)["ngram_repeat_ratio"] == 0
    assert abs(repetition((unique + " ") * 3)["ngram_repeat_ratio"] - 2/3) < 1e-12
    assert covered_length([(2, 5), (3, 7), (10, 12)]) == 7
    print("Self-test passed: 3-copy minimum, decimals, numeric changes, symbols, offsets, overlap union.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    require(args.out is not None, "Specify a new --out directory")
    out, source = args.out.resolve(), args.source.resolve()
    require(not source.is_relative_to(out), "Output must not contain the source")
    require(not out.exists() or not any(out.iterdir()), "Use an empty output directory")
    summary, rows = analyze(source)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "per_item.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (out / "REPORT.md").write_text(report(summary, rows), encoding="utf-8")
    print(json.dumps({"overall": summary["overall"], "repetition_20pct": summary["threshold_sensitivity"]["0.2"],
                      "detectors": summary["detectors"], "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
