"""Audit D1 Judge receipts, then apply the prespecified offline D1 decision."""

from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
REPORTS = ROOT / 'reports'
SCORES = ROOT / 'scoring/D1'
RAW = ROOT / 'runs/D1/generation/raw.jsonl'
M0 = WORK.parent / 'assignment/experiments/received_20260922/results/input/raw.jsonl'
IDS_SHA = '54c8ec398206406e8a81ea7a4ffe97169bb90f21ef159c6ddd5a73b1edfd2d9f'
BASELINES = {
    'M0': (M0, WORK / 'team-evaluator-no-length/assignment/experiments/raw_evaluation/outputs/release-v2-latest', 600),
    'E1': (WORK / 'eval_B_20261101/runs/E1/generation/raw.jsonl', WORK / 'eval_B_20261101/scoring/E1', 527),
    'E3': (WORK / 'eval_E3_20261101/runs/E3/generation/raw.jsonl', WORK / 'eval_E3_20261101/scoring/E3', 597),
}
BASELINE_SHA = {
    'M0': 'ef23f0c49d9b1ae6c62b9625fbd52cce474a0c035c1a644cfa737e0967daa313',
    'E1': 'a6e307b88a0c8ab6afd4a76d6706004736ec4b8319a94a014e54fe0faa43e317',
    'E3': 'cf0aba5b6941d46620244e3a57653c944aed0e1eb5b47a54136a7a4c39376722',
}


def load(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(name):
    path = SCORES / name
    return [json.loads(line) for line in path.read_bytes().splitlines() if line] if path.is_file() else []


def save(name, value):
    (REPORTS / name).write_text(value, encoding='utf-8', newline='\n')


def table(header, records):
    def line(values):
        return '| ' + ' | '.join(map(str, values)) + ' |'
    return '\n'.join([line(header), line(['---'] * len(header)), *(line(row) for row in records)])


def acc(value):
    return f"{value['correct']}/{value['total']} ({value['accuracy']:.2%})" if value['accuracy'] is not None else '미확정'


def scoring():
    launch = load(ROOT / 'scoring/judge_launcher/status.json')
    assert launch['state'] in {'complete', 'failed'} and launch['stage'] == 'finished'
    prepared = load(REPORTS / 'SCORING_PREPARATION_SUMMARY.json')
    for name in ('manifest.json', 'items.jsonl', 'requests.jsonl'):
        assert sha(SCORES / name) == prepared['artifact_sha256'][name], name
    assert sha(RAW) == sha(SCORES / 'input/raw.jsonl') == prepared['source_sha256']
    manifest, summary = (load(SCORES / name) for name in ('manifest.json', 'summary.json'))
    assert manifest['judge_settings'] == prepared['judge_settings']
    requests, responses, events, receipts = map(rows, ('requests.jsonl', 'responses.jsonl', 'api_attempts.jsonl', 'http_receipts.jsonl'))
    ids = {r['id'] for r in requests}
    assert len(ids) == len(requests) == 316
    started = Counter(r['id'] for r in events if r['event'] == 'started')
    received = Counter(r['id'] for r in events if r['event'] == 'received')
    assert all(count == 1 for count in started.values()) and started.keys() <= ids
    assert received == Counter(r['id'] for r in responses)
    http_ok = {r['id'] for r in receipts if r['http_status'] == 200}
    assert len(http_ok) == len(receipts) and received.keys() <= http_ok <= started.keys()
    assert len(responses) == summary['response_count']
    assert all(r['response']['model'] == manifest['judge_settings']['model'] for r in responses)
    pending = summary['overall']['pending']
    assert pending == 316 - len(responses) == launch['pending']['D1']
    if pending:
        assert summary['overall']['accuracy'] is None
    else:
        assert launch['state'] == 'complete' and len(started) == len(received) == 316
        progress = (ROOT / 'scoring/judge_launcher/progress.log').read_text(encoding='utf-8')
        assert progress.index('running D1_pilot') < progress.index('pilot_passed D1_pilot') < progress.index('running D1_full') < progress.index('running D1_summarize')

    usage, finishes = Counter(), Counter()
    for saved in responses:
        response = saved['response']
        counts = response['usage']
        assert counts['total_tokens'] == counts['prompt_tokens'] + counts['completion_tokens']
        usage.update(input=counts['prompt_tokens'], cached_input=(counts.get('prompt_tokens_details') or {}).get('cached_tokens', 0),
                     output=counts['completion_tokens'])
        finishes[response['choices'][0]['finish_reason']] += 1
    prices = {key: Decimal(str(value)) for key, value in manifest['price_per_million_usd'].items()}
    actual = ((usage['input'] - usage['cached_input']) * prices['input'] + usage['cached_input'] * prices['cached_input']
              + usage['output'] * prices['output']) / Decimal(1_000_000)
    assert abs(actual - Decimal(str(summary['estimated_recorded_response_cost_usd']))) < Decimal('0.000000001')
    failures = Counter(str(row.get('http_status') or row.get('error_type')) for row in events if row['event'] == 'failed')
    cohort = table(['범위', '정확도', '완료', 'pending'],
                   [(name, acc(value), value['complete'], value['pending']) for name, value in [('전체', summary['overall']), *summary['by_type'].items()]])
    record = {'api_attempts': len(started), 'http_200': len(receipts), 'pending': pending, 'failures': failures,
              'recorded_usage_cost_usd': str(actual), 'usage': dict(usage), 'finish_reasons': dict(finishes),
              'summary_sha256': sha(SCORES / 'summary.json'), 'scores_sha256': sha(SCORES / 'final_results.jsonl')}
    save('JUDGE_COMPLETION_CHECK.json', json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    save('SCORING_REPORT.md', f"""# D1 v2 Judge 채점 보고

고정 `gpt-4.1-mini-2025-04-14`, temperature 0, top_p 1, seed 3407, max_tokens 8로 실행했다. 5건 확인 후 남은 요청을 실행하는 E3 런처 흐름을 사용했다. API 키는 사용자 입력으로 런처 프로세스와 채점 자식 프로세스에만 전달했고, summarize 전에 환경에서 제거했다.

{cohort}

- 자동 **{summary['automatic_count']}건**, Judge 대상 **316건**(규칙 미추출 314, 충돌 2), Judge 응답 **{len(responses)}건**, pending **{pending}건**.
- API 시도 **{len(started)}회**, HTTP 200 **{len(receipts)}회**, 기록된 API 실패 **{sum(failures.values())}건**. 실패 분류 `{dict(failures)}`. 중복 호출 0회.
- 기록된 usage는 입력 {usage['input']:,}토큰(캐시 {usage['cached_input']:,}), 출력 {usage['output']:,}토큰이다. usage 기준 비용 **${actual:.8f}**이고 승인한 보수적 예산은 $1.49였다.
- 최종 추출 실패는 현재 완료분에서 **{summary['overall']['extraction_failures']}건**이다. Judge 응답의 종료 사유는 `{dict(finishes)}`다. 실패한 API 요청은 pending에 남긴다.
- raw SHA-256 `{prepared['source_sha256']}`와 요청 해시를 재확인했다. `scoring/D1/`에 summarize 결과를, `reports/JUDGE_COMPLETION_CHECK.json`에 대조 결과를 보관했다.
""")
    print(json.dumps(record, ensure_ascii=False))
    return pending


def judgment(pending):
    sys.path.insert(0, str(ROOT / 'source/assignment/experiments/hypB_eval/yoonseok'))
    import aggregate as agg

    models = {}
    ids_source = WORK / 'eval_E3_20261101/reference/m0_length_ids.txt'
    assert sha(ids_source) == IDS_SHA
    ids_path = ROOT / 'reference/m0_length_ids.txt'
    if not ids_path.exists():
        ids_path.write_bytes(ids_source.read_bytes())
    assert sha(ids_path) == IDS_SHA
    ids = ids_path.read_text(encoding='utf-8').splitlines()
    assert len(ids) == len(set(ids)) == 128
    d1_raw, d1_sha = agg.raw_rows(RAW)
    m0_raw, _ = agg.raw_rows(M0)
    assert set(ids) == {row['id'] for row in m0_raw if row['finish_reason'] == 'length'}
    direction = sum(len(agg.MARKERS.findall(row['raw_text'])) >= 21 for row in d1_raw)
    assert sum(len(agg.MARKERS.findall(row['raw_text'])) >= 21 for row in m0_raw) == 217
    length = sum(row['finish_reason'] == 'length' for row in d1_raw)
    length_pass = length < 128
    direction_pass = 100 * direction < 24.1 * 900
    flag = lambda value: '통과' if value else '불통과'
    if pending:
        save('JUDGMENT_REPORT.md', f"""# D1 최종 판정 보류

Judge pending **{pending}건**으로 전체 v2 정확도는 `null`이다. 1차 128문항 판정, 안전 기준, 최종 판정은 보류한다.

생성 지표는 length **{length}/900**으로 `<128` {flag(length_pass)}, 방향 전환 과다 **{direction}/900 ({direction/900:.2%})**로 `<24.1%` {flag(direction_pass)}다. API 실패 항목은 pending 상태로 보존했다.
""")
        return

    _, d1_scores = agg.score_rows(SCORES, d1_raw, d1_sha)
    comparisons = {}
    pair_rows = {}
    for name, (raw, scores, expected_correct) in BASELINES.items():
        comparison = agg.compare(raw, scores, RAW, SCORES, ids_path, 10)
        assert comparison['left']['raw_sha256'] == BASELINE_SHA[name]
        assert comparison['left']['v2']['overall']['correct'] == expected_correct
        assert comparison['safety_900_v2']['pending_pairs'] == 0
        comparisons[name] = comparison
        models[name] = comparison['left']
        raw_rows, digest = agg.raw_rows(raw)
        _, pair_rows[name] = agg.score_rows(scores, raw_rows, digest)
    models['D1'] = comparisons['M0']['right']
    pair_rows['D1'] = d1_scores
    primary = comparisons['M0']['primary_128_v2']
    safety = comparisons['M0']['safety_900_v2']
    accuracy_pass = safety['accuracy_delta_pp'] >= -1.7
    gains_pass = safety['losses'] <= safety['gains']
    primary_pass = primary['net_gain'] >= 14
    ref_pass = primary['net_gain'] >= 18
    all_pass = all((primary_pass, accuracy_pass, gains_pass, length_pass, direction_pass))
    fixed = {name: sum(scores[item]['final_correct'] for item in ids) for name, scores in pair_rows.items()}
    assert fixed['D1'] - fixed['M0'] == primary['net_gain']
    criteria = table(['기준', 'D1 결과', '판정'], [
        ('1차 net_gain ≥14', f"{primary['net_gain']:+d} (획득 {primary['gains']} / 상실 {primary['losses']})", flag(primary_pass)),
        ('1차 참고 ≥18', f"{primary['net_gain']:+d}", flag(ref_pass)),
        ('안전 정확도 하락 ≤1.7%p', f"{safety['accuracy_delta_pp']:+.2f}%p", flag(accuracy_pass)),
        ('안전 상실 ≤ 획득', f"상실 {safety['losses']} / 획득 {safety['gains']}", flag(gains_pass)),
        ('메커니즘 length <128', f'{length}/900', flag(length_pass)),
        ('메커니즘 방향 전환 과다율 <24.1%', f'{direction}/900 ({direction/900:.2%})', flag(direction_pass)),
    ])

    def rollup(model):
        value = models[model]
        route = value['processing_routes']
        overall = value['v2']['overall']
        tokens = value['output_tokens']
        return (model, acc(overall), f"{fixed[model]}/128", value['length']['count'], value['direction_overuse']['count'],
                value['repetition_candidates']['count'], f"{tokens['mean']:,.2f} / {tokens['median']:,.1f}",
                route.get('auto', 0), route.get('judge_rule_failure', 0), route.get('judge_conflict', 0), overall['extraction_failures'])

    summary = table(['모델', 'v2 정확도', '고정 128 정답', 'length', '방향 과다', '반복 후보', '출력 토큰 평균 / 중앙값',
                     '자동', '규칙 미추출', '충돌', '최종 추출 실패'], [rollup(name) for name in models])
    route_transitions = Counter((pair_rows['M0'][item]['route'] == 'auto', d1_scores[item]['route'] == 'auto') for item in d1_scores)
    route_table = table(['M0 경로', 'D1 자동', 'D1 Judge'], [
        (label, route_transitions[old, True], route_transitions[old, False]) for old, label in ((True, '자동'), (False, 'Judge'))])
    technical = table(['비교', '전체 정확도 차이', '전체 획득 / 상실', '고정 128 net_gain'], [
        (f'{name} → D1', f"{r['safety_900_v2']['accuracy_delta_pp']:+.2f}%p",
         f"{r['safety_900_v2']['gains']} / {r['safety_900_v2']['losses']}", f"{r['primary_128_v2']['net_gain']:+d}")
        for name, r in comparisons.items()])
    grouped = {}
    rendered = []
    for label, classify in [('유형', lambda r: r['question_type']), ('분야', lambda r: agg.CATEGORIES[r['subject']]), ('과목', lambda r: r['subject'])]:
        data = []
        for name in sorted({classify(r) for r in d1_raw}):
            keys = [r['id'] for r in d1_raw if classify(r) == name]
            scores = {model: sum(result[item]['final_correct'] for item in keys) for model, result in pair_rows.items()}
            delta = scores['D1'] - scores['M0']
            paired = agg.paired(pair_rows['M0'], d1_scores, keys)
            assert paired['net_gain'] == delta
            data.append({'group': name, 'n': len(keys), 'correct': scores, 'paired_vs_M0': paired})
        assert sum(row['paired_vs_M0']['net_gain'] for row in data) == safety['net_gain']
        grouped[label] = data
        rendered.append(f"### {label}\n\n" + table([label, '문항', 'M0 정답', 'E1 정답', 'E3 정답', 'D1 정답', 'D1−M0 정답'], [
            (r['group'], r['n'], *(r['correct'][model] for model in ('M0', 'E1', 'E3', 'D1')), f"{r['paired_vs_M0']['net_gain']:+d}") for r in data]))

    save('JUDGMENT_REPORT.md', f"""# D1 평가 최종 판정

**종합 {flag(all_pass)}.** 학습 전에 고정한 `PLAN_D1.md` 6절을 완료된 900문항 v2 레이블에 적용했다. 비교 기준은 M0다.

{criteria}

M0 length 128문항 목록의 SHA-256은 `{IDS_SHA}`이며, 원본 M0에서 `length`로 끝난 ID 집합과 일치했다. 1차의 k0는 10이다. ≥18은 k0=14에 대한 참고 결과다. D1 raw SHA-256은 `{d1_sha}`다.

## 모델별 지표와 채점 경로

{summary}

방향 전환 과다는 고정된 8개 표지의 합이 21회 이상인 문항 수다. 규칙 미추출과 충돌의 합이 Judge 대상이다. 경로별 문항 구성이 달라지므로 자동 처리 비율 변화만으로 학습 효과를 해석하지 않는다.

{route_table}

## E1·E3와 기술 비교

{technical}

E1·E3 비교는 같은 900개 입력에서의 기술 비교다. 최종 판정은 M0 대비 기준으로 계산했다.

## 유형·분야·과목별 변화

{chr(10).join(rendered)}

이 집계는 로컬에서 저장된 raw와 완료 채점 결과로 계산했다. 판정용 추가 API 호출은 0회다.
""")
    agg.write_json(REPORTS / 'JUDGMENT_SUMMARY.json', {'primary': primary, 'safety': safety, 'length': length, 'direction_overuse': direction,
                   'overall_pass': all_pass, 'frozen_ids_sha256': IDS_SHA, 'models': models, 'comparisons': comparisons, 'by_group': grouped})
    print(json.dumps({'overall_pass': all_pass, 'primary': primary, 'safety': safety, 'length': length, 'direction': direction}, ensure_ascii=False))


if __name__ == '__main__':
    judgment(scoring())
