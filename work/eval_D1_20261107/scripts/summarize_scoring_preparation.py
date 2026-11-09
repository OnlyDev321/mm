"""Read prepared v2 routes and estimate the same conservative budget as HypB."""
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'scoring/D1'
RAW_SHA = json.loads((ROOT/'reports/TRANSFER_CHECK.json').read_bytes())['raw_sha256']


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line]


def main():
    raw_path = ROOT/'runs/D1/generation/raw.jsonl'
    assert sha(raw_path) == sha(OUT/'input/raw.jsonl') == RAW_SHA
    manifest = json.loads((OUT/'manifest.json').read_text())
    summary = json.loads((OUT/'summary.json').read_text())
    assert manifest['policy_id'] == 'hybrid100_mc_qwen_ab_open_no_length_gate_v2'
    assert manifest['judge_settings'] == {'model':'gpt-4.1-mini-2025-04-14',
        'temperature':0,'top_p':1,'seed':3407,'max_tokens':8}
    assert sha(OUT/'input/config.yaml') == sha(ROOT/'source/code/config.yaml') == manifest['config_sha256']
    assert sha(OUT/'items.jsonl') == manifest['items_sha256']
    assert sha(OUT/'requests.jsonl') == manifest['requests_sha256']
    raw, items, requests = rows(raw_path), rows(OUT/'items.jsonl'), rows(OUT/'requests.jsonl')
    by_id = {item['id']:item for item in items}
    assert len(by_id) == len(raw) == 900
    assert {r['id'] for r in raw} == set(by_id)
    assert {r['id'] for r in requests} == {i['id'] for i in items if i['route']!='auto'}
    assert summary['response_count'] == 0 and summary['overall']['accuracy'] is None
    assert summary['overall']['pending'] == len(requests)
    assert not (OUT/'responses.jsonl').exists() and not (OUT/'api_attempts.jsonl').exists()

    def group(selected):
        counts = Counter(by_id[r['id']]['route'] for r in selected)
        result = {'total':len(selected),'automatic':counts['auto'],
            'judge_total':counts['judge_rule_failure']+counts['judge_conflict'],
            'rule_extraction_failure':counts['judge_rule_failure'],'conflict':counts['judge_conflict']}
        assert result['automatic']+result['judge_total'] == result['total']
        return result

    length = [r for r in raw if r['finish_reason']=='length']
    closed = [r for r in length if '</think>' in r['raw_text']]
    unclosed = [r for r in length if '</think>' not in r['raw_text'] and '<think>' in r['raw_text']]
    no_marker = [r for r in length if '</think>' not in r['raw_text'] and '<think>' not in r['raw_text']]
    groups = {'all':group(raw),'stop':group([r for r in raw if r['finish_reason']=='stop']),
              'length':group(length),'length_think_closed':group(closed),
              'length_think_unclosed':group(unclosed),'length_no_think_marker':group(no_marker)}
    for key in ['total','automatic','judge_total','rule_extraction_failure','conflict']:
        assert groups['length'][key] == groups['length_think_closed'][key]+groups['length_think_unclosed'][key]+groups['length_no_think_marker'][key]
    assert groups['all']['automatic'] == manifest['automatic_count']
    assert groups['all']['judge_total'] == manifest['judge_count']
    prices = manifest['price_per_million_usd']
    assert prices == {'input':0.4,'cached_input':0.1,'output':1.6}
    payload_bytes = sum(len(json.dumps(r['request'],ensure_ascii=False,sort_keys=True,
                                      separators=(',',':')).encode('utf-8')) for r in requests)
    input_bound = payload_bytes+len(requests)*128
    output_bound = sum(r['request']['max_tokens'] for r in requests)
    budget = (Decimal(input_bound)*Decimal(str(prices['input']))+
              Decimal(output_bound)*Decimal(str(prices['output'])))/Decimal(1_000_000)
    result = {'status':'PASS','source_sha256':RAW_SHA,'policy_id':manifest['policy_id'],
        'judge_settings':manifest['judge_settings'],'groups':groups,'api_calls':0,
        'pending':summary['overall']['pending'],'accuracy':None,
        'final_judge_extraction_failures':None,
        'closure_rule':'literal </think> present (same as E3), <think> without closing tag, or neither marker',
        'budget':{'expected_calls':len(requests),'request_json_utf8_bytes':payload_bytes,
            'input_tokens_conservative_bound':input_bound,'output_tokens_max':output_bound,
            'price_per_million_usd':prices,'conservative_cost_usd':str(budget),
            'method':'same as HypB: canonical request JSON UTF-8 bytes + 128 input tokens/request; no cached-input discount; max output 8/request',
            'pricing_source':'https://developers.openai.com/api/docs/models/gpt-4.1-mini',
            'pricing_checked_date_kst':'2026-11-07'},
        'artifact_sha256':{name:sha(OUT/name) for name in ['manifest.json','items.jsonl','requests.jsonl','summary.json']}}
    target = ROOT/'reports/SCORING_PREPARATION_SUMMARY.json'
    assert not target.exists()
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
