"""Enter a key only in a real console, then run the frozen v2 evaluator."""
import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import warnings

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT / 'source/assignment/experiments/hypB_eval/yoonseok'))
from verify_code import ROOT, verify

EVALUATOR = ROOT / 'code/scoring/evaluate.py'
EXPECTED_RAW = {'D1': '60ee87670752b663edacf7a42f24320f97c33198741a6b1465594939361ba23e'}

SETTINGS = {'model': 'gpt-4.1-mini-2025-04-14', 'temperature': 0,
            'top_p': 1, 'seed': 3407, 'max_tokens': 8}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def estimate_cost(scoring_root):
    total = 0.0
    for run, expected in EXPECTED_RAW.items():
        folder = scoring_root / run
        manifest = read_json(folder / 'manifest.json')
        assert manifest['policy_id'] == 'hybrid100_mc_qwen_ab_open_no_length_gate_v2'
        assert manifest['judge_settings'] == SETTINGS
        assert manifest['price_per_million_usd'] == {'input': 0.4, 'cached_input': 0.1, 'output': 1.6}
        assert hashlib.sha256((folder / 'input/raw.jsonl').read_bytes()).hexdigest() == manifest['source_sha256'] == expected
        prepared = read_json(TASK_ROOT / 'reports/SCORING_PREPARATION_SUMMARY.json')
        for name in ('manifest.json', 'items.jsonl', 'requests.jsonl'):
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == prepared['artifact_sha256'][name], name
        data = (folder / 'requests.jsonl').read_bytes()
        assert hashlib.sha256(data).hexdigest() == manifest['requests_sha256']
        requests = [json.loads(line) for line in data.splitlines() if line]
        assert len(requests) == manifest['judge_count'] == 316
        bound = sum(len(json.dumps(r['request'], ensure_ascii=False, sort_keys=True,
                                   separators=(',', ':')).encode('utf-8')) + 128 for r in requests)
        total += (bound * 0.4 + len(requests) * 8 * 1.6) / 1_000_000
    if total > 1.49 + 1e-9:
        raise ValueError('Estimated cost exceeds the approved preparation estimate USD 1.49')
    return total


def pilot_check(folder):
    summary = read_json(folder / 'summary.json')
    target = min(5, summary['judge_count'])
    assert summary['response_count'] >= target
    responses = [json.loads(line) for line in (folder / 'responses.jsonl').read_bytes().splitlines() if line]
    for saved in responses[:target]:
        response = saved['response']
        assert response['model'] == SETTINGS['model'], 'Pilot model snapshot mismatch'
        # The frozen evaluator validates schema and grades non-stop/empty/Z as failures.
        # These are completed judgments, not API errors or reasons to alter scoring.


def workflow(scoring_root, logs=None, check_only=False):
    verify()
    estimate = estimate_cost(scoring_root)
    print(f'Conservative cost ceiling: USD {estimate:.6f} (approved preparation estimate USD 1.49)', flush=True)
    if check_only:
        print('Offline preflight passed; no key entry or API calls.')
        return 0
    logs = logs or scoring_root / 'judge_launcher'
    logs.mkdir(parents=True, exist_ok=True)
    state = {'pid': os.getpid(), 'estimated_ceiling_usd': estimate, 'state': 'starting', 'stage': None}

    def status(phase, stage):
        state.update(state=phase, stage=stage, updated_at=datetime.now(timezone.utc).isoformat())
        temp = logs / 'status.json.tmp'
        temp.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
        temp.replace(logs / 'status.json')
        with (logs / 'progress.log').open('a', encoding='utf-8') as handle:
            handle.write(f"{state['updated_at']} {phase} {stage}\n")
        print(f'{phase}: {stage}', flush=True)

    def execute(label, command, run, *extra):
        status('running', label)
        args = [sys.executable, '-B', '-X', 'utf8', str(EVALUATOR), command,
                '--out', str(scoring_root / run), *extra]
        with (logs / (label + '.log')).open('a', encoding='utf-8') as handle:
            handle.write(f"\n--- {datetime.now(timezone.utc).isoformat()} {label} ---\n")
            handle.flush()
            result = subprocess.run(args, stdout=handle, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError('Evaluator failed; pending results and receipts preserved')

    failed = False
    try:
        status('waiting_for_key', 'key_entry')
        with warnings.catch_warnings():
            warnings.simplefilter('error', getpass.GetPassWarning)
            secret = getpass.getpass('OpenAI API key (hidden, one entry, never saved): ').strip()
        if not secret:
            raise ValueError('No key supplied')
        os.environ['OPENAI_API_KEY'] = secret
        secret = ''
        prior = read_json(scoring_root / 'D1/summary.json')
        needed = max(0, min(5, prior['judge_count']) - prior['response_count'])
        if needed:
            execute('D1_pilot', 'run', 'D1', '--limit', str(needed))
        pilot_check(scoring_root / 'D1')
        status('pilot_passed', 'D1_pilot')
        execute('D1_full', 'run', 'D1')
    except BaseException as exc:
        failed = True
        state['failure_type'] = type(exc).__name__
        state['failure_stage'] = state['stage']
    finally:
        secret = ''
        os.environ.pop('OPENAI_API_KEY', None)

    for run in EXPECTED_RAW:
        try:
            execute(run + '_summarize', 'summarize', run)
        except BaseException as exc:
            failed = True
            state['failure_type'] = type(exc).__name__
            state['failure_stage'] = state['stage']
    summaries = {run: read_json(scoring_root / run / 'summary.json') for run in EXPECTED_RAW}
    state['pending'] = {run: value['overall']['pending'] for run, value in summaries.items()}
    if any(state['pending'].values()):
        failed = True
    status('failed' if failed else 'complete', 'finished')
    if failed:
        print('FAILED / PENDING. Review logs and frozen evaluator receipts before resuming. No automatic retries.')
    else:
        print('COMPLETE. D1 summarized; no paired aggregation or decision was run.')
    return 1 if failed else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scoring-root', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    options = parser.parse_args()
    raise SystemExit(workflow(options.scoring_root.resolve(), check_only=options.check_only))
