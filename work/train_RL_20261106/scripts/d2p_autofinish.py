"""Wait for the existing sampler, validate, and finish D2' stages 4.2–8.

Never starts training, restarts sampling, uploads artifacts, or stops the Pod.
"""
import collections
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path('/workspace/mm/work/train_RL_20261106')
os.chdir(ROOT)
RUN = ROOT / 'runs/d2p_k4'
REPORT = ROOT / 'reports'
REPORT.mkdir(exist_ok=True)

def save(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)

def status(state, **kw):
    data = dict(state=state, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **kw)
    save(RUN / 'autofinish_status.json', data)
    print(json.dumps(data, ensure_ascii=False), flush=True)

def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def command(*args):
    subprocess.run(['python3.12', *args], check=True)

def main():
    lock = (RUN / 'autofinish.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
    assert '4090' in gpu, gpu
    expected = {
        ROOT / 'adapter/d1_full/adapter_model.safetensors': '2d4bcce4972b36b6ef49b0b73d46c9d72a3803f43632b6847ae6f4d65e0132e8',
        Path('/workspace/models/D1/model-00001-of-00002.safetensors'): '31d4561504484269e16e1cc00baef16d694cd44f25c9d282cfd45977caaab351',
        Path('/workspace/models/D1/model-00002-of-00002.safetensors'): '1280742e74baab97037603e9bc2912626f2c85d6347d9762a1286566b4b7f431',
    }
    hashes = {str(p): sha(p) for p in expected}
    assert hashes == {str(p): h for p, h in expected.items()}, 'Weights SHA256 mismatch'
    save(REPORT / 'd2p_weights_verified.json', hashes)
    # The existing sampler writes whole chunks only after engine.generate returns.
    # Do not treat a long interval without new rows as a stalled run.
    while True:
        active = False
        for p in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                argv = p.read_bytes().split(b'\0')
                active |= any(a.endswith(b'/sample_policy.py') or a == b'sample_policy.py' for a in argv)
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                pass
        count = sum(1 for _ in (RUN / 'raw.jsonl').open())
        status('waiting_sampling', responses=count, expected=6000, sampler_active=active)
        if not active:
            assert count == 6000, f'Sampler exited with {count}/6000 responses; no automatic restart'
            break
        time.sleep(30)
    raw = rows(RUN / 'raw.jsonl')
    pool = rows(ROOT / 'data/r1_pool.jsonl')
    lookup = {(str(r['row']), r['repeat']): r for r in raw}
    assert len(raw) == len(lookup) == 6000
    assert set(lookup) == {(str(p['row']), k) for p in pool for k in range(4)}
    assert all(r['reward'] in (0, 1) and r['finish_reason'] in ('stop', 'length') for r in raw)
    status('stage_4_grouping', responses=6000)
    command('scripts/build_groups.py', '--input', 'runs/d2p_k4/raw.jsonl', '--output-groups', 'runs/d2p_k4/groups.json', '--output-train', 'runs/d2p_k4/d2p_train.jsonl', '--seed', '3407')
    save(RUN / 'SHA256SUMS.json', {n: sha(RUN / n) for n in ('raw.jsonl', 'groups.json', 'd2p_train.jsonl')})
    status('stage_5_pairs')
    command('scripts/build_dpo_pairs.py', '--raw', 'runs/d2p_k4/raw.jsonl', '--groups', 'runs/d2p_k4/groups.json', '--train', 'runs/d2p_k4/d2p_train.jsonl', '--output', 'data/d2p_pairs_swift.jsonl', '--index', 'data/d2p_pairs_index.jsonl', '--seed', '3407')
    pairs = rows(ROOT / 'data/d2p_pairs_swift.jsonl')
    indices = rows(ROOT / 'data/d2p_pairs_index.jsonl')
    assert len(pairs) == len(indices)
    if len(pairs) < 300:
        status('stopped_below_300_pairs', pairs=len(pairs), training_prepared=False)
        (REPORT / 'D2P_STAGE1_REPORT.md').write_text(f'# D2\' — STOP\n\nSampling: 6000/6000. Pairs: {len(pairs)} < 300.\nKhông chuẩn bị training. Cần duyệt kế hoạch mở rộng dữ liệu.\n')
        return
    status('stage_6_validation', pairs=len(pairs))
    used = set()
    per_question = collections.Counter()
    max_tokens = 0
    for pair, idx in zip(pairs, indices, strict=True):
        row = str(idx['row'])
        good = lookup[row, idx['chosen_repeat']]
        bad = lookup[row, idx['rejected_repeat']]
        assert good['reward'] == 1 and good['finish_reason'] != 'length' and bad['reward'] == 0
        assert pair['messages'][-1]['content'] == good['raw_text']
        assert pair['rejected_response'] == bad['raw_text']
        for response in (good, bad):
            key = row, response['repeat']
            assert key not in used
            used.add(key)
            # Recorded vLLM prompt length includes expanded visual tokens;
            # add a small reserve for training template/EOS differences.
            max_tokens = max(max_tokens, response['prompt_tokens'] + response['output_tokens'] + 32)
        per_question[row] += 1
    assert max(per_question.values()) <= 2
    assert max_tokens <= 12288, f'Conservative branch token bound {max_tokens} exceeds 12288'
    pilot = max_tokens > 10240
    limit = 10240 if not pilot else min(12288, max_tokens + 512)
    cfg = json.loads((ROOT / 'adapter/d1_full/dpo_config.json').read_text())
    cfg.update(model='/workspace/models/D1', max_length=limit, dataset=['data/d2p_pairs_swift.jsonl'])
    cfg.pop('model_revision', None)
    save(ROOT / 'scripts/d2p_config.json', cfg)
    token_check = dict(method='vLLM prompt_tokens + output_tokens for each chosen/rejected branch + 32 token reserve; includes image tokens', conservative_max_tokens=max_tokens, max_length=limit, memory_pilot_required=pilot, training_template_verification_required=True)
    save(REPORT / 'd2p_token_check.json', token_check)
    status('stage_7_report', pairs=len(pairs))
    by_row = collections.defaultdict(list)
    for r in raw:
        by_row[str(r['row'])].append(r)
    dist = collections.Counter(sum(r['reward'] for r in rs) for rs in by_row.values())
    groups = json.loads((RUN / 'groups.json').read_text())
    correct = sum(r['reward'] for r in raw)
    lengths = sum(r['finish_reason'] == 'length' for r in raw)
    baseline = [162, 140, 159, 249, 790]
    table = '\n'.join(f'| {i}/4 | {dist[i]} | {baseline[i]} |' for i in range(5))
    artifacts = [RUN / n for n in ('raw.jsonl', 'groups.json', 'd2p_train.jsonl', 'manifest.json')]
    artifacts += [ROOT / 'data/d2p_pairs_swift.jsonl', ROOT / 'data/d2p_pairs_index.jsonl', ROOT / 'scripts/d2p_config.json']
    save(REPORT / 'd2p_artifacts_sha256.json', {str(p.relative_to(ROOT)): sha(p) for p in artifacts})
    text = f'''# D2' — Sampling & Pair Construction

- GPU: {gpu}
- D1 adapter + 2 merged shards: SHA-256 verified (`d2p_weights_verified.json`).
- Sampling: 1,500 questions × K=4; 6,000 unique responses validated against the pool.
- Correct: {correct}/6000 ({correct/60:.2f}%). Length: {lengths}/6000 ({lengths/60:.2f}%).
- Seed 3407; temperature 0.7; top_p 0.8; top_k 20; presence_penalty 1.5; max_new_tokens 8192; context 16384.

| Correct / 4 | D1 | M0 baseline (task.md) |
| --- | ---: | ---: |
{table}

## Groups and pairs

- Target: {len(groups['target'])}; retain/keep: {len(groups['keep'])}; excluded/discard: {len(groups['discard'])}.
- Training questions: {len(rows(RUN / 'd2p_train.jsonl'))}; all mixed reward questions. All-correct and all-wrong questions are excluded by the original classifier.
- DPO pairs: **{len(pairs)}**, meets ≥300 rule. At most 2/question, no reused responses, original text verified.
- Pair construction verifies image sizes and SHA-256.

## Token bound and configuration

- Conservative per-branch bound: {max_tokens} tokens, from recorded vLLM prompt + completion + 32 reserve (including image tokens).
- This corrects task.md's illustrative sum of serialized chosen messages plus rejected text, which is not a DPO branch length.
- Prepared `scripts/d2p_config.json` from the original adapter's D1 config, with D1 merged start model and `dataset` set to the pair file.
- max_length: {limit}. A100 memory pilot required: {pilot}.
- Exact ms-swift training-template encoding and the planned A100 memory checks remain prerequisites for training; no training was launched.

## Stage 8

Sampling and pair construction completed. Wait for user approval before A100 training. Pod left running. No GitHub/HF upload.
Checksums: `reports/d2p_artifacts_sha256.json`.
'''
    (REPORT / 'D2P_STAGE1_REPORT.md').write_text(text)
    status('completed_stage_8', responses=6000, pairs=len(pairs), report='reports/D2P_STAGE1_REPORT.md', training_started=False)

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('stopped_error', error=str(exc))
        traceback.print_exc()
        raise
