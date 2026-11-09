"""Validate generation and M0 input identity; never score answers or call Judge."""
from collections import Counter
import hashlib,json
from pathlib import Path
import statistics,sys
ROOT=Path('/workspace/eval_D1_20261107')
name=sys.argv[1]
expected=3 if name=='smoke' else 900
run=ROOT/'runs'/name
raw=run/'generation/raw.jsonl'
digest=hashlib.sha256(raw.read_bytes()).hexdigest()
rows=[json.loads(l) for l in raw.read_text().splitlines() if l]
assert len(rows)==len({r['id'] for r in rows})==expected
assert all(r['status']=='ok' and r['finish_reason'] in ['stop','length'] for r in rows)
reference=json.loads((ROOT/'reference/m0_inputs.json').read_text())
fields=['subject','question_type','gold','options','image_indices','prompt','num_prompt_tokens']
for row in rows:
    body=json.dumps({k:row[k] for k in fields},ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
    assert hashlib.sha256(body).hexdigest()==reference['rows'][row['id']]['identity_sha256'],row['id']
    assert row['precomputed_prompt_tokens']==row['num_prompt_tokens']
counts=Counter(r['subject'] for r in rows)
if expected==900:
    assert len(counts)==30 and set(counts.values())=={30}
    assert {r['id'] for r in rows}==set(reference['rows'])
metadata=json.loads((run/'generation/run_metadata.json').read_text())
env=json.loads((run/'generation/env.json').read_text())
assert metadata['run']['raw_file']['sha256']==digest
tokens=[r['output_tokens'] for r in rows]
assert sum(tokens)==metadata['run']['token_stats']['total']==sum(v['output_tokens_total'] for v in env['invocations'])
assert len(env['invocations'])==1
vram=[v['vram'] for v in env['invocations']]
assert all(v['nvidia_smi']['samples']>0 for v in vram)
result={'status':'PASS','rows':len(rows),'unique_ids':len(rows),'subjects':dict(counts),
 'errors':0,'m0_input_and_prompt_token_mismatches':0,'m0_sha256':reference['source_sha256'],
 'raw_sha256':digest,'raw_bytes':raw.stat().st_size,'generation_seconds':metadata['run']['duration_seconds'],
 'prompt_tokens_total':sum(r['num_prompt_tokens'] for r in rows),
 'output_tokens_total':sum(tokens),'output_tokens_mean':statistics.mean(tokens),'output_tokens_median':statistics.median(tokens),
 'length_count':sum(r['finish_reason']=='length' for r in rows),
 'max_vram_mib':max(v['nvidia_smi']['max_used_mib'] for v in vram),'invocation_vram':vram,
 'generation_config_sha256':metadata['code_and_reproduction']['config']['sha256']}
(run/'generation_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
