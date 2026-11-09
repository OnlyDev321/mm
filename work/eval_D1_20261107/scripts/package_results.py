"""Preserve completed raw/records only; exclude models, caches and credentials."""
import hashlib,json,re,shutil,tarfile
from pathlib import Path
ROOT=Path('/workspace/eval_D1_20261107')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert (ROOT/'records/pipeline.exit').read_text().strip()=='0'
assert (ROOT/'records/pipeline.stage').read_text().strip()=='complete'
for name in ['smoke','D1']:
    assert json.loads((ROOT/'runs'/name/'generation_summary.json').read_text())['status']=='PASS'
receipt=json.loads((ROOT/'reports/d1_source_receipt.json').read_text())
for name,wanted in receipt['files_sha256'].items():assert sha(ROOT/name)==wanted,name
shutil.copyfile(ROOT/'reports/d1_source_receipt.json',ROOT/'records/copied_source_receipt.json')
actual={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in (ROOT/'source').rglob('*') if p.is_file()}
(ROOT/'records/evaluation_source_sha256.json').write_text(json.dumps(actual,indent=2)+'\n')
scripts={p.name:sha(p) for p in (ROOT/'scripts').iterdir() if p.is_file()}
(ROOT/'records/executed_scripts_sha256.json').write_text(json.dumps(scripts,indent=2)+'\n')
paths=[p for folder in ['records','runs'] for p in sorted((ROOT/folder).rglob('*')) if p.is_file()]
credential=re.compile(rb'(?:hf_[A-Za-z0-9]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{24,})')
for p in paths:
    assert credential.search(p.read_bytes()) is None,'Possible credential in '+str(p.relative_to(ROOT))
manifest={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
manifest_path=ROOT/'records/result_files_sha256.json'
assert not manifest_path.exists()
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
archive=ROOT/'generation_results.tar.gz'
assert not archive.exists()
with tarfile.open(archive,'w:gz',compresslevel=1) as out:
    for p in paths+[manifest_path]:out.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
summary=json.loads((ROOT/'runs/D1/generation_summary.json').read_text())
result={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),
        'files_verified':len(manifest),'raw_sha256':summary['raw_sha256']}
(ROOT/'reports/transfer_ready.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
