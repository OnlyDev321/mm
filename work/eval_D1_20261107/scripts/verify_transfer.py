"""Verify the complete remote archive and unchanged frozen sources locally."""
import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
ready=json.loads((ROOT/'reports/transfer/transfer_ready.json').read_text())
archive=ROOT/'reports/transfer/generation_results.tar.gz'
assert archive.stat().st_size==ready['bytes'] and sha(archive)==ready['sha256']
assert not (ROOT/'records').exists() and not (ROOT/'runs').exists()
with tarfile.open(archive,'r:gz') as source:
    members=source.getmembers()
    names=[m.name for m in members]
    assert len(names)==len(set(names))
    manifest=json.load(source.extractfile('records/result_files_sha256.json'))
    assert set(names)==set(manifest)|{'records/result_files_sha256.json'}
    for member in members:
        path=(ROOT/member.name).resolve()
        assert member.isfile() and Path(member.name).parts[0] in {'records','runs'}
        assert path.is_relative_to(ROOT) and not path.exists()
        data=source.extractfile(member).read()
        if member.name in manifest:assert hashlib.sha256(data).hexdigest()==manifest[member.name],member.name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
for name,wanted in manifest.items():assert sha(ROOT/name)==wanted,name
for name,wanted in json.loads((ROOT/'records/evaluation_source_sha256.json').read_text()).items():
    assert sha(ROOT/name)==wanted,name
assert (ROOT/'records/copied_source_receipt.json').read_bytes()==(ROOT/'reports/d1_source_receipt.json').read_bytes()
raw=ROOT/'runs/D1/generation/raw.jsonl'
assert sha(raw)==ready['raw_sha256']
result={'status':'PASS','remote_local_sha256_equal':True,'files_verified':len(manifest),
        'raw_sha256':sha(raw),'archive_sha256':sha(archive),'raw_path':str(raw)}
(ROOT/'reports/TRANSFER_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
