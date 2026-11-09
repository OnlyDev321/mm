"""Unpack only regular task files and verify all copied sources and adapter bytes."""
import hashlib,json
from pathlib import Path
import tarfile
ROOT=Path('/workspace/eval_D1_20261107')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def unpack(archive):
    with tarfile.open(archive,'r:gz') as source:
        for item in source.getmembers():
            target=(ROOT/item.name).resolve()
            assert item.isfile() and target.is_relative_to(ROOT) and not target.exists(),item.name
            target.parent.mkdir(parents=True,exist_ok=True)
            with source.extractfile(item) as stream,target.open('wb') as out:
                import shutil
                shutil.copyfileobj(stream,out)
receipt=json.loads((ROOT/'reports/d1_source_receipt.json').read_text())
assert sha(ROOT/'input/source_bundle.tar.gz')==receipt['source_archive_sha256']
unpack(ROOT/'input/source_bundle.tar.gz')
for name,wanted in receipt['files_sha256'].items():assert sha(ROOT/name)==wanted,name
assert sha(ROOT/'input/D1_adapter_full.tar.gz')==receipt['adapter_archive_sha256']
unpack(ROOT/'input/D1_adapter_full.tar.gz')
manifest=json.loads((ROOT/'reference/d1_adapter_sha256.json').read_text())
for name,wanted in manifest.items():assert sha(ROOT/name)==wanted,name
assert sha(ROOT/'adapter/full/adapter_model.safetensors')==receipt['adapter_sha256']
result={'status':'PASS','source_files':len(receipt['files_sha256']),
        'adapter_files':len(manifest),'adapter_sha256':receipt['adapter_sha256']}
(ROOT/'records/input_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
