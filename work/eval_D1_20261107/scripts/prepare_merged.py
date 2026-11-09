"""Merge the verified local D1 adapter with the pinned base using frozen HypB code."""
import hashlib,json,subprocess,sys
from pathlib import Path
from huggingface_hub import snapshot_download
from safetensors import safe_open
ROOT=Path('/workspace/eval_D1_20261107')
BASE='Qwen/Qwen3-VL-4B-Instruct'
REV='ebb281ec70b05090aa6165b016eac8ec08e71b17'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
base=Path(snapshot_download(BASE,revision=REV,max_workers=2))
assert base.name==REV
adapter=ROOT/'adapter/full'
assert sha(adapter/'adapter_model.safetensors')=='2d4bcce4972b36b6ef49b0b73d46c9d72a3803f43632b6847ae6f4d65e0132e8'
out=ROOT/'models/D1'
assert not out.exists()
merge=ROOT/'source/assignment/experiments/hypB_train/yoonseok/merge.py'
subprocess.run([sys.executable,str(merge),str(adapter),str(out)],check=True)
hashes={}
dtypes={}
for path in sorted(out.glob('*.safetensors')):
    hashes[path.name]=sha(path)
    with safe_open(path,framework='pt',device='cpu') as handle:
        for key in handle.keys():
            dtype=handle.get_slice(key).get_dtype()
            assert dtype=='BF16',(key,dtype)
            dtypes[dtype]=dtypes.get(dtype,0)+1
assert hashes
previous=json.loads((ROOT/'reference/d1_a100_merged_sha256.json').read_text())
processor={}
for path in base.iterdir():
    if path.suffix in {'.json','.jinja','.txt'} and path.name not in {'config.json','model.safetensors.index.json'}:
        assert sha(path)==sha(out/path.name),path.name
        processor[path.name]=sha(path)
receipt={'status':'PASS','model':BASE,'revision':REV,'adapter_sha256':sha(adapter/'adapter_model.safetensors'),
 'safetensors_sha256':hashes,'dtypes':dtypes,'processor_original_hashes':processor,
 'a100_safetensors_sha256':previous,'a100_hashes_equal':hashes==previous,
 'weight_bytes':sum(p.stat().st_size for p in out.glob('*.safetensors'))}
(ROOT/'records/model_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
assert hashes==previous, 'D1 merged weight hashes differ from the A100 run'
