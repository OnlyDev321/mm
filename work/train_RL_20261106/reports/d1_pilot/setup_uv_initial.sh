#!/usr/bin/env bash
set -euo pipefail
cd /workspace/train_D1
export HF_HOME=/workspace/hf_cache TMPDIR=$PWD/tmp XDG_CACHE_HOME=$PWD/cache UV_CACHE_DIR=/workspace/cache/uv PYTHONDONTWRITEBYTECODE=1
trap 'echo $? > reports/setup.exit' EXIT
printf '%s\n' 'b23792472c15e0882de633a47021460b2b0bfdc8103aa2e030bcd4b009643e2c  scripts/requirements-train-lock.txt' '3840bec5681d63fda05a6b8c687540d8a370ce2c4d3b42246f5241066a0e45f3  scripts/requirements-eval-lock.txt' | sha256sum --strict -c -
nvidia-smi > reports/nvidia_smi.txt
uv --version
for role in train eval; do
    venv=".venv-$role"
    uv venv --system-site-packages --python /usr/bin/python3.12 "$venv"
    "$venv/bin/python" - "$role" <<'PY'
from importlib import metadata as m
from pathlib import Path
from packaging.specifiers import SpecifierSet
import sys
role=sys.argv[1]
missing=[]
for line in Path(f'scripts/requirements-{role}-lock.txt').read_text().splitlines():
    if not line or line.startswith('#'): continue
    name,version=line.split('==')
    try: installed=m.version(name)
    except m.PackageNotFoundError: installed=None
    if installed is None or not SpecifierSet(f'=={version}').contains(installed):
        assert name.lower().replace('_','-') not in {'torch','torchvision','torchaudio','triton'} and not name.startswith('nvidia-'), line
        missing.append(line)
Path(f'reports/{role}_missing_only.txt').write_text('\n'.join(missing)+'\n')
print(role, 'pinned packages to install', len(missing), flush=True)
PY
    uv pip install --python "$venv/bin/python" --no-deps -r "reports/${role}_missing_only.txt"
    uv pip check --python "$venv/bin/python" > "reports/${role}_pip_check.txt" 2>&1
    uv pip freeze --python "$venv/bin/python" > "reports/packages_${role}.txt"
    "$venv/bin/python" - "$role" <<'PY'
from importlib import metadata as m
from packaging.specifiers import SpecifierSet
from pathlib import Path
import sys, torch
role=sys.argv[1]
assert m.version('torch')==torch.__version__=='2.8.0+cu128'
assert str(Path(torch.__file__).resolve()).startswith('/usr/local/lib/python3.12/dist-packages/torch/')
assert torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
for line in Path(f'scripts/requirements-{role}-lock.txt').read_text().splitlines():
    if not line or line.startswith('#'): continue
    name,version=line.split('==')
    assert SpecifierSet(f'=={version}').contains(m.version(name)), line
print(role, torch.__version__, torch.__file__, torch.cuda.get_device_name(0), flush=True)
PY
done
echo SETUP_COMPLETE
