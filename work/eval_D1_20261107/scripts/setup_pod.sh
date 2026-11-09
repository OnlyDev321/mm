#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/workspace/hf_cache
ROOT=/workspace/eval_D1_20261107
cd "$ROOT"
python3.12 scripts/unpack_inputs.py
python3.12 source/assignment/experiments/hypB_eval/yoonseok/verify_code.py
python3.12 - <<'PY'
import importlib.metadata as m,subprocess,sys,torch
from pathlib import Path
assert m.version('torch')==torch.__version__=='2.8.0+cu128'
assert Path(subprocess.check_output(['python','-c','import sys; print(sys.executable)'],text=True).strip()).resolve()==Path(sys.executable).resolve()
PY
mkdir -p records/base_install/logs
python3.12 - <<'PY'
from pathlib import Path
import hashlib,json
root=Path('/workspace/eval_D1_20261107')
text=(root/'source/code/run_generate.sh').read_bytes()
start=text.index(b'if (( INSTALL )); then\n')
end=text.index(b'python -m pip freeze',start)
block=text[start:end]
(root/'records/install_block.sh').write_bytes(block)
(root/'records/install_block_receipt.json').write_text(json.dumps({'source':'source/code/run_generate.sh',
 'block_sha256':hashlib.sha256(block).hexdigest(),'unchanged_install_block':True},indent=2)+'\n')
PY
export SCRIPT_DIR="$ROOT/source/code" OUT="$ROOT/records/base_install" INSTALL=1
bash -euo pipefail records/install_block.sh
python3.12 - <<'PY' | tee records/torch_after_install.txt
import importlib.metadata as m,importlib.util,torch
from pathlib import Path
assert m.version('torch')==torch.__version__=='2.8.0+cu128'
assert importlib.util.find_spec('peft') is None
log=Path('records/base_install/logs/01_install.log').read_text()
assert 'Attempting uninstall: torch\n' not in log
print(m.version('torch'),torch.__version__)
PY
python3.12 -m pip check > records/pip_check_generation.txt
python3.12 -m pip freeze > records/packages_generation.txt
python3.12 -m venv --system-site-packages merge_env
merge_env/bin/python -m pip install --no-deps peft==0.17.1 > records/install_merge.log 2>&1
merge_env/bin/python -m pip check > records/pip_check_merge.txt
merge_env/bin/python -m pip freeze > records/packages_merge.txt
merge_env/bin/python - <<'PY'
import importlib.metadata as m,torch
assert m.version('peft')=='0.17.1'
assert m.version('torch')==torch.__version__=='2.8.0+cu128'
print('MERGE_ENV_PEFT',m.version('peft'),'REUSED_TORCH',torch.__version__)
PY
python3.12 source/assignment/experiments/hypB_eval/yoonseok/verify_code.py
echo SETUP_COMPLETE
