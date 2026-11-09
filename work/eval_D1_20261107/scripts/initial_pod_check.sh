set -euo pipefail
export HF_HOME=/workspace/hf_cache PYTHONDONTWRITEBYTECODE=1
cd /workspace
test ! -e /workspace/eval_D1_20261107
mkdir -p /workspace/eval_D1_20261107/records /workspace/eval_D1_20261107/input /workspace/eval_D1_20261107/scripts
python3.12 - <<'PY' | tee /workspace/eval_D1_20261107/records/torch_before_install.txt
import importlib.metadata as m,torch,sys
print(m.version('torch'),torch.__version__)
assert m.version('torch')==torch.__version__=='2.8.0+cu128'
print('PYTHON',sys.version.split()[0])
assert torch.cuda.device_count()==1 and 'RTX 4090' in torch.cuda.get_device_name(0)
print('GPU',torch.cuda.get_device_name(0))
PY
nvidia-smi > /workspace/eval_D1_20261107/records/nvidia-smi.txt
echo INITIAL_GPU_AND_TORCH_PASS
