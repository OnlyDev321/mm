set -euo pipefail
export HF_HOME=/workspace/hf_cache PYTHONDONTWRITEBYTECODE=1
cd /workspace/eval_D1_20261107
python3.12 scripts/package_results.py
