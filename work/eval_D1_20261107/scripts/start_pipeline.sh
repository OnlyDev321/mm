set -euo pipefail
export HF_HOME=/workspace/hf_cache
cd /workspace/eval_D1_20261107
test ! -e records/pipeline.pid
test ! -e records/pipeline.exit
nohup bash scripts/run_generation.sh > records/pipeline_launcher.log 2>&1 < /dev/null &
printf '%s\n' "$!" > records/pipeline.pid
echo D1_GPU_GENERATION_PIPELINE_STARTED
