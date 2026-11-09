#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/workspace/hf_cache
export XDG_CACHE_HOME=/workspace/cache PIP_CACHE_DIR=/workspace/pip_cache TMPDIR=/workspace/tmp
export TRITON_CACHE_DIR=/workspace/triton_cache CUDA_CACHE_PATH=/workspace/cuda_cache
export TORCHINDUCTOR_CACHE_DIR=/workspace/torchinductor_cache VLLM_CACHE_ROOT=/workspace/vllm_cache
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
ROOT=/workspace/eval_D1_20261107
CODE="$ROOT/source/code"
cd "$ROOT"
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR" records runs
finish() { result=$?; printf '%s\n' "$result" > records/pipeline.exit; date -u -Iseconds > records/pipeline_ended_utc.txt; }
trap finish EXIT
stage() { printf '%s\n' "$1" > records/pipeline.stage; }
date -u -Iseconds > records/pipeline_started_utc.txt
stage setup
bash scripts/setup_pod.sh > records/setup.log 2>&1
printf '0\n' > records/setup.exit
while IFS='=' read -r key value; do [[ -n "$key" ]] && export "$key=$value"; done < <(python3.12 -c 'import yaml; c=yaml.safe_load(open("source/code/config.yaml")); [print(k+"="+str(v)) for k,v in c["environment"]["env_vars"].items()]')
stage download_and_merge
merge_env/bin/python -u scripts/prepare_merged.py > records/merge.log 2>&1
printf '0\n' > records/merge.exit
stage environment_gate
python3.12 "$CODE/check_env.py" --config "$CODE/config.yaml" --out records/gates --model_path "$ROOT/models/D1" --data_root /workspace/mmmu_cache --download > records/env_gate.log 2>&1
python3.12 "$CODE/gate_env_check.py" records/gates/env_check.json > records/gate_result.txt
stage preflight
python3.12 "$CODE/preflight.py" --config "$CODE/config.yaml" --model_path "$ROOT/models/D1" --data_root /workspace/mmmu_cache --out records/preflight > records/preflight.log 2>&1
python3.12 - <<'PY'
import json
s=json.load(open('records/preflight/preflight.json'))['summary']
assert s['processed']==s['selected']==900 and not s['partial']
assert not s['validation_failures'] and s['exceeds_count']==0
PY
for RUN in smoke D1; do
    stage "generation_$RUN"
    DIR="$ROOT/runs/$RUN"
    test ! -e "$DIR"
    mkdir "$DIR"
    EXTRA=()
    if [[ "$RUN" == smoke ]]; then EXTRA=(--limit 3); EXPECTED=3; else EXPECTED=900; fi
    date -u -Iseconds > "$DIR/started_utc.txt"
    bash "$CODE/run_generate.sh" --install --model_path "$ROOT/models/D1" --data_root /workspace/mmmu_cache --out "$DIR/generation" "${EXTRA[@]}" > "$DIR/generate.log" 2>&1
    date -u -Iseconds > "$DIR/finished_utc.txt"
    stage "validation_$RUN"
    python3.12 "$CODE/validate_raw.py" --raw "$DIR/generation/raw.jsonl" --config "$CODE/config.yaml" --expect "$EXPECTED" > "$DIR/raw_validation.log" 2>&1
    python3.12 scripts/validate_generation.py "$RUN" > "$DIR/summary.log" 2>&1
    python3.12 - <<'PY' > "$DIR/torch_after_generation.txt"
import importlib.metadata as m,importlib.util,torch
assert m.version('torch')==torch.__version__=='2.8.0+cu128'
assert importlib.util.find_spec('peft') is None
print(m.version('torch'),torch.__version__)
PY
    python3.12 -m pip check > "$DIR/pip_check.txt"
    python3.12 source/assignment/experiments/hypB_eval/yoonseok/verify_code.py > "$DIR/source_verification.txt"
    printf '0\n' > "$DIR/run.exit"
done
stage complete
echo GPU_GENERATION_COMPLETE_NO_SCORING
