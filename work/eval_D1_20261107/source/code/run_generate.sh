#!/usr/bin/env bash
set -euo pipefail

# One-shot inference runner. Put --out on persistent pod storage.
# Automatic stop syntax is intentionally unverified: check `runpodctl --help`
# on RunPod before relying on `runpodctl stop pod "$RUNPOD_POD_ID"`.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH=""; REVISION=""; DATA_ROOT=""; OUT=""; CONFIG="$SCRIPT_DIR/config.yaml"; LIMIT=""; SUBJECTS=""
INSTALL=0; STOP_POD=0; SKIP_GATE=0; PASSTHROUGH=(); STAGE="argument parsing"
STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
usage() {
  cat <<EOF
Usage: $0 --out DIR [--model_path PATH] [--revision REVISION] [--data_root DIR]
       [--config PATH] [--install] [--stop_pod_when_done] [--skip_env_gate]
       [--limit N] [--subjects A,B] [--engine_override KEY=JSON ...]
Use a persistent pod volume for --out so raw.jsonl and logs survive restarts.
EOF
}
while (($#)); do
  case "$1" in
    --model_path|--revision|--data_root|--config|--limit|--subjects)
      [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }
      key="$1"; value="$2"; shift 2
      case "$key" in
        --model_path) MODEL_PATH="$value";; --revision) REVISION="$value";;
        --data_root) DATA_ROOT="$value";; --config) CONFIG="$value";;
        --limit) LIMIT="$value"; PASSTHROUGH+=("$key" "$value");;
        --subjects) SUBJECTS="$value"; PASSTHROUGH+=("$key" "$value");;
        *) PASSTHROUGH+=("$key" "$value");;
      esac;;
    --out) [[ $# -ge 2 ]] || { echo "missing value for --out" >&2; exit 2; }; OUT="$2"; shift 2;;
    --install) INSTALL=1; shift;;
    --stop_pod_when_done) STOP_POD=1; shift;;
    --skip_env_gate) SKIP_GATE=1; shift;;
    --engine_override) [[ $# -ge 2 ]] || { echo "missing value for --engine_override" >&2; exit 2; }; PASSTHROUGH+=("$1" "$2"); shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done
[[ -n "$OUT" ]] || { echo "--out is required" >&2; usage >&2; exit 2; }
mkdir -p "$OUT/logs"
if [[ -f "$OUT/logs/FAILED" ]]; then
  FAILED_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  mv "$OUT/logs/FAILED" "$OUT/logs/FAILED.$FAILED_STAMP"
fi
FAILED=0; FAIL_CODE=0
cleanup() {
  local code=$?
  if (( code != 0 || FAILED )); then
    FAILED=1; FAIL_CODE=$(( code != 0 ? code : FAIL_CODE ))
    { echo "stage=$STAGE"; echo "exit_code=$FAIL_CODE"; echo "time=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; nvidia-smi 2>&1 || true; df -h 2>&1 || true; } >"$OUT/logs/FAILED"
  fi
  if (( STOP_POD )); then
    sync || true
    if command -v runpodctl >/dev/null 2>&1 && [[ -n "${RUNPOD_POD_ID:-}" ]]; then
      echo "stop command: runpodctl stop pod $RUNPOD_POD_ID" | tee -a "$OUT/logs/stop_pod.log"
      runpodctl stop pod "$RUNPOD_POD_ID" || true
    else
      echo "automatic pod stop unavailable: runpodctl or RUNPOD_POD_ID is missing" | tee -a "$OUT/logs/stop_pod.log"
    fi
  fi
}
trap cleanup EXIT
STAGE="system snapshot"
{ echo "started_utc=$STARTED"; nvidia-smi 2>&1 || true; df -h; free -g; python --version 2>&1; } >"$OUT/logs/00_system.log"
if (( INSTALL )); then
  STAGE="install"
  if [[ ! -s "$SCRIPT_DIR/requirements.txt" ]]; then echo "requirements.txt is empty" | tee "$OUT/logs/01_install.log"; exit 3; fi
  set +e
  python -m pip install -r "$SCRIPT_DIR/requirements.txt" 2>&1 | tee "$OUT/logs/01_install.log"
  INSTALL_CODE=${PIPESTATUS[0]}
  set -e
  if (( INSTALL_CODE != 0 )); then
    if grep -qi "externally-managed-environment" "$OUT/logs/01_install.log"; then
      echo "externally-managed-environment detected; retrying with --break-system-packages" | tee -a "$OUT/logs/01_install.log"
      python -m pip install --break-system-packages -r "$SCRIPT_DIR/requirements.txt" 2>&1 | tee -a "$OUT/logs/01_install.log"
    else
      exit "$INSTALL_CODE"
    fi
  fi
fi
python -m pip freeze > "$OUT/logs/pip_freeze.txt"
STAGE="environment export"
export CONFIG_PATH="$CONFIG"
while IFS='=' read -r key value; do
  [[ -n "$key" ]] && export "$key=$value"
done < <(python -c 'import os,yaml; c=yaml.safe_load(open(os.environ["CONFIG_PATH"])); [print(k+"="+str(v)) for k,v in c["environment"]["env_vars"].items()]')
if [[ -z "${HF_HOME:-}" ]]; then
  if (( SKIP_GATE )); then echo "WARNING: HF_HOME is unset; using process defaults" | tee -a "$OUT/logs/00_system.log"; else echo "HF_HOME is unset; export HF_HOME=... before running" | tee -a "$OUT/logs/00_system.log"; exit 4; fi
else
  echo "HF_HOME=$HF_HOME" | tee -a "$OUT/logs/00_system.log"
  mkdir -p "$HF_HOME"
fi
if [[ -n "$DATA_ROOT" ]]; then mkdir -p "$DATA_ROOT"; fi
if python -c 'import hf_transfer' >/dev/null 2>&1; then export HF_HUB_ENABLE_HF_TRANSFER=1; echo "HF transfer: enabled" | tee -a "$OUT/logs/00_system.log"; else echo "HF transfer: unavailable" | tee -a "$OUT/logs/00_system.log"; fi
MIN_FREE_GB=$(python -c 'import os,yaml; print(yaml.safe_load(open(os.environ["CONFIG_PATH"]))["run"]["min_free_disk_gb"])')
for CHECK_PATH in "${HF_HOME:-.}" "$OUT"; do
  DISK_PATH="$CHECK_PATH"
  while [[ ! -e "$DISK_PATH" ]]; do
    NEXT_PATH="$(dirname "$DISK_PATH")"
    if [[ "$NEXT_PATH" == "$DISK_PATH" ]]; then echo "cannot determine free disk space: $CHECK_PATH"; exit 6; fi
    DISK_PATH="$NEXT_PATH"
  done
  set +e
  FREE_KB=$(df -Pk "$DISK_PATH" 2>/dev/null | awk 'NR==2 {print $4}')
  DF_CODE=${PIPESTATUS[0]}
  set -e
  if (( DF_CODE != 0 )) || [[ ! "$FREE_KB" =~ ^[0-9]+$ ]]; then echo "cannot determine free disk space: $CHECK_PATH"; exit 6; fi
  FREE_GB=$((FREE_KB / 1024 / 1024))
  echo "free_disk path=$CHECK_PATH filesystem_path=$DISK_PATH gb=$FREE_GB required=$MIN_FREE_GB" | tee -a "$OUT/logs/00_system.log"
  if (( FREE_GB < MIN_FREE_GB )); then echo "insufficient free disk space"; exit 5; fi
done
if (( ! SKIP_GATE )); then
  STAGE="environment check"
  CHECK_ARGS=(--config "$CONFIG" --out "$OUT" --sections system,packages,vllm_import,gpu,qwen_vl_utils,model --download)
  [[ -n "$MODEL_PATH" ]] && CHECK_ARGS+=(--model_path "$MODEL_PATH"); [[ -n "$REVISION" ]] && CHECK_ARGS+=(--revision "$REVISION"); [[ -n "$DATA_ROOT" ]] && CHECK_ARGS+=(--data_root "$DATA_ROOT")
  set +e; python "$SCRIPT_DIR/check_env.py" "${CHECK_ARGS[@]}" 2>&1 | tee "$OUT/logs/02_env_check.log"; CHECK_CODE=${PIPESTATUS[0]}; set -e
  python "$SCRIPT_DIR/gate_env_check.py" "$OUT/env_check.json" | tee -a "$OUT/logs/02_env_check.log"
else
  echo "WARNING: --skip_env_gate is for local rehearsal only; environment was not gated" | tee "$OUT/logs/02_env_check.log"
fi
STAGE="generation"
GEN_ARGS=(--config "$CONFIG" --out "$OUT")
[[ -n "$MODEL_PATH" ]] && GEN_ARGS+=(--model_path "$MODEL_PATH"); [[ -n "$REVISION" ]] && GEN_ARGS+=(--revision "$REVISION"); [[ -n "$DATA_ROOT" ]] && GEN_ARGS+=(--data_root "$DATA_ROOT")
[[ -f "$OUT/raw.jsonl" ]] && echo "Resume: existing raw.jsonl found"
set +e; python "$SCRIPT_DIR/generate.py" "${GEN_ARGS[@]}" "${PASSTHROUGH[@]}" 2>&1 | tee "$OUT/logs/03_generate.log"; GEN_CODE=${PIPESTATUS[0]}; set -e
if (( GEN_CODE != 0 )) && [[ ! -f "$OUT/raw.jsonl" ]]; then
  FAILED=1; FAIL_CODE=$GEN_CODE; exit "$GEN_CODE"
fi
STAGE="raw validation"
VALIDATE_ARGS=(--raw "$OUT/raw.jsonl" --config "$CONFIG")
if [[ -n "${EXPECT_ROWS:-}" ]]; then
  VALIDATE_ARGS+=(--expect "$EXPECT_ROWS")
elif [[ -n "$LIMIT" ]]; then
  VALIDATE_ARGS+=(--expect "$LIMIT")
elif [[ -n "$SUBJECTS" ]]; then
  SUBJECT_COUNT=$(awk -F, '{print NF}' <<<"$SUBJECTS")
  ROWS_PER_SUBJECT=$(python -c 'import os,yaml; print(yaml.safe_load(open(os.environ["CONFIG_PATH"]))["dataset"]["expected_rows_per_subject"])')
  VALIDATE_ARGS+=(--expect "$((SUBJECT_COUNT * ROWS_PER_SUBJECT))")
fi
set +e; python "$SCRIPT_DIR/validate_raw.py" "${VALIDATE_ARGS[@]}" 2>&1 | tee "$OUT/logs/04_validate_raw.log"; VALIDATE_CODE=${PIPESTATUS[0]}; set -e
{ sha256sum "$OUT/raw.jsonl"; stat -c 'bytes=%s' "$OUT/raw.jsonl" 2>/dev/null || stat -f 'bytes=%z' "$OUT/raw.jsonl" 2>/dev/null || true; echo "started_utc=$STARTED"; echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; } | tee -a "$OUT/logs/04_validate_raw.log"
if [[ -f "$OUT/run_metadata.json" ]]; then
  { sha256sum "$OUT/run_metadata.json"; } | tee -a "$OUT/logs/04_validate_raw.log"
fi
if (( GEN_CODE != 0 || VALIDATE_CODE != 0 )); then
  FAILED=1
  FAIL_CODE=$(( GEN_CODE != 0 ? GEN_CODE : VALIDATE_CODE ))
  (( GEN_CODE != 0 )) && STAGE="generation"
  exit "$FAIL_CODE"
fi
STAGE="complete"; echo "SUCCESS: output=$OUT raw=$OUT/raw.jsonl metadata=$OUT/run_metadata.json logs=$OUT/logs"
