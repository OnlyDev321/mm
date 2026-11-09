#!/usr/bin/env bash
# One-command MMMU-val pipeline: generation (run_generate.sh) -> raw validation -> scoring (prepare/run/summarize).
# Paths come only from arguments. Scoring 'run' calls the Judge API and needs OPENAI_API_KEY.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=""; CONFIG="$SCRIPT_DIR/config.yaml"; SKIP_GENERATE=0; GEN_ARGS=()
usage() {
  cat <<EOF
Usage: $0 --out DIR [--data_root DIR] [--model_path PATH] [--revision SHA] [--config PATH]
          [--install] [--skip_generate] [other run_generate.sh options]
  --skip_generate  score an existing DIR/raw.jsonl without regenerating
Outputs: DIR/raw.jsonl, DIR/run_metadata.json, DIR/logs/, DIR/scoring/
EOF
}
while (($#)); do
  case "$1" in
    --out) OUT="$2"; shift 2;;
    --config) CONFIG="$2"; GEN_ARGS+=("$1" "$2"); shift 2;;
    --skip_generate) SKIP_GENERATE=1; shift;;
    -h|--help) usage; exit 0;;
    --install|--stop_pod_when_done|--skip_env_gate) GEN_ARGS+=("$1"); shift;;
    *) [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }; GEN_ARGS+=("$1" "$2"); shift 2;;
  esac
done
[[ -n "$OUT" ]] || { usage >&2; exit 2; }
if (( ! SKIP_GENERATE )); then
  bash "$SCRIPT_DIR/run_generate.sh" --out "$OUT" "${GEN_ARGS[@]}"
fi
[[ -f "$OUT/raw.jsonl" ]] || { echo "missing $OUT/raw.jsonl" >&2; exit 3; }
SCORING="$OUT/scoring"
if [[ ! -f "$SCORING/manifest.json" ]]; then
  python "$SCRIPT_DIR/scoring/evaluate.py" prepare --raw "$OUT/raw.jsonl" --config "$CONFIG" --out "$SCORING"
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set. Prepared requests are in $SCORING." >&2
  echo "Set the key and rerun the same command with --skip_generate to finish scoring." >&2
  exit 4
fi
python "$SCRIPT_DIR/scoring/evaluate.py" run --out "$SCORING"
python "$SCRIPT_DIR/scoring/evaluate.py" summarize --out "$SCORING"
