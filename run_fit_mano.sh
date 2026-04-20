#!/usr/bin/env bash
set -euo pipefail

DEFAULT_INPUT_JSON="sample_joints21.json"
INPUT_JSON=""

print_help() {
  cat <<'EOF'
Usage: ./run_fit_mano.sh [fit_mano.py options]

Convenience wrapper for fit_mano.py.
If --input-json is not provided, it auto-fills:
  --input-json sample_joints21.json

If sample_joints21.json does not exist, this script creates a starter
21-keypoint JSON file.

Examples:
  ./run_fit_mano.sh
  ./run_fit_mano.sh --output-joint-format same --iterations 300
  ./run_fit_mano.sh --input-json my_hand_points.json --hand-side left
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_help
  exit 0
fi

args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  arg="${args[$i]}"
  case "$arg" in
    --input-json)
      if (( i + 1 < ${#args[@]} )); then
        INPUT_JSON="${args[$((i + 1))]}"
      fi
      ;;
    --input-json=*)
      INPUT_JSON="${arg#--input-json=}"
      ;;
  esac
done

if [[ -z "$INPUT_JSON" ]]; then
  INPUT_JSON="$DEFAULT_INPUT_JSON"
  args=("--input-json" "$INPUT_JSON" "${args[@]}")
fi

if [[ ! -f "$INPUT_JSON" ]]; then
  cat > "$INPUT_JSON" <<'EOF'
{
  "joints": [
    [0.0000, 0.0000, 0.0000],
    [0.0150, 0.0120, 0.0000],
    [0.0300, 0.0240, 0.0000],
    [0.0450, 0.0360, 0.0000],
    [0.0600, 0.0480, 0.0000],
    [0.0100, 0.0160, 0.0020],
    [0.0200, 0.0320, 0.0030],
    [0.0300, 0.0480, 0.0040],
    [0.0400, 0.0640, 0.0050],
    [0.0000, 0.0180, 0.0030],
    [0.0000, 0.0360, 0.0040],
    [0.0000, 0.0540, 0.0050],
    [0.0000, 0.0720, 0.0060],
    [-0.0100, 0.0160, 0.0020],
    [-0.0200, 0.0320, 0.0030],
    [-0.0300, 0.0480, 0.0040],
    [-0.0400, 0.0640, 0.0050],
    [-0.0180, 0.0100, 0.0010],
    [-0.0300, 0.0200, 0.0020],
    [-0.0420, 0.0300, 0.0030],
    [-0.0540, 0.0400, 0.0040]
  ]
}
EOF
  echo "Created starter keypoint file at $INPUT_JSON"
fi

python fit_mano.py "${args[@]}"
