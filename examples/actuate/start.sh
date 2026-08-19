#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if ! command -v inspect-robots >/dev/null 2>&1; then
  printf '%s\n' \
    'error: inspect-robots not found; activate the rig venv first (source /path/to/rig-venv/bin/activate)' >&2
  exit 1
fi
PYTHON="$(command -v python)"

if [[ ! -f .env ]]; then
  printf '%s\n' 'warning: .env is missing from the repository root' >&2
fi

if (( $# > 0 )) && [[ $1 == "--" ]]; then
  shift
fi

serve_argv=("$PYTHON" "examples/actuate/serve.py")
printf -v serve_command '%q ' "${serve_argv[@]}"

loop_argv=("$PYTHON" "examples/actuate/run.py")
if (( $# > 0 )); then
  loop_argv+=("--" "$@")
fi
printf -v loop_command '%q ' "${loop_argv[@]}"

if tmux has-session -t =actuate-serve 2>/dev/null; then
  printf '%s\n' \
    'reusing display server session; tmux kill-session -t =actuate-serve resets it'
else
  tmux new -d -s actuate-serve "$serve_command"
  sleep 1
  if tmux has-session -t =actuate-serve 2>/dev/null; then
    :
  else
    printf '%s\n' \
      'error: display server exited; run tmux new -s actuate-serve manually and check port 8377' >&2
    exit 1
  fi
fi

printf '%s\n' 'Display: http://localhost:8377/'

if tmux has-session -t =actuate 2>/dev/null; then
  printf '%s\n' 'error: demo loop already running; tmux attach -t =actuate' >&2
  exit 1
fi

tmux new -s actuate "$loop_command"
