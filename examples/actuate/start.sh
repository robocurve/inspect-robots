#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -n ${TMUX:-} ]]; then
  printf '%s\n' \
    'error: already inside tmux; run start.sh from a plain shell, or use the manual recipe in the README' >&2
  exit 1
fi

if ! command -v inspect-robots >/dev/null 2>&1; then
  printf '%s\n' \
    'error: inspect-robots not found; activate the rig venv first (source /path/to/rig-venv/bin/activate)' >&2
  exit 1
fi
PYTHON="$(command -v python || true)"
if [[ -z $PYTHON ]]; then
  PYTHON="$(command -v python3 || true)"
fi
if [[ -z $PYTHON ]]; then
  printf '%s\n' 'error: no python on PATH; activate the rig venv first' >&2
  exit 1
fi
if ! "$PYTHON" -c 'import inspect_robots' >/dev/null 2>&1; then
  printf '%s\n' \
    "error: $PYTHON cannot import inspect_robots; activate the venv that owns inspect-robots" >&2
  exit 1
fi

if [[ ! -f .env ]]; then
  printf '%s\n' 'warning: .env is missing from the repository root' >&2
fi

if (( $# > 0 )) && [[ $1 == "--" ]]; then
  shift
fi

# A pre-existing tmux server hands new sessions the server's stale
# environment, so pin PATH explicitly and keep the crashed loop's pane open
# long enough to read the error.
serve_argv=(env "PATH=$PATH" "$PYTHON" "examples/actuate/serve.py")
printf -v serve_command '%q ' "${serve_argv[@]}"

loop_argv=(env "PATH=$PATH" "$PYTHON" "examples/actuate/run.py")
if (( $# > 0 )); then
  loop_argv+=("--" "$@")
fi
hold_wrapper='"$@"; status=$?; printf "\nrun.py exited %s; press Enter to close\n" "$status"; read -r _'
printf -v loop_command '%q ' bash -c "$hold_wrapper" _ "${loop_argv[@]}"

if tmux has-session -t =actuate-serve 2>/dev/null; then
  printf '%s\n' \
    'reusing display server session; tmux kill-session -t =actuate-serve resets it'
else
  tmux new -d -s actuate-serve "$serve_command"
  sleep 1
  if ! tmux has-session -t =actuate-serve 2>/dev/null; then
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
