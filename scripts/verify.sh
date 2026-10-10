#!/usr/bin/env bash
# MVP acceptance gates. Exits nonzero if any gate fails or cannot run.
#
# Needs a read-only Hermes host checkout (HERMES_HOST_SOURCE), uv, git, and network access the
# first time (uv venvs, PyPI wheels, the pinned BGE-M3 INT8 and MiniLM Hugging Face models). Optional overrides,
# also read from scripts/verify.local.env when present:
#   HERMES_HOST_SOURCE      Hermes source tree (required)
#   HERMES_MEM_WORK         scratch dir for venvs, homes and logs (default: $TMPDIR/hermes-mem-verify)
#   HERMES_MEM_CACHE_HOME   cache root; models live in <root>/hermes-mem/models, the plugin's shared default
#                           (default: $XDG_CACHE_HOME or ~/.cache)
#   UNIT_PYTHON / HOST_PYTHON  interpreters for uv venvs (default: 3.14 / 3.13; host needs 3.11-3.13)
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO/scripts/verify.local.env" ] && . "$REPO/scripts/verify.local.env"
: "${HERMES_HOST_SOURCE:?set HERMES_HOST_SOURCE to a Hermes host source tree}"
WORK="${HERMES_MEM_WORK:-${TMPDIR:-/tmp}/hermes-mem-verify}"
CACHE_HOME="${HERMES_MEM_CACHE_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}}"
MODEL_CACHE="$CACHE_HOME/hermes-mem/models"
UNIT_PYTHON="${UNIT_PYTHON:-3.14}"
HOST_PYTHON="${HOST_PYTHON:-3.13}"
LOGS="$WORK/logs"
mkdir -p "$LOGS" "$MODEL_CACHE"
export PYTHONPYCACHEPREFIX="$WORK/pycache" PYTHONDONTWRITEBYTECODE=1

venv() {  # venv NAME PYTHON PACKAGES...
  local dir="$WORK/$1" py="$2"; shift 2
  if [ ! -x "$dir/bin/python" ] || [ ! -f "$dir/.ready" ]; then
    uv venv -q --python "$py" "$dir" && uv pip install -q --python "$dir/bin/python" "$@" && touch "$dir/.ready" || return 1
  fi
}

declare -a NAMES CODES
gate() {  # gate NAME CMD...
  local name="$1"; shift
  echo "== $name"
  "$@" >"$LOGS/$name.log" 2>&1
  local code=$?
  tail -n 15 "$LOGS/$name.log"
  echo "== $name exit=$code"
  NAMES+=("$name"); CODES+=("$code")
}

gate bootstrap-unit-venv venv venv-unit "$UNIT_PYTHON" pyyaml pytest pyflakes
# The embedding engine never runs in a Hermes/unit interpreter: provision its locked runtime once
# through the real `setup` command and point every gate's config at it.
ENGINE_HOME="$WORK/engine-home"
gate engine-setup "$WORK/venv-unit/bin/python" -m hermes_mem --home "$ENGINE_HOME" setup
ENGINE_PY="$(ls -d "$ENGINE_HOME"/hermes-mem/runtime/engine-*/bin/python 2>/dev/null | head -n1)"

# The default profile (bge-m3-int8) is what every embedding gate runs; only the cache is pinned here.
cat >"$WORK/embedding.yaml" <<EOF
embedding:
  cache_dir: $MODEL_CACHE
  threads: 2
  batch_size: 8
engine:
  command: ["$ENGINE_PY", "-I", "$REPO/engine/hermes_mem_engine.py"]
  timeout: 1800  # a model's first use downloads it inside one request
sync:
  max_sessions: 200
  background: true
EOF

host_venv() {
  uv pip compile -q --python-version "$HOST_PYTHON" "$HERMES_HOST_SOURCE/pyproject.toml" -o "$WORK/host-reqs.txt" &&
    venv venv-host "$HOST_PYTHON" -r "$WORK/host-reqs.txt" pytest
}
gate bootstrap-host-venv host_venv
U="$WORK/venv-unit/bin/python"
H="$WORK/venv-host/bin/python"

gate lint "$U" -m pyflakes "$REPO/hermes_mem" "$REPO/__init__.py" "$REPO/engine" "$REPO/scripts" "$REPO/tests"
pytest_on() { "$1" -VV && "$1" -m pytest -q "$REPO/tests"; }  # the log names the interpreter that ran
gate unit-tests pytest_on "$U"
gate unit-tests-host-python pytest_on "$H"
gate host-integration env PYTHONPATH="$HERMES_HOST_SOURCE" "$H" "$REPO/scripts/host_integration.py" \
  "$HERMES_HOST_SOURCE" "$WORK/host-run" "$WORK/embedding.yaml"
gate real-embedding-eval "$U" "$REPO/scripts/eval_paraphrase.py" "$WORK/eval-run" "$WORK/embedding.yaml" "$WORK/eval.json"
# The default and its control model, one after the other, on the same corpus and chunks.
gate model-comparison "$U" "$REPO/scripts/compare_models.py" "$WORK/compare-run" "$WORK/embedding.yaml" "$WORK/compare.json" \
  bge-m3-int8 minilm
# The documented install/setup/index/tool paths on an installed copy, host venv replacement and reinstall.
# Also the compatibility command for a candidate host: same script, other --host-source/--host-python.
gate installed-copy "$U" "$REPO/scripts/install_acceptance.py" --host-source "$HERMES_HOST_SOURCE" \
  --host-python "$HOST_PYTHON" --work "$WORK/install-run" --cache-home "$CACHE_HOME"

echo
echo "Gate summary (logs in $LOGS):"
status=0
for i in "${!NAMES[@]}"; do
  printf '  %-26s exit=%s\n' "${NAMES[$i]}" "${CODES[$i]}"
  [ "${CODES[$i]}" -eq 0 ] || status=1
done
echo "verify: $([ $status -eq 0 ] && echo PASS || echo FAIL)"
exit $status
