#!/usr/bin/env bash
# SASTO-V G1b relabel: GB200 setup + preflight.
# Prepares the environment and proves the solver stack works BEFORE the real run.
# See HANDOFF-GB200.md for full context. Run from anywhere; it is idempotent.
set -euo pipefail

WORK="${WORK:-$HOME/sasto}"
COMMIT="f46f500c96f7780f81da7f7e4c16814f67ef2af0"
SPLIT_SHA="ca526a068137308ca4bb05325d62bab5a7ad45c81d54566d5fa8e3ef62a91650"
ARCHIVE_SHA="79640406e1e0921c0ccfdc1df7ce51e05a8ecfece2ceccb7dec42c981736beda"

say() { printf '\n=== %s ===\n' "$1"; }
die() { printf '\nFAIL: %s\n' "$1" >&2; exit 1; }

say "inputs present and hash-correct"
[ -f "$WORK/inputs/fea_ml.zip" ] || die "missing $WORK/inputs/fea_ml.zip"
[ -f "$WORK/inputs/split-manifest.json" ] || die "missing $WORK/inputs/split-manifest.json"
echo "${ARCHIVE_SHA}  $WORK/inputs/fea_ml.zip" | sha256sum -c - || die "archive hash mismatch"
echo "${SPLIT_SHA}  $WORK/inputs/split-manifest.json" | sha256sum -c - || die "split hash mismatch"

say "toolchain (PyAMG has no aarch64 wheel; it compiles from sdist)"
if ! command -v g++ >/dev/null 2>&1; then
  sudo apt-get update && sudo apt-get install -y build-essential g++ gfortran git curl
fi
g++ --version | head -1

say "uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

say "repo at frozen commit"
mkdir -p "$WORK"
if [ ! -d "$WORK/repo/.git" ]; then
  git clone https://github.com/erichou1/fea.git "$WORK/repo"
fi
cd "$WORK/repo"
git fetch origin
git checkout modernize/sasto-v
git pull --ff-only origin modernize/sasto-v || true
HEAD_SHA="$(git rev-parse HEAD)"
[ "$HEAD_SHA" = "$COMMIT" ] || die "HEAD is $HEAD_SHA, expected $COMMIT"
echo "HEAD $HEAD_SHA OK"

say "locked python and dependencies"
uv python install 3.11.15
uv sync --frozen --group fea --group test
uv run --frozen python -VV

say "PyAMG preconditioner actually constructs"
uv run --frozen --group fea python - <<'PY'
import numpy as np, scipy, scipy.sparse as sp, pyamg
print("numpy", np.__version__, "scipy", scipy.__version__, "pyamg", pyamg.__version__)
A = (sp.diags([2.0]*50) + sp.diags([-1.0]*49, 1) + sp.diags([-1.0]*49, -1)).tocsr()
pyamg.smoothed_aggregation_solver(A).aspreconditioner(cycle="V")
print("PYAMG_PRECONDITIONER_OK")
PY

say "locked test suite on this architecture"
PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS="" \
  uv run --frozen --group test --group fea python -m pytest -q -p no:cacheprovider

say "two-sample live solver smoke (throwaway root)"
rm -rf /tmp/g1b-smoke
uv run --frozen --group fea python -m sasto.g1b_relabel \
  --root /tmp/g1b-smoke --mode run --limit 2 \
  --split-manifest "$WORK/inputs/split-manifest.json" \
  --expected-split-manifest-sha256 "$SPLIT_SHA" \
  --archive "$WORK/inputs/fea_ml.zip" \
  --expected-fea-archive-sha256 "$ARCHIVE_SHA"

uv run --frozen python - <<'PY'
import json, sys
from pathlib import Path
cases = sorted(Path("/tmp/g1b-smoke/cases").glob("*.json"))
if not cases:
    sys.exit("FAIL: smoke produced no case records")
bad = []
for p in cases:
    c = json.loads(p.read_text())
    pre = json.dumps(c)
    if "preconditioner_unavailable" in pre:
        bad.append((p.name, "preconditioner_unavailable"))
    print(p.name, "eligible=", c.get("eligible"), "reason=", c.get("exclusion_reason"))
if bad:
    sys.exit("FAIL: PyAMG missing at runtime -> {}".format(bad))
print("SMOKE_OK", len(cases), "cases")
PY
rm -rf /tmp/g1b-smoke

say "READY"
cat <<EOF
Preflight passed. Launch the real run with:

  cd $WORK/repo
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  export ROOT=$WORK/out/relabel-gb200-v1 K=32
  mkdir -p $WORK/logs
  for N in \$(seq 1 \$K); do
    nohup uv run --frozen --group fea python -m sasto.g1b_relabel \\
      --root \$ROOT --mode run --shard \$N/\$K \\
      --split-manifest $WORK/inputs/split-manifest.json \\
      --expected-split-manifest-sha256 $SPLIT_SHA \\
      --archive $WORK/inputs/fea_ml.zip \\
      --expected-fea-archive-sha256 $ARCHIVE_SHA \\
      > $WORK/logs/shard-\$N.log 2>&1 &
  done
  wait

Then finalize TWICE and confirm identical digests. See HANDOFF-GB200.md section 5.
EOF
