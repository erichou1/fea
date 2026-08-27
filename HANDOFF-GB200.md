# SASTO-V G1b relabel: GB200 handoff

Target: run the canonical baseline relabel over all 11,178 retained samples on a
GB200 (Grace ARM64 + Blackwell) node, producing a certifiable cohort manifest.

Repo: `https://github.com/erichou1/fea`
Branch: `modernize/sasto-v`

Use the branch tip. Do not pin a commit SHA — documentation commits move it. What
must match exactly is the **code**, verified by these hashes after checkout:

| Path | SHA-256 |
|---|---|
| `src/sasto/g1b_relabel.py` | `1aeaaeaf7902203148d5d98e0ba3904fa558678376d81aca3d2b3374b6365ab8` |
| `src/sasto/voxel_fea.py` | `d258ed803a1b5be7d8e05d3f1d1176d962f482178057ecefa925d360349daece` |
| `uv.lock` | `a3e29520c85af76f98ddba591d2a779f6426a1844d282235d0a4465fe831c35d` |

Solver lineage commit (the code these hashes come from):
`f46f500c96f7780f81da7f7e4c16814f67ef2af0`.

```sh
shasum -a 256 src/sasto/g1b_relabel.py src/sasto/voxel_fea.py uv.lock
```

If any of the three differs, stop — the produced labels would not correspond to
the certified solver.

---

## 0. Read this before you start

**This job does not use the GPU.** The relabel is CPU-bound sparse conjugate
gradient (SciPy + PyAMG). The value of a GB200 node here is the Grace CPU core
count (72 per superchip), not Blackwell. Expect the GPUs to sit idle. GPU work
begins at G2 (model training), not in this stage.

**Run the entire cohort on one machine.** Compliance, stress, and displacement
are floating-point outputs of an iterative solver. Different CPU architectures
and BLAS builds can differ in the last bits. Timing is excluded from the
scientific digests, but solver outputs are not. Do not merge records produced on
macOS with records produced on ARM Linux. Start the GB200 run in a fresh output
root and let it produce all 11,178 records itself.

**PyAMG has no aarch64 wheel.** `pyamg==5.3.0` ships no manylinux aarch64 wheel;
uv will fall back to the sdist and compile it. Without a C++ toolchain the
install fails, and if PyAMG is missing at runtime the solver does not crash — it
returns `preconditioner_unavailable` for every sample and you get a cohort of
11,178 failures that looks superficially like a completed run. Verify the
preflight in step 3 before launching.

---

## 1. Inputs you must copy to the node

Three files are not in Git (too large, or license-restricted):

| File | Size | SHA-256 |
|---|---|---|
| `fea_ml.zip` | 1,347,503,638 B | `79640406e1e0921c0ccfdc1df7ce51e05a8ecfece2ceccb7dec42c981736beda` |
| `split-manifest.json` | 2,616,236 B | `ca526a068137308ca4bb05325d62bab5a7ad45c81d54566d5fa8e3ef62a91650` |
| `near_duplicate_summary.json` | 278,587 B | (verify on arrival) |
| `near_duplicate_verified_pairs.csv` | 57,619 B | (verify on arrival) |

Local source paths:

```
/Users/eric/workspace/sasto-modernization-control/archives/fea_ml.zip
/Users/eric/workspace/sasto-modernization-control/v2/g1/split-manifest.json
/Users/eric/workspace/sasto-modernization-control/v2/near-duplicate-audit/near_duplicate_summary.json
/Users/eric/workspace/sasto-modernization-control/v2/near-duplicate-audit/near_duplicate_verified_pairs.csv
```

Copy them into `$WORK/inputs/` on the node:

```sh
export WORK=$HOME/sasto
mkdir -p $WORK/inputs
# from the Mac:
rsync -avP --partial \
  /Users/eric/workspace/sasto-modernization-control/archives/fea_ml.zip \
  /Users/eric/workspace/sasto-modernization-control/v2/g1/split-manifest.json \
  /Users/eric/workspace/sasto-modernization-control/v2/near-duplicate-audit/near_duplicate_summary.json \
  /Users/eric/workspace/sasto-modernization-control/v2/near-duplicate-audit/near_duplicate_verified_pairs.csv \
  user@gb200:~/sasto/inputs/
```

The archive is redistribution-restricted (no explicit 3DWire license was
located). Do not publish it, do not put it in object storage with public read,
and delete it from the node when the run is done.

---

## 2. Environment

Ubuntu 22.04/24.04 aarch64 assumed. Requires Python 3.11.15 exactly.

```sh
# toolchain — REQUIRED, PyAMG compiles from sdist on ARM
sudo apt-get update
sudo apt-get install -y build-essential g++ gfortran git curl

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# repo
cd $WORK
git clone https://github.com/erichou1/fea.git repo
cd repo
git checkout modernize/sasto-v
# verify the CODE, not the commit
shasum -a 256 src/sasto/g1b_relabel.py src/sasto/voxel_fea.py uv.lock

# python + deps, locked
uv python install 3.11.15
uv sync --frozen --group fea
```

`uv sync --frozen` must succeed with no resolution changes. If it tries to
re-resolve, stop: the lockfile is part of the provenance bundle and must not
move.

---

## 3. Preflight (do not skip)

```sh
cd $WORK/repo

# a. PyAMG actually imports and builds a preconditioner
uv run --frozen --group fea python -c "
import pyamg, scipy, numpy
print('pyamg', pyamg.__version__, 'scipy', scipy.__version__, 'numpy', numpy.__version__)
import numpy as np, scipy.sparse as sp
A = sp.diags([2.0]*50) + sp.diags([-1.0]*49, 1) + sp.diags([-1.0]*49, -1)
pyamg.smoothed_aggregation_solver(A.tocsr()).aspreconditioner(cycle='V')
print('PYAMG_PRECONDITIONER_OK')
"

# b. input hashes match
sha256sum $WORK/inputs/fea_ml.zip $WORK/inputs/split-manifest.json

# c. locked test suite passes on this architecture
uv run --frozen --group test --group fea python -m pytest -q -p no:cacheprovider

# d. two-sample smoke through the real solver, throwaway root
uv run --frozen --group fea python -m sasto.g1b_relabel \
  --root /tmp/g1b-smoke --mode run --limit 2 \
  --split-manifest $WORK/inputs/split-manifest.json \
  --expected-split-manifest-sha256 ca526a068137308ca4bb05325d62bab5a7ad45c81d54566d5fa8e3ef62a91650 \
  --archive $WORK/inputs/fea_ml.zip \
  --expected-fea-archive-sha256 79640406e1e0921c0ccfdc1df7ce51e05a8ecfece2ceccb7dec42c981736beda
```

Gate on (d): open the two files in `/tmp/g1b-smoke/cases/`. Each must show
`"eligible": true` with finite positive `compliance_j`, and
`preconditioner_identity` containing `pyamg.smoothed_aggregation_solver`. If you
see `preconditioner_unavailable`, the toolchain step failed — fix it before
launching the real run. Delete `/tmp/g1b-smoke` afterward.

---

## 4. Launch the sharded run

Sharding is deterministic by `SHA-256(namespace || NUL || sample_id) mod K`, so
shards are disjoint by construction and each writes only its own case files.
Records are append-only and digest-verified; resume regenerates only what is
missing.

Use 32 shards. Each process is single-threaded in the CG path, so oversubscribe
modestly rather than pinning all 72 cores.

```sh
cd $WORK/repo
export ROOT=$WORK/out/relabel-gb200-v1
export K=32
mkdir -p $WORK/logs

for N in $(seq 1 $K); do
  nohup uv run --frozen --group fea python -m sasto.g1b_relabel \
    --root $ROOT --mode run --shard $N/$K \
    --split-manifest $WORK/inputs/split-manifest.json \
    --expected-split-manifest-sha256 ca526a068137308ca4bb05325d62bab5a7ad45c81d54566d5fa8e3ef62a91650 \
    --archive $WORK/inputs/fea_ml.zip \
    --expected-fea-archive-sha256 79640406e1e0921c0ccfdc1df7ce51e05a8ecfece2ceccb7dec42c981736beda \
    > $WORK/logs/shard-$N.log 2>&1 &
done
wait
```

Also set threading to 1 per process so 32 workers do not each spawn 72 BLAS
threads and thrash:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

Progress:

```sh
watch -n 30 'ls '$ROOT'/cases | wc -l'
```

Expected wall clock: the Mac sustained ~12 solves/min single-process. At 32
concurrent Grace cores, expect roughly **25–45 minutes** for all 11,178, subject
to memory bandwidth. Peak RSS is a few hundred MB per worker.

---

## 5. Finalize

Only after every shard process has exited:

```sh
cd $WORK/repo
uv run --frozen --group fea python -m sasto.g1b_relabel \
  --root $ROOT --mode finalize \
  --split-manifest $WORK/inputs/split-manifest.json \
  --expected-split-manifest-sha256 ca526a068137308ca4bb05325d62bab5a7ad45c81d54566d5fa8e3ef62a91650 \
  --near-duplicate-summary $WORK/inputs/near_duplicate_summary.json \
  --near-duplicate-pairs $WORK/inputs/near_duplicate_verified_pairs.csv
```

Then run it a second time. Finalize is deterministic; the two runs must produce
byte-identical `g1b-summary.json`. If they differ, something is nondeterministic
and the result is not promotable.

```sh
sha256sum $ROOT/g1b-summary.json
uv run ... --mode finalize   # again
sha256sum $ROOT/g1b-summary.json   # must match
```

---

## 6. What to send back

Do not ship the archive back. Ship only the evidence:

```sh
cd $ROOT
tar czf $WORK/g1b-gb200-evidence.tgz \
  cohort-manifest.json cluster-role-manifest.json g1b-summary.json \
  generation-*.json invocations/ cases/
sha256sum $WORK/g1b-gb200-evidence.tgz
```

Report back:

1. `sha256sum` of the tarball
2. `g1b-summary.json` in full
3. cohort counts: eligible total, and excluded broken out by reason
4. cluster count and role distribution
5. the two finalize digests, showing they matched
6. `uname -m`, `python -VV`, `pip list` for scipy/pyamg/numpy versions
7. total wall clock and peak memory
8. confirmation that `/tmp/g1b-smoke` and `inputs/fea_ml.zip` were deleted

---

## 7. Invariants you must not break

- **Confirmation stays sealed.** Confirmation-role samples get explicit
  `confirmation_sealed` exclusion records. Their payloads are never opened. If
  any confirmation sample produces a solve record, the run is void.
- **No solver or config changes.** Force is fixed at `(0, 0, -100 N)`,
  self-weight off, admission tolerance `2e-8`. Any change to the solver or load
  contract invalidates every label and requires a new versioned output root.
- **No edits to `uv.lock`, `pyproject.toml`, or `src/sasto/`.** These are inside
  the hashed provenance bundle. A change there means the produced records no
  longer correspond to the certified code.
- **Append-only.** Never delete or overwrite a case record to "retry" it. If a
  record is wrong, the run is wrong; report it.
- **Cohort membership is solver-validity only** — connectivity, solver status,
  residual bound, finite positive outputs, stable loaded-node set. Never filter
  on target magnitude.

---

## 8. If it goes wrong

| Symptom | Cause | Action |
|---|---|---|
| `preconditioner_unavailable` on every sample | PyAMG missing/failed to compile | Install `build-essential g++`, re-run `uv sync --frozen --group fea`, redo preflight |
| `uv sync` re-resolves | Wrong Python or lock drift | `uv python install 3.11.15`, ensure `--frozen` |
| `split manifest sha256 mismatch` | Corrupt or wrong file | Re-copy, verify hash |
| `archive sha256 mismatch` | Truncated transfer | Re-rsync with `--partial`, verify hash |
| Shards write same file | Should be impossible | Stop everything and report; sharding is disjoint by hash |
| Finalize digests differ | Nondeterminism | Stop; do not promote. Report both digests |
| OOM with 32 shards | Memory bandwidth | Drop to `K=16`, resume — existing records are kept |

Resume is always safe: relaunch the same shard command. Verified records are
skipped, only missing ones are generated.
