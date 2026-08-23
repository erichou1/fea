# SASTO-V evidence-contract bootstrap

## Canonical scope

`src/sasto` is the canonical SASTO-V evidence-contract package. It currently
provides the G0 substrate only: named proxy-target contracts, deterministic
family-disjoint roles, manifest admission, global digital topology utilities,
and a public smoke artifact. It does **not** establish FEA validity, engineering
safety, construction readiness, surrogate accuracy, or paper results.

The canonical digital topology convention is **6-connected foreground and
26-connected background with one explicit exterior node**.  Two contracts are
intentionally separate:

* `exact_global_6_26` is the authoritative offline oracle.  It compares complete
  pre/post foreground-6 and exterior-aware-background-26 component counts.
  `is_simple_point_6_26` remains a backward-compatible alias for that **exact,
  intentionally slow** oracle.
* `conservative_local_6_26` is the production admissibility filter.  Its 26-bit
  local proof makes accepted deletions sound for those two counts, including
  boundary/exterior semantics; it can and does false-reject exact-admissible
  deletions.  It is never exact-equivalent to the oracle.

Both APIs accept nested Boolean grids or 3-D NumPy `dtype=bool` arrays and fail
closed for malformed/non-Boolean inputs without mutating the caller's volume.
The local production path must use
`apply_conservative_deletions_sequentially`, not a set-wide one-time decision;
it independently enforces `protected_mask` and `edit_mask` policy inputs before
each recheck.  Neither predicate implies physical safety, mesh-component
guarantees, printability, or full digital homotopy.

`exact_topology_preflight_6_26` deliberately **reports**, rather than silently
rejecting, `foreground_6_components`,
`background_26_components_with_exterior`, `has_cavities`, `shape`,
`occupied_count`, exterior-boundary semantics, and a canonical input SHA-256.
The caller must impose connectedness/cavity policy explicitly.  A conservative
trajectory artifact is built with `topology_artifact_record`, whose schema has
`topology_mode: "conservative_local_6_26"`, the exact preflight facts,
`campaign_hash`, and `sequential_recheck: true`.

A deterministic differential campaign is available through
`make topology-campaign`.  Its default is one million seeded 3³ neighborhoods
against an independent exact 3³ component-count reference, plus exhaustive
2³ volumes and historical/adversarial witnesses; it reports false accepts and
exact-only false rejects separately, including recall loss.  Supply archived
occupancies explicitly to add the required ten-sample real-64³ rate distribution:

```sh
make topology-campaign TOPOLOGY_DATA_ROOT=/path/to/fea_ml/data/runs_real
```

The soundness gate requires zero false accepts.  The 1,000 tests/s requirement
applies to the optional real-data benchmark, while false-reject recall loss is a
primary reported production cost rather than a hidden equivalence claim.

## Target contract

Every constrained response is addressed by its immutable name, with a physical
unit, inequality direction, finite numeric threshold, and explicit
`normalization`. Names and units must be non-empty strings; thresholds and
runtime responses must be finite non-boolean real numbers, including common
finite NumPy real scalars such as `numpy.int64` and `numpy.float32`. Python and
NumPy booleans, strings, complex values, NaN, and infinities are rejected. Lists or tensor positions
are not a compliance API. Absolute targets must not use unit
`1`. A baseline ratio must use unit `1`, an explicit `_ratio` name (for example
`compliance_ratio`), and a non-empty `base_target` provenance label; the base
reference is external metadata and need not be another constrained registry
member. The smoke registry uses a compliance ratio plus absolute proxy targets:

Accepted thresholds are normalized to JSON-native `float` values at construction,
so a manifest can serialize the same contract independently of whether its caller
provided a Python or NumPy finite real scalar.

| Target | Unit | Normalization | Base target | Direction | Threshold |
| --- | --- | --- | --- | --- | ---: |
| `compliance_ratio` | 1 | `baseline_ratio` | `compliance` | upper | 1.15 |
| `max_von_mises` | Pa | `absolute` | — | upper | 5,000,000 |
| `max_displacement` | m | `absolute` | — | upper | 0.028 |

These are explicitly **linear-elastic simulator proxy constraints**, not
structural-code or material acceptance limits.

## Data roles and leakage gate

`build_family_split` requires explicit `sample_id` and `family_id` fields. A
single seeded family shuffle creates the fixed `fit` (60%), `development`
(20%), `calibration` (10%), and `confirmation` (10%) roles. Every positive
functional role receives at least one family; all remaining families are
allocated deterministically to stay as close as possible to those fractions.
Every derivative of a family remains in exactly one role; `validate_family_split`
rejects missing assignments, unknown IDs, duplicate IDs, and family leakage. The
self-describing split artifact also carries a sorted `sample_to_family` mapping;
artifact verification independently validates its schema, source coverage,
family allocation, and role isolation rather than trusting its digest alone.

## Evidence records

`run-manifest.json` uses schema version `1.0.0` and records named targets, a
declared split-artifact logical ID and canonical family-split digest, plus
SHA-256 digests for every declared input and output. Record paths are portable
POSIX-relative paths beneath the manifest's artifact root. Build and verification
reject absolute/traversal paths, symlinks (including lexical manifest-path or
artifact-root components), non-regular files, files outside that root, malformed
records, and changed hashes. Verification opens the artifact root and every path
component through held directory descriptors with `O_NOFOLLOW` where supported;
leaves use `O_NOFOLLOW|O_NONBLOCK`, are `fstat`-checked as regular on that same
descriptor, and are read and hashed from that descriptor. `_verify_records`
returns those exact byte snapshots, so the split is parsed and semantically
validated from the bytes whose record digest passed rather than from a later
pathname reread. Before returning, the verifier reopens the manifest and every
declared record through the held root descriptor and rejects observed inode,
metadata, byte, or hash drift. This is a bounded verification snapshot only:
external mutation after return is outside the snapshot and no filesystem
immutability or continuing-soundness claim is made. Zero-failure
binomial helpers are intentionally separate from conformal coverage; do not
translate one claim into the other.

### External manifest trust anchor

The manifest is a payload, not its own authenticator. Every verification requires
a caller-supplied lowercase SHA-256 digest of the **exact manifest bytes** via
`expected_manifest_sha256` (or `--expected-manifest-sha256` in the CLI). The
verifier opens the manifest leaf no-follow and nonblocking through its held
artifact-root descriptor, `fstat`s that same descriptor as a regular file, and
hashes before parsing exact captured bytes. Consequently, changing a run ID,
target, record path, or a self-consistent path/hash pair changes the anchored
digest and is rejected.
An unauthenticated sidecar in the artifact directory is not an external trust
anchor and is insufficient. The expected digest must be retained or transmitted
by a separately protected control plane. Identical manifest bytes remain valid
after artifact relocation when supplied with the same external digest.

## Commands

```sh
make smoke
make verify-artifact EXPECTED_MANIFEST_SHA256=<externally-recorded-digest>
make reproduce-paper
make test
make test-locked
make test-g1-locked
make topology-campaign
```

`make smoke` writes one deterministic artifact at `artifacts/smoke` and then
prints the manifest digest and verifies it using that emitted value. Persist that
digest outside the artifact directory before a later independent verification.
Existing artifact roots are deliberately rejected rather than
silently overwritten. To generate a new run, choose a fresh `ARTIFACT_DIR`.
`make verify-artifact ARTIFACT_DIR=... EXPECTED_MANIFEST_SHA256=...` verifies
the selected immutable run against its external anchor.

### Canonical locked Python environment

G0 uses CPython 3.11.15: `.python-version` pins that interpreter and
`pyproject.toml` bounds the package to `>=3.11,<3.12`. Runtime dependencies
(including NumPy) and the test dependency (pytest) are declared in
`pyproject.toml`; the build requirement is exact-pinned; and committed
`uv.lock` is the canonical resolved environment. From a clean local virtual
environment, run:

```sh
rm -rf .venv
uv sync --frozen --group test
make test-locked
```

`make test-locked` repeats the frozen sync and runs pytest through `uv`; it does
not regenerate the lock. This G0 Python lock does **not** lock CUDA, an FEA
solver, recovered benchmark data, or any G1 execution environment.

`make reproduce-paper` is deliberately a **fail-closed G1 preflight**, not a
paper-results runner. It checks the declared configuration/data/runner and
notice/license gates, prints `G1 UNAVAILABLE`, and exits nonzero until the real
runner and assets exist and have been independently validated. It must never be
used to imply that paper results were reproduced.

## Canonical T0 Hex8 verifier (G1a Slice B)

`src/sasto/voxel_fea.py` owns the canonical regular full-integration Hex8
linear-elastic voxel proxy; it imports no executable legacy solver. Occupancy
`[a0,a1,a2]` maps to physical `(x,y,z)`, displacement DOFs are physical
`(x,y,z)` triples, the minimum occupied physical-x face is fully fixed, and the
protected load face is the maximum occupied physical-x face. Self-weight is a
separately reported body force in physical negative-z. The fixed benchmark
force is distributed across the declared load nodes while preserving its exact
requested vector, and a caller may lock the expected load-node count and exact
node-coordinate set for baseline/candidate comparisons.

Every solve is an append-only serializable `success` or `failure` record.
Success records include stress (maximum and p99 in Pa), maximum displacement
(m), compliance (J), counts, force sums, solver/preconditioner identity,
iterations, residual, input/config digests, bounded timing, and a scientific
digest. The scientific digest intentionally excludes wall time and iterative
residual roundoff. Invalid, disconnected, undersized, unsupported, load-free,
non-Boolean, nonfinite, nonconvergent, or non-postprocessable inputs fail
closed without modifying occupancy. The direct sparse path is restricted to
small independent V&V fixtures; the canonical path requires SciPy CG with a
PyAMG preconditioner.

Use the frozen V&V group and test command:

```sh
uv sync --frozen --group test --group fea
make test-fea-locked
```

`make fit-only-probe` invokes `sasto.fit_probe`. It requires explicit
`SPLIT_MANIFEST`, `FEA_ARCHIVE`, and a new `FIT_PROBE_OUTPUT`; it validates
that every requested sample is fit-role and that fit/development/calibration/
confirmation memberships are disjoint **before opening any archive payload**.
It has no confirmation override. The archived historical probe sample `00000`
is calibration-role in the frozen manifest, so this command must reject it
before payload access.

## Legacy pipelines are noncanonical

The archived `fea_ml/` module CMA-ES optimizer, `run_batch_all.py`,
`run_opt_part_aware.py`, and `run_opt_uniform.py` are legacy/historical inputs,
not SASTO-V runners or evidence generators. In particular, no legacy script
may be used to create a release claim without being wrapped by this contract and
separately promoted in a later gate.

## Release gates still unresolved

No open-source license has been selected. `LICENSE_STATUS.md`,
`THIRD_PARTY_NOTICES.md`, and `DATA_NOTICE.md` record unresolved external
choices without selecting a license. A project license plus reviewed notices
remain a required release gate; this bootstrap intentionally makes no license
choice on Eric's behalf. Full reproduction remains data- and solver-gated and
is outside G0; the G0 Python build/test environment is locked as documented
above.
