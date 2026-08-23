# SASTO-V evidence-contract bootstrap

## Canonical scope

`src/sasto` is the canonical SASTO-V evidence-contract package. It currently
provides the G0 substrate only: named proxy-target contracts, deterministic
family-disjoint roles, manifest admission, local digital topology utilities,
and a public smoke artifact. It does **not** establish FEA validity, engineering
safety, construction readiness, surrogate accuracy, or paper results.

The sole canonical G0 SASTO-PA admission runner is
`evaluate_erosion_candidate` in `sasto.sasto_pa`. It applies the topology gate
before any surrogate proxy gate. The canonical digital topology convention is
**6-connected foreground and 26-connected background**. Any erosion runner must
call `is_simple_point_6_26`; reversed 26/6 conventions are noncanonical.

## Target contract

Every constrained response is addressed by its immutable name, with a physical
unit, inequality direction, threshold, and explicit `normalization`. Lists or
tensor positions are not a compliance API. Absolute targets must not use unit
`1`. A baseline ratio must use unit `1`, an explicit `_ratio` name (for example
`compliance_ratio`), and a non-empty `base_target` provenance label; the base
reference is external metadata and need not be another constrained registry
member. The smoke registry uses a compliance ratio plus absolute proxy targets:

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
rejects missing assignments, unknown IDs, duplicate IDs, and family leakage.

## Evidence records

`run-manifest.json` uses schema version `1.0.0` and records named targets, a
declared split-artifact logical ID and canonical family-split digest, plus
SHA-256 digests for every declared input and output. Record paths are portable
POSIX-relative paths beneath the manifest's artifact root. Build and verification
reject absolute/traversal paths, symlinks, non-regular files, files outside that
root, malformed records, and changed hashes. `verify_run_manifest` recomputes
the canonical digest after parsing the declared split artifact. Zero-failure
binomial helpers are intentionally separate from conformal coverage; do not
translate one claim into the other.

## Commands

```sh
make smoke
make verify-artifact
make reproduce-paper
make test
```

`make smoke` writes one deterministic artifact at `artifacts/smoke` and then
verifies it. Existing artifact roots are deliberately rejected rather than
silently overwritten. To generate a new run, choose a fresh `ARTIFACT_DIR`.
`make verify-artifact ARTIFACT_DIR=...` verifies the selected immutable run.

`make reproduce-paper` is deliberately a **fail-closed G1 preflight**, not a
paper-results runner. It checks the declared configuration/data/runner and
notice/license gates, prints `G1 UNAVAILABLE`, and exits nonzero until the real
runner and assets exist and have been independently validated. It must never be
used to imply that paper results were reproduced.

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
choice on Eric's behalf. Full reproduction is also data-, solver-, and
locked-environment-gated and is outside G0.
