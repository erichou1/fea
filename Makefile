PYTHON ?= python3
ARTIFACT_DIR ?= artifacts/smoke

# Pass caller-controlled values through the environment, never recipe source.
export ARTIFACT_DIR
export EXPECTED_MANIFEST_SHA256
export PYTHON

.PHONY: smoke verify-artifact reproduce-paper test test-locked test-g1-locked test-fea-locked topology-campaign fit-only-probe

smoke:
	@set -eu; \
	PYTHONPATH=src "$$PYTHON" -m sasto.smoke --fixture fixtures/smoke/families.json --output "$$ARTIFACT_DIR"; \
	digest=$$(PYTHONPATH=src "$$PYTHON" -c 'from pathlib import Path; from sasto.manifest import sha256_file; import sys; print(sha256_file(Path(sys.argv[1])))' "$$ARTIFACT_DIR/run-manifest.json"); \
	printf 'external_manifest_sha256=%s\n' "$$digest"; \
	PYTHONPATH=src "$$PYTHON" -m sasto.verify_artifact --expected-manifest-sha256 "$$digest" "$$ARTIFACT_DIR/run-manifest.json"

verify-artifact:
	@set -eu; \
	if [ -z "$${EXPECTED_MANIFEST_SHA256:-}" ]; then \
		printf '%s\n' 'EXPECTED_MANIFEST_SHA256 is required from an external trust anchor' >&2; \
		exit 2; \
	fi; \
	PYTHONPATH=src "$$PYTHON" -m sasto.verify_artifact --expected-manifest-sha256 "$$EXPECTED_MANIFEST_SHA256" "$$ARTIFACT_DIR/run-manifest.json"

reproduce-paper:
	@set -eu; \
	PYTHONPATH=src "$$PYTHON" -m sasto.reproduce_paper

test:
	@set -eu; \
	PYTHONPATH=src "$$PYTHON" -m pytest -q

test-locked:
	uv sync --frozen --group test --group fea
	uv run --frozen --group test --group fea python -m pytest -q

test-g1-locked:
	uv sync --frozen --group test --group fea
	uv run --frozen --group test --group fea python -m pytest -q tests/test_topology_g1.py tests/test_voxel_fea_g1.py tests/test_fit_probe_g1.py

test-fea-locked:
	uv sync --frozen --group test --group fea
	uv run --frozen --group test --group fea python -m pytest -q tests/test_voxel_fea_g1.py tests/test_fit_probe_g1.py

fit-only-probe:
	@set -eu; \
	if [ -z "$${SPLIT_MANIFEST:-}" ] || [ -z "$${FEA_ARCHIVE:-}" ] || [ -z "$${FIT_PROBE_OUTPUT:-}" ]; then \
		printf '%s\n' 'SPLIT_MANIFEST, FEA_ARCHIVE, and FIT_PROBE_OUTPUT are required' >&2; \
		exit 2; \
	fi; \
	uv run --frozen --group fea python -m sasto.fit_probe --split-manifest "$$SPLIT_MANIFEST" --archive "$$FEA_ARCHIVE" --output "$$FIT_PROBE_OUTPUT" --limit "$${FIT_PROBE_LIMIT:-4}" --fixed-force-z "$${FIT_PROBE_FORCE_Z:--100.0}"

topology-campaign:
	@set -eu; \
	if [ -n "$${TOPOLOGY_DATA_ROOT:-}" ]; then \
		uv run --frozen --group test python -m sasto.topology_campaign --neighborhoods "$${TOPOLOGY_NEIGHBORHOODS:-1000000}" --data-root "$$TOPOLOGY_DATA_ROOT"; \
	else \
		uv run --frozen --group test python -m sasto.topology_campaign --neighborhoods "$${TOPOLOGY_NEIGHBORHOODS:-1000000}"; \
	fi
