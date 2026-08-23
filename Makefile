PYTHON ?= python3
ARTIFACT_DIR ?= artifacts/smoke

# Pass caller-controlled values through the environment, never recipe source.
export ARTIFACT_DIR
export EXPECTED_MANIFEST_SHA256
export PYTHON

.PHONY: smoke verify-artifact reproduce-paper test test-locked

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
	PYTHONPATH=src "$(PYTHON)" -m sasto.reproduce_paper

test:
	PYTHONPATH=src "$(PYTHON)" -m pytest -q

test-locked:
	uv sync --frozen --group test
	uv run --frozen --group test python -m pytest -q
