PYTHON ?= python3
ARTIFACT_DIR ?= artifacts/smoke

.PHONY: smoke verify-artifact reproduce-paper test test-locked

smoke:
	PYTHONPATH=src $(PYTHON) -m sasto.smoke --fixture fixtures/smoke/families.json --output $(ARTIFACT_DIR)
	@digest=$$(shasum -a 256 "$(ARTIFACT_DIR)/run-manifest.json" | awk '{print $$1}'); \
	printf 'external_manifest_sha256=%s\n' "$$digest"; \
	$(MAKE) verify-artifact ARTIFACT_DIR=$(ARTIFACT_DIR) EXPECTED_MANIFEST_SHA256=$$digest

verify-artifact:
	@test -n "$(EXPECTED_MANIFEST_SHA256)" || (printf '%s\n' 'EXPECTED_MANIFEST_SHA256 is required from an external trust anchor' >&2; exit 2)
	PYTHONPATH=src $(PYTHON) -m sasto.verify_artifact --expected-manifest-sha256 "$(EXPECTED_MANIFEST_SHA256)" $(ARTIFACT_DIR)/run-manifest.json

reproduce-paper:
	PYTHONPATH=src $(PYTHON) -m sasto.reproduce_paper

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

test-locked:
	uv sync --frozen --group test
	uv run --frozen --group test python -m pytest -q
