PYTHON ?= python3
ARTIFACT_DIR ?= artifacts/smoke

.PHONY: smoke verify-artifact test

smoke:
	PYTHONPATH=src $(PYTHON) -m sasto.smoke --fixture fixtures/smoke/families.json --output $(ARTIFACT_DIR)
	$(MAKE) verify-artifact ARTIFACT_DIR=$(ARTIFACT_DIR)

verify-artifact:
	PYTHONPATH=src $(PYTHON) -m sasto.verify_artifact $(ARTIFACT_DIR)/run-manifest.json

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q
