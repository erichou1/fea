PYTHON ?= python3
ARTIFACT_DIR ?= artifacts/smoke

.PHONY: smoke verify-artifact reproduce-paper test

smoke:
	PYTHONPATH=src $(PYTHON) -m sasto.smoke --fixture fixtures/smoke/families.json --output $(ARTIFACT_DIR)
	$(MAKE) verify-artifact ARTIFACT_DIR=$(ARTIFACT_DIR)

verify-artifact:
	PYTHONPATH=src $(PYTHON) -m sasto.verify_artifact $(ARTIFACT_DIR)/run-manifest.json

reproduce-paper:
	PYTHONPATH=src $(PYTHON) -m sasto.reproduce_paper

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q
