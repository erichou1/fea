from __future__ import annotations

import pytest


def test_fit_probe_selection_is_manifest_bound_and_denies_nonfit_roles() -> None:
    from sasto.fit_probe import FitOnlyAccessError, select_fit_sample_ids

    manifest = {
        "partitions": {
            "fit": {"sample_ids": ["00003", "00004"]},
            "development": {"sample_ids": ["00001"]},
            "calibration": {"sample_ids": ["00000"]},
            "confirmation": {"sample_ids": ["00002"]},
        }
    }
    assert select_fit_sample_ids(manifest, ["00004", "00003"], limit=1) == ["00004"]
    with pytest.raises(FitOnlyAccessError, match="non-fit"):
        select_fit_sample_ids(manifest, ["00000"], limit=1)
    with pytest.raises(FitOnlyAccessError, match="confirmation"):
        select_fit_sample_ids(manifest, ["00002"], limit=1)
