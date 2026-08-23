from __future__ import annotations

import json

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


def test_fit_role_validation_consumes_entire_request_before_limit_and_before_archive_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object,
) -> None:
    from sasto import fit_probe
    from sasto.fit_probe import FitOnlyAccessError, run_fit_probe, select_fit_sample_ids

    manifest = {
        "partitions": {
            "fit": {"sample_ids": ["05134", "04900"]},
            "development": {"sample_ids": ["00001"]},
            "calibration": {"sample_ids": ["00000"]},
            "confirmation": {"sample_ids": ["00002"]},
        }
    }
    with pytest.raises(FitOnlyAccessError, match="non-fit"):
        select_fit_sample_ids(manifest, ["05134", "00000"], limit=1)
    with pytest.raises(FitOnlyAccessError, match="unique"):
        select_fit_sample_ids(manifest, ["05134", "05134"], limit=1)
    with pytest.raises(FitOnlyAccessError, match="positive integer"):
        select_fit_sample_ids(manifest, ["05134"], limit=True)
    malformed_manifest = json.loads(json.dumps(manifest))
    malformed_manifest["partitions"]["fit"]["sample_ids"].append("05134")
    with pytest.raises(FitOnlyAccessError, match="invalid fit membership"):
        select_fit_sample_ids(malformed_manifest, ["05134"], limit=1)

    split_manifest = tmp_path / "split.json"  # type: ignore[operator]
    split_manifest.write_text(json.dumps(manifest), encoding="utf-8")  # type: ignore[union-attr]
    archive_opens: list[object] = []

    def spy_archive_open(*args: object, **kwargs: object) -> object:
        archive_opens.append((args, kwargs))
        raise AssertionError("archive must not be opened for rejected role request")

    monkeypatch.setattr(fit_probe.zipfile, "ZipFile", spy_archive_open)
    with pytest.raises(FitOnlyAccessError, match="non-fit"):
        run_fit_probe(
            split_manifest=split_manifest, archive_path=tmp_path / "forbidden.zip",  # type: ignore[operator]
            sample_ids=["05134", "00000"], limit=1, fixed_force=(0.0, 0.0, -100.0),
        )
    assert archive_opens == []
