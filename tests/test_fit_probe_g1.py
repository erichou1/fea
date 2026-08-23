from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np

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
            expected_split_manifest_sha256=hashlib.sha256(json.dumps(manifest).encode("utf-8")).hexdigest(),
            expected_archive_sha256="0" * 64,
            sample_ids=["05134", "00000"], limit=1, fixed_force=(0.0, 0.0, -100.0),
        )
    assert archive_opens == []


def test_fit_probe_rejects_tampered_split_before_archive_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An external split anchor rejects semantic role tampering before archive access."""
    from sasto import fit_probe
    from sasto.fit_probe import FitOnlyAccessError, run_fit_probe

    frozen = {
        "partitions": {
            "fit": {"sample_ids": ["05134", "04900", "02606", "08966"]},
            "development": {"sample_ids": ["00001"]},
            "calibration": {"sample_ids": ["00000"]},
            "confirmation": {"sample_ids": ["00002"]},
        }
    }
    split_manifest = tmp_path / "split.json"
    frozen_bytes = json.dumps(frozen, sort_keys=True).encode("utf-8")
    split_manifest.write_bytes(frozen_bytes)
    tampered = json.loads(frozen_bytes)
    tampered["partitions"]["fit"]["sample_ids"].append("00000")
    tampered["partitions"]["calibration"]["sample_ids"].remove("00000")
    split_manifest.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
    archive_opens: list[object] = []

    def forbidden_archive_open(*args: object, **kwargs: object) -> object:
        archive_opens.append((args, kwargs))
        raise AssertionError("archive must not open after split-anchor rejection")

    monkeypatch.setattr(fit_probe.zipfile, "ZipFile", forbidden_archive_open)
    with pytest.raises(FitOnlyAccessError, match="split manifest sha256 mismatch"):
        run_fit_probe(
            split_manifest=split_manifest,
            archive_path=tmp_path / "archive.zip",
            expected_split_manifest_sha256=hashlib.sha256(frozen_bytes).hexdigest(),
            expected_archive_sha256="0" * 64,
            sample_ids=["00000"],
            limit=1,
            fixed_force=(0.0, 0.0, -100.0),
        )
    assert archive_opens == []


def test_fit_probe_rejects_wrong_archive_digest_before_zip_member_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sasto import fit_probe
    from sasto.fit_probe import FitOnlyAccessError, run_fit_probe

    manifest = {
        "partitions": {
            "fit": {"sample_ids": ["05134"]},
            "development": {"sample_ids": ["00001"]},
            "calibration": {"sample_ids": ["00000"]},
            "confirmation": {"sample_ids": ["00002"]},
        }
    }
    split_manifest = tmp_path / "split.json"
    split_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    split_manifest.write_bytes(split_bytes)
    archive_path = tmp_path / "archive.zip"
    archive_path.write_bytes(b"not a zip payload")
    archive_opens: list[object] = []

    def forbidden_archive_open(*args: object, **kwargs: object) -> object:
        archive_opens.append((args, kwargs))
        raise AssertionError("Zip payload must not open after archive-anchor rejection")

    monkeypatch.setattr(fit_probe.zipfile, "ZipFile", forbidden_archive_open)
    with pytest.raises(FitOnlyAccessError, match="archive sha256 mismatch"):
        run_fit_probe(
            split_manifest=split_manifest,
            archive_path=archive_path,
            expected_split_manifest_sha256=hashlib.sha256(split_bytes).hexdigest(),
            expected_archive_sha256="0" * 64,
            sample_ids=["05134"],
            limit=1,
            fixed_force=(0.0, 0.0, -100.0),
        )
    assert archive_opens == []


def test_fit_probe_cli_requires_both_external_hash_anchors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    from sasto import fit_probe

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fit_probe", "--split-manifest", str(tmp_path / "split.json"), "--archive", str(tmp_path / "archive.zip"),
            "--expected-fea-archive-sha256", "0" * 64, "--output", str(tmp_path / "output.json"),
        ],
    )
    assert fit_probe.main() == 2
    captured = capsys.readouterr()
    assert "expected split manifest sha256" in captured.out.lower()


def test_fit_probe_cli_creates_output_once_through_append_only_writer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from sasto import fit_probe

    output = tmp_path / "output.json"
    result = {
        "schema_version": "1.1.0", "role": "fit", "selected_role": "fit", "selected_sample_ids": ["05134"],
        "split_manifest_sha256": "1" * 64, "archive_sha256": "2" * 64,
        "nonfit_payload_access_count": 0, "sample_count": 1, "records": [],
    }
    monkeypatch.setattr(fit_probe, "run_fit_probe", lambda **_: result)
    arguments = [
        "fit_probe", "--split-manifest", str(tmp_path / "split.json"), "--expected-split-manifest-sha256", "1" * 64,
        "--archive", str(tmp_path / "archive.zip"), "--expected-fea-archive-sha256", "2" * 64, "--output", str(output),
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    assert fit_probe.main() == 0
    original = output.read_bytes()
    assert json.loads(original) == result
    assert fit_probe.main() == 2
    assert output.read_bytes() == original


def test_make_fit_only_probe_requires_quoted_external_anchors_and_never_executes_them(tmp_path: Path) -> None:
    repository = Path(__file__).parents[1]
    marker = tmp_path / "unexpected-marker"
    environment = {
        **os.environ,
        "SPLIT_MANIFEST": str(tmp_path / "split.json"),
        "FEA_ARCHIVE": str(tmp_path / "archive.zip"),
        "FIT_PROBE_OUTPUT": str(tmp_path / "output.json"),
    }
    missing = subprocess.run(
        ["make", "fit-only-probe"], cwd=repository, env=environment, capture_output=True, text=True, check=False,
    )
    assert missing.returncode == 2
    assert "EXPECTED_SPLIT_MANIFEST_SHA256 and EXPECTED_FEA_ARCHIVE_SHA256 are required" in missing.stderr

    malicious = subprocess.run(
        ["make", "fit-only-probe"],
        cwd=repository,
        env={
            **environment,
            "EXPECTED_SPLIT_MANIFEST_SHA256": "0" * 64,
            "EXPECTED_FEA_ARCHIVE_SHA256": '0"; touch {}; #'.format(marker),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert malicious.returncode == 2, malicious.stdout + malicious.stderr
    assert not marker.exists(), malicious.stdout + malicious.stderr


def test_frozen_hashes_admit_exactly_the_four_fit_ids_and_record_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sasto import fit_probe

    fit_ids = ["05134", "04900", "02606", "08966"]
    manifest = {
        "partitions": {
            "fit": {"sample_ids": fit_ids},
            "development": {"sample_ids": ["00001"]},
            "calibration": {"sample_ids": ["00000"]},
            "confirmation": {"sample_ids": ["00002"]},
        }
    }
    split_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    split_manifest = tmp_path / "split.json"
    split_manifest.write_bytes(split_bytes)
    archive_path = tmp_path / "archive.zip"
    occupancy = io.BytesIO()
    np.savez(occupancy, data=np.ones((64, 64, 64), dtype=np.uint8))
    with zipfile.ZipFile(archive_path, "w") as archive:
        for sample_id in fit_ids:
            archive.writestr("fea_ml/data/runs_real/{}/occ.npz".format(sample_id), occupancy.getvalue())
            archive.writestr("fea_ml/data/runs_real/{}/meta.json".format(sample_id), '{"voxel_size": 1.0}')
    monkeypatch.setattr(fit_probe, "solve_voxels", lambda *_: {"status": "success"})

    result = fit_probe.run_fit_probe(
        split_manifest=split_manifest,
        archive_path=archive_path,
        expected_split_manifest_sha256=hashlib.sha256(split_bytes).hexdigest(),
        expected_archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        sample_ids=fit_ids,
        limit=4,
        fixed_force=(0.0, 0.0, -100.0),
    )

    assert result["selected_role"] == "fit"
    assert result["selected_sample_ids"] == fit_ids
    assert result["split_manifest_sha256"] == hashlib.sha256(split_bytes).hexdigest()
    assert result["archive_sha256"] == hashlib.sha256(archive_path.read_bytes()).hexdigest()
    expected_members = [
        member for sample_id in fit_ids
        for member in (
            "fea_ml/data/runs_real/{}/occ.npz".format(sample_id),
            "fea_ml/data/runs_real/{}/meta.json".format(sample_id),
        )
    ]
    assert result["execution_mode"] == "live_anchored_solver_run"
    assert result["fixed_total_force_n"] == [0.0, 0.0, -100.0]
    assert result["admission_relative_tolerance"] == 2e-8
    assert result["archive_payload_members"] == expected_members
    assert result["fit_payload_access_count"] == len(expected_members)
    assert result["nonfit_payload_access_count"] == 0
    assert [record["sample_id"] for record in result["records"]] == fit_ids
    assert "00000" not in json.dumps(result)


def test_payload_access_ledger_rejects_unselected_member_before_its_payload_is_read(tmp_path: Path) -> None:
    from sasto.fit_probe import FitOnlyAccessError, _PayloadAccessLedger, _read_member

    unexpected = "fea_ml/data/runs_real/00000/occ.npz"
    archive_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(unexpected, b"malicious payload")
    ledger = _PayloadAccessLedger(["05134"])
    with zipfile.ZipFile(archive_path, "r") as archive:
        with pytest.raises(FitOnlyAccessError, match="outside selected fit members"):
            _read_member(archive, ledger, "00000", "occ.npz")

    assert ledger.members == (unexpected,)
    with pytest.raises(FitOnlyAccessError, match="outside selected fit members"):
        ledger.evidence()


@pytest.mark.parametrize("source", ("split", "archive"))
@pytest.mark.parametrize("kind", ("missing", "symlink", "fifo", "directory"))
def test_fit_probe_rejects_nonregular_or_missing_anchored_sources(
    tmp_path: Path, source: str, kind: str,
) -> None:
    from sasto.fit_probe import FitOnlyAccessError, run_fit_probe

    manifest = {
        "partitions": {
            "fit": {"sample_ids": ["05134"]},
            "development": {"sample_ids": ["00001"]},
            "calibration": {"sample_ids": ["00000"]},
            "confirmation": {"sample_ids": ["00002"]},
        }
    }
    split_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    split_manifest = tmp_path / "split.json"
    split_manifest.write_bytes(split_bytes)
    archive_path = tmp_path / "archive.zip"
    archive_path.write_bytes(b"placeholder archive")
    target = split_manifest if source == "split" else archive_path
    target.unlink()
    if kind == "symlink":
        external = tmp_path / "external"
        external.write_bytes(b"external")
        target.symlink_to(external)
    elif kind == "fifo":
        os.mkfifo(target)
    elif kind == "directory":
        target.mkdir()

    with pytest.raises(FitOnlyAccessError, match="regular file|symlink|missing|safely"):
        run_fit_probe(
            split_manifest=split_manifest,
            archive_path=archive_path,
            expected_split_manifest_sha256=hashlib.sha256(split_bytes).hexdigest(),
            expected_archive_sha256=hashlib.sha256(b"placeholder archive").hexdigest(),
            sample_ids=["05134"],
            limit=1,
            fixed_force=(0.0, 0.0, -100.0),
        )


@pytest.mark.parametrize("invalid", (None, "", "0" * 63, "A" * 64, "0" * 63 + "g"))
def test_fit_probe_rejects_malformed_external_anchors_before_source_access(invalid: object, tmp_path: Path) -> None:
    from sasto.fit_probe import FitOnlyAccessError, run_fit_probe

    with pytest.raises(FitOnlyAccessError, match="lowercase SHA-256"):
        run_fit_probe(
            split_manifest=Path(str(tmp_path / "split") + "\x00"),
            archive_path=Path(str(tmp_path / "archive") + "\x00"),
            expected_split_manifest_sha256=invalid,  # type: ignore[arg-type]
            expected_archive_sha256="0" * 64,
            sample_ids=None,
            limit=1,
            fixed_force=(0.0, 0.0, -100.0),
        )


def test_fit_probe_rejects_a_nul_source_path_cleanly(tmp_path: Path) -> None:
    from sasto.fit_probe import FitOnlyAccessError, run_fit_probe

    with pytest.raises(FitOnlyAccessError, match="non-traversal|NUL|safely"):
        run_fit_probe(
            split_manifest=Path(str(tmp_path / "split") + "\x00"),
            archive_path=tmp_path / "archive.zip",
            expected_split_manifest_sha256="0" * 64,
            expected_archive_sha256="0" * 64,
            sample_ids=None,
            limit=1,
            fixed_force=(0.0, 0.0, -100.0),
        )
