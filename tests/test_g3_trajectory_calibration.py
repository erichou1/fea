"""Focused K6-preparation contracts: frozen sampling, calibration, and sealing."""
from __future__ import annotations

import hashlib
import inspect
import math
from pathlib import Path

import pytest


def test_family_seed_is_cryptographic_and_stable() -> None:
    from sasto.g3_trajectory_calibration import FAMILY_SEED_NAMESPACE, family_seed

    expected = int.from_bytes(
        hashlib.sha256((FAMILY_SEED_NAMESPACE + "\0" + "20260828" + "\0" + "family-001").encode("utf-8")).digest()[:8],
        "big",
    ) % (2 ** 31 - 1)
    assert family_seed("family-001") == expected
    assert family_seed("family-001") == expected
    assert family_seed("family-002") != expected


def test_depth_bins_follow_the_frozen_open_closed_boundaries() -> None:
    from sasto.g3_trajectory_calibration import depth_bin_index

    assert depth_bin_index(0.0) is None
    assert depth_bin_index(0.05) == 0
    assert depth_bin_index(0.050000001) == 1
    assert depth_bin_index(0.10) == 1
    assert depth_bin_index(0.15) == 2
    assert depth_bin_index(0.20) == 3
    assert depth_bin_index(0.25) == 4
    assert depth_bin_index(0.250000001) == 5


def test_sampling_rule_accepts_only_identifiers_and_ignores_response_values() -> None:
    from sasto.g3_trajectory_calibration import select_state_index

    signature = inspect.signature(select_state_index)
    assert tuple(signature.parameters) == ("family_id", "bin_index", "state_indices")
    candidates = [3, 7, 11, 19]
    selected = select_state_index("family-001", 2, candidates)
    # The pre-registration hash is over exactly these identifier components.
    expected = min(
        candidates,
        key=lambda state_index: hashlib.sha256(
            ("sasto-v-k6-sampling-v1" + "\0" + "family-001" + "\0" + "2" + "\0" + str(state_index)).encode("utf-8")
        ).hexdigest(),
    )
    assert selected == expected
    # Deliberately extreme response/surrogate changes cannot enter the API.
    responses_a = {state: {"Y": -1e300, "mu": 1e300, "sigma": 0.0} for state in candidates}
    responses_b = {state: {"Y": 1e300, "mu": -1e300, "sigma": 1e300} for state in candidates}
    assert responses_a != responses_b
    assert select_state_index("family-001", 2, list(responses_a)) == select_state_index("family-001", 2, list(responses_b)) == selected


def test_split_conformal_quantile_uses_the_conservative_order_statistic() -> None:
    from sasto.g3_trajectory_calibration import split_conformal_quantile

    scores = list(range(59))
    # alpha = 1/60 makes ceil((n + 1) * (1 - alpha)) = 59.
    assert split_conformal_quantile(scores, alpha=1.0 / 60.0) == 58.0
    assert math.isinf(split_conformal_quantile([0.0], alpha=0.05))


def test_confirmation_is_unconditionally_sealed_before_paths_are_inspected() -> None:
    from sasto.g3_trajectory_calibration import G3RoleError, open_g3_role

    with pytest.raises(G3RoleError, match="confirmation.*sealed"):
        open_g3_role(
            role="confirmation",
            split_manifest=__import__("pathlib").Path("/definitely/not/a/split"),
            expected_split_sha256="0" * 64,
            archive=__import__("pathlib").Path("/definitely/not/an/archive"),
            expected_archive_sha256="0" * 64,
            g1b_root=__import__("pathlib").Path("/definitely/not/a/cohort"),
            expected_cohort_manifest_sha256="0" * 64,
            expected_cluster_role_manifest_sha256="0" * 64,
        )


def test_family_map_preserves_eligible_subset_membership() -> None:
    """G1b exclusions may remove samples without changing their frozen families."""
    from sasto.g3_trajectory_calibration import _family_map

    sample_to_family = [{"sample_id": "{:02d}".format(index), "family_id": "family-{:02d}".format(index)} for index in range(10)]
    split = {"schema_version": "1.0.0", "seed": 42, "algorithm": "family-id-v1", "fractions": {"fit": 0.6, "development": 0.2, "calibration": 0.1, "confirmation": 0.1}, "sample_to_family": sample_to_family, "partitions": {
        "fit": {"sample_ids": ["00", "01", "02", "03", "04", "05"], "family_ids": ["family-00", "family-01", "family-02", "family-03", "family-04", "family-05"]},
        "development": {"sample_ids": ["06", "07"], "family_ids": ["family-06", "family-07"]},
        "calibration": {"sample_ids": ["08"], "family_ids": ["family-08"]},
        "confirmation": {"sample_ids": ["09"], "family_ids": ["family-09"]},
    }}
    assert _family_map(split, "development", ["06"]) == {"06": "family-06"}


def test_run_shard_loads_verified_baseline_rows_without_recomputation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Shard workers must use initialize's evidence, never perform a baseline pass."""
    from types import SimpleNamespace
    import sasto.g3_trajectory_calibration as g3

    from sasto.g3_trajectory_calibration import BASELINE_ROWS_FILENAME

    seen: list[str] = []
    (tmp_path / BASELINE_ROWS_FILENAME).touch()

    def verified(path: __import__("pathlib").Path, _label: str, _digest_field: str) -> dict[str, object]:
        seen.append(path.name)
        return {"roles": {"development": {"sample_count": 0, "rows": []}, "calibration": {"sample_count": 0, "rows": []}}}

    monkeypatch.setattr(g3, "_verified_json", verified)
    monkeypatch.setattr(g3, "EnsemblePredictor", lambda **_kwargs: object())
    empty_dataset = type("EmptyDataset", (), {"sample_ids": (), "__len__": lambda self: 0})()
    monkeypatch.setattr(g3, "open_g3_role", lambda **kwargs: SimpleNamespace(role=kwargs["role"], dataset=empty_dataset, family_by_sample={}))
    monkeypatch.setattr(g3, "_load_or_generate_trajectories", lambda **kwargs: ([], {"role": kwargs["role"], "generated_count": 0}))
    monkeypatch.setattr(g3, "_open_or_build_channel_cache", lambda **_kwargs: object())
    monkeypatch.setattr(g3, "_baseline_rows", lambda *_args: (_ for _ in ()).throw(AssertionError("baseline recomputation")))

    result = g3.run_precoverage_shard(
        output_root=tmp_path, split_manifest=tmp_path / "split.json", expected_split_sha256="0" * 64,
        archive=tmp_path / "archive.zip", expected_archive_sha256="1" * 64, g1b_root=tmp_path / "g1b",
        expected_cohort_manifest_sha256="2" * 64, expected_cluster_role_manifest_sha256="3" * 64,
        ensemble_root=tmp_path / "ensemble", normalization_path=tmp_path / "normalization.json", shard_index=0,
        shard_count=4, device="cpu",
    )

    assert "baseline-rows.json" in seen
    assert result["development_case_count"] == 0
    assert result["calibration_case_count"] == 0


def test_run_shard_reuses_frozen_quantiles_when_baseline_rows_are_absent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A stopped predecessor may have q_base but no optional baseline-row receipt."""
    from types import SimpleNamespace
    import sasto.g3_trajectory_calibration as g3

    monkeypatch.setattr(g3, "_verified_json", lambda *_args: {"roles": {"development": {"sample_count": 0, "rows": []}, "calibration": {"sample_count": 0, "rows": []}}})
    monkeypatch.setattr(g3, "EnsemblePredictor", lambda **_kwargs: object())
    empty_dataset = type("EmptyDataset", (), {"sample_ids": (), "__len__": lambda self: 0})()
    monkeypatch.setattr(g3, "open_g3_role", lambda **kwargs: SimpleNamespace(role=kwargs["role"], dataset=empty_dataset, family_by_sample={}, provenance={}))
    monkeypatch.setattr(g3, "_open_or_build_channel_cache", lambda **_kwargs: object())
    monkeypatch.setattr(g3, "_baseline_rows", lambda *_args: (_ for _ in ()).throw(AssertionError("baseline recomputation")))
    monkeypatch.setattr(g3, "_load_or_generate_trajectories", lambda **kwargs: ([], {"role": kwargs["role"], "generated_count": 0}))

    result = g3.run_precoverage_shard(
        output_root=tmp_path, split_manifest=tmp_path / "split.json", expected_split_sha256="0" * 64,
        archive=tmp_path / "archive.zip", expected_archive_sha256="1" * 64, g1b_root=tmp_path / "g1b",
        expected_cohort_manifest_sha256="2" * 64, expected_cluster_role_manifest_sha256="3" * 64,
        ensemble_root=tmp_path / "ensemble", normalization_path=tmp_path / "normalization.json", shard_index=0,
        shard_count=4, device="cpu",
    )
    assert result["development_case_count"] == 0
    assert result["calibration_case_count"] == 0


def test_run_shard_passes_verified_channel_cache_to_each_role(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Trajectory workers must use initialize's decoded channels, never payload reads."""
    from types import SimpleNamespace
    import sasto.g3_trajectory_calibration as g3

    cache = object()
    seen: list[object] = []
    monkeypatch.setattr(g3, "_verified_json", lambda *_args: {"roles": {"development": {"sample_count": 0, "rows": []}, "calibration": {"sample_count": 0, "rows": []}}})
    monkeypatch.setattr(g3, "EnsemblePredictor", lambda **_kwargs: object())
    empty_dataset = type("EmptyDataset", (), {"sample_ids": (), "__len__": lambda self: 0})()
    monkeypatch.setattr(g3, "open_g3_role", lambda **kwargs: SimpleNamespace(role=kwargs["role"], dataset=empty_dataset, family_by_sample={}, provenance={}))
    monkeypatch.setattr(g3, "_open_or_build_channel_cache", lambda **_kwargs: cache)

    def load_or_generate(**kwargs: object):
        seen.append(kwargs["channel_cache"])
        return [], {"role": kwargs["role"], "generated_count": 0}

    monkeypatch.setattr(g3, "_load_or_generate_trajectories", load_or_generate)
    g3.run_precoverage_shard(
        output_root=tmp_path, split_manifest=tmp_path / "split.json", expected_split_sha256="0" * 64,
        archive=tmp_path / "archive.zip", expected_archive_sha256="1" * 64, g1b_root=tmp_path / "g1b",
        expected_cohort_manifest_sha256="2" * 64, expected_cluster_role_manifest_sha256="3" * 64,
        ensemble_root=tmp_path / "ensemble", normalization_path=tmp_path / "normalization.json", shard_index=0,
        shard_count=4, device="cpu",
    )
    assert seen == [cache, cache]
