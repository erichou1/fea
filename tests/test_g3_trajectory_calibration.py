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


# --------------------------------------------------------------------------
# Solver non-convergence is data, not an exception.
#
# Regression tests for the GB200 shard-7 loss (2026-08-31): one non-converging
# sample raised G3Error, aborted the shard, and cost 187 collateral cases that
# were never attempted.  G1b already treats the identical event as data.
# --------------------------------------------------------------------------


def _case_with_bins(bins, *, unsolved=(), family_id="fam-A"):
    """Build a trajectory case whose selected states match the frozen rule."""
    import sasto.g3_trajectory_calibration as g3

    batches = []
    selected_states = []
    unsolved_states = []
    fractions = {0: 0.03, 1: 0.07, 2: 0.12, 3: 0.17, 4: 0.22, 5: 0.30}
    for state_index, bin_index in enumerate(bins, start=1):
        batches.append({"batch_index": state_index, "proposed_material_reduction": fractions[bin_index],
                        "state_occupancy_sha256": "a" * 64})
    for bin_index in set(bins):
        candidates = [i for i, b in enumerate(bins, start=1) if b == bin_index]
        chosen = g3.select_state_index(family_id, bin_index, candidates)
        entry = {"state_index": chosen, "bin_index": bin_index, "bin_label": g3.DEPTH_BINS[bin_index],
                 "fraction_removed": fractions[bin_index], "state_occupancy_sha256": "a" * 64}
        if bin_index in unsolved:
            unsolved_states.append({**entry, "solver_status": "failure",
                                    "solver_reason": "iterative_nonconvergence"})
        else:
            selected_states.append({**entry, "solver": {"status": "success", "compliance_j": 1.0,
                                                        "max_displacement_m": 1.0, "max_gauss_von_mises_pa": 1.0},
                                    "prediction": {"mu": {}, "sigma": {}}})
    case = {"sample_id": "00001", "family_id": family_id, "role": "development",
            "trajectory": {"batches": batches}, "selected_states": selected_states,
            "intermediate_solver_call_count": 0}
    if unsolved_states or unsolved:
        case["unsolved_states"] = unsolved_states
    return case


def test_unsolved_bin_is_skipped_without_violating_the_sampling_rule() -> None:
    """A bin whose chosen state failed to solve is absent, and that is legal."""
    import sasto.g3_trajectory_calibration as g3

    case = _case_with_bins([1, 2, 5], unsolved={5})
    rows, _, selected = g3._selected_trajectory_rows([case])
    assert len(rows) == 2
    assert selected[">25%"] == 0
    assert selected["(5,10%]"] == 1


def test_case_without_unsolved_states_key_still_validates() -> None:
    """Records written before the fix must keep verifying unchanged."""
    import sasto.g3_trajectory_calibration as g3

    case = _case_with_bins([1, 2, 5])
    case.pop("unsolved_states", None)
    rows, _, _ = g3._selected_trajectory_rows([case])
    assert len(rows) == 3


def test_missing_bin_without_an_unsolved_record_is_still_rejected() -> None:
    """The relaxation must not become a hole: an unexplained gap must still fail."""
    import sasto.g3_trajectory_calibration as g3

    case = _case_with_bins([1, 2, 5], unsolved={5})
    case["unsolved_states"] = []
    # Caught by the frozen-sampling-rule guard, which fires first.
    with pytest.raises(g3.G3Error, match="violates the frozen sampling rule"):
        g3._selected_trajectory_rows([case])


def test_malformed_unsolved_record_is_rejected() -> None:
    import sasto.g3_trajectory_calibration as g3

    case = _case_with_bins([1, 2, 5], unsolved={5})
    case["unsolved_states"] = [{"bin_index": "five"}]
    with pytest.raises(g3.G3Error, match="unsolved state entry is malformed"):
        g3._selected_trajectory_rows([case])

    case2 = _case_with_bins([1, 2, 5], unsolved={5})
    case2["unsolved_states"] = "not-a-list"
    with pytest.raises(g3.G3Error, match="unsolved state record is malformed"):
        g3._selected_trajectory_rows([case2])


def test_preconditioner_unavailable_still_fails_closed() -> None:
    """An environment fault affects every later solve and must not be recorded as data."""
    import inspect
    import sasto.g3_trajectory_calibration as g3

    source = inspect.getsource(g3)
    assert 'if reason == "preconditioner_unavailable":' in source
    assert 'raise G3Error("selected trajectory solver preconditioner is unavailable")' in source


def test_non_convergence_no_longer_raises_in_the_generation_path() -> None:
    """The exact line that aborted GB200 shard 7 must be gone."""
    import inspect
    import sasto.g3_trajectory_calibration as g3

    source = inspect.getsource(g3)
    assert "selected trajectory canonical solver response is unavailable" not in source


def test_trajectory_channels_match_g2_training_representation() -> None:
    """G3-D1: the surrogate was trained on RAW part labels (surrogate.py:225,:250).

    G3 must feed it the same representation at every trajectory state. Masking
    parts by current occupancy produces an input distribution the ensemble never
    saw, shifting predictions by ~0.35 sigma on baseline states alone.
    """
    import numpy as np
    from sasto.g3_trajectory_calibration import _channels

    rng = np.random.default_rng(0)
    occupancy = rng.random((64, 64, 64)) < 0.3
    # Parts are nonzero OUTSIDE occupancy, exactly as in the real archive.
    parts = rng.integers(1, 6, size=(64, 64, 64), dtype=np.uint8)

    channels = _channels(occupancy, parts)

    assert channels.shape == (2, 64, 64, 64)
    np.testing.assert_array_equal(channels[0], occupancy.astype(np.float32))
    # The parts channel must be the RAW labels, not masked by occupancy.
    np.testing.assert_array_equal(channels[1], parts.astype(np.float32))
    # And specifically: parts must survive where occupancy is zero.
    outside = ~occupancy
    assert outside.any()
    assert np.all(channels[1][outside] == parts[outside])
