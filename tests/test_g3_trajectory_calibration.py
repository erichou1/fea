"""Focused K6-preparation contracts: frozen sampling, calibration, and sealing."""
from __future__ import annotations

import hashlib
import inspect
import math

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
