"""G1a production topology gate: exact oracle and conservative local subset."""
from __future__ import annotations

import hashlib
from itertools import product

import numpy as np
import pytest

from sasto.topology import (
    apply_conservative_deletions_sequentially,
    conservative_local_6_26,
    exact_global_6_26,
    exact_topology_preflight_6_26,
    is_simple_point_6_26,
    make_background_remote_pair,
    make_foreground_remote_pair,
    topology_artifact_record,
)


def _prior_cavity() -> np.ndarray:
    volume = np.ones((3, 3, 3), dtype=bool)
    volume[1, 1, 1] = False
    return volume


def _prior_isolated() -> np.ndarray:
    return np.array(
        [[[False, False], [False, True]], [[True, True], [False, False]]], dtype=bool
    )


@pytest.mark.parametrize("maker", (make_foreground_remote_pair, make_background_remote_pair))
def test_remote_pairs_are_exactly_distinguished_but_conservatively_rejected(maker: object) -> None:
    joined, split, point = maker()  # type: ignore[operator]
    assert np.array_equal(joined[2:5, 2:5, 2:5], split[2:5, 2:5, 2:5])
    assert exact_global_6_26(joined, point) is True
    assert exact_global_6_26(split, point) is False
    assert conservative_local_6_26(joined, point) is False
    assert conservative_local_6_26(split, point) is False


@pytest.mark.parametrize(
    ("volume", "point"),
    ((_prior_cavity(), (0, 0, 0)), (_prior_isolated(), (0, 1, 1))),
)
def test_historical_cavity_and_isolated_witnesses_are_rejected(volume: np.ndarray, point: tuple[int, int, int]) -> None:
    assert exact_global_6_26(volume, point) is False
    assert conservative_local_6_26(volume, point) is False


def test_exact_named_oracle_keeps_backward_compatible_exact_contract() -> None:
    volume = np.zeros((4, 4, 4), dtype=bool)
    volume[1, 1, 1] = volume[1, 1, 2] = True
    assert exact_global_6_26(volume, (1, 1, 2)) is True
    assert is_simple_point_6_26(volume, (1, 1, 2)) is True


def test_conservative_has_no_false_accepts_on_all_2cubed_volumes() -> None:
    for bits in range(1 << 8):
        volume = np.array([(bits >> i) & 1 for i in range(8)], dtype=bool).reshape(2, 2, 2)
        for point in product(range(2), repeat=3):
            assert not (conservative_local_6_26(volume, point) and not exact_global_6_26(volume, point))


def test_numpy_list_parity_and_input_immutability() -> None:
    volume = np.zeros((4, 4, 4), dtype=bool)
    volume[1, 1, 1] = volume[1, 1, 2] = True
    before = volume.copy()
    nested = volume.tolist()
    nested_before = [[row[:] for row in plane] for plane in nested]
    point = (1, 1, 2)
    assert exact_global_6_26(volume, point) == exact_global_6_26(nested, point)
    assert conservative_local_6_26(volume, point) == conservative_local_6_26(nested, point)
    assert np.array_equal(volume, before)
    assert nested == nested_before


@pytest.mark.parametrize(
    "volume",
    (
        np.zeros((2, 2, 2), dtype=np.uint8),
        np.zeros((2, 2), dtype=bool),
        [[[True], [1]]],
        [[[True], []]],
    ),
)
def test_new_public_predicates_fail_closed_on_malformed_or_nonbool_inputs(volume: object) -> None:
    assert exact_global_6_26(volume, (0, 0, 0)) is False
    assert conservative_local_6_26(volume, (0, 0, 0)) is False
    with pytest.raises(ValueError):
        exact_topology_preflight_6_26(volume)


def test_preflight_records_explicit_counts_cavity_shape_occupied_and_canonical_digest() -> None:
    volume = _prior_cavity()
    before = volume.copy()
    result = exact_topology_preflight_6_26(volume)
    assert result.foreground_6_components == 1
    assert result.background_26_components_with_exterior == 2
    assert result.has_cavities is True
    assert result.shape == (3, 3, 3)
    assert result.occupied_count == 26
    assert result.boundary_semantics == "explicit_exterior_node_connected_to_boundary_background"
    assert result.input_sha256 == hashlib.sha256(b"sasto-topology-6-26-v1\x003,3,3\x00" + np.packbits(volume.reshape(-1), bitorder="little").tobytes()).hexdigest()
    assert np.array_equal(volume, before)


def test_preflight_reports_disconnected_without_silent_rejection() -> None:
    volume = np.zeros((3, 3, 3), dtype=bool)
    volume[0, 0, 0] = volume[2, 2, 2] = True
    result = exact_topology_preflight_6_26(volume)
    assert result.foreground_6_components == 2
    assert result.background_26_components_with_exterior == 1
    assert result.has_cavities is False


def test_sequential_helper_rechecks_after_each_accepted_deletion_and_obeys_masks() -> None:
    volume = np.zeros((5, 5, 5), dtype=bool)
    volume[2, 2, 1:4] = True
    points = [(2, 2, 1), (2, 2, 3)]
    result = apply_conservative_deletions_sequentially(volume, points)
    assert result.accepted_points == tuple(points)
    assert result.rejected_points == ()
    assert result.sequential_recheck is True
    assert result.volume[2, 2, :].tolist() == [False, False, True, False, False]
    assert volume[2, 2, :].tolist() == [False, True, True, True, False]
    protected = np.zeros_like(volume); protected[2, 2, 1] = True
    edit = np.ones_like(volume); edit[2, 2, 3] = False
    masked = apply_conservative_deletions_sequentially(volume, points, protected_mask=protected, edit_mask=edit)
    assert masked.accepted_points == ()
    assert masked.rejected_points == tuple(points)


def test_topology_artifact_record_declares_conservative_mode_preflight_campaign_and_rechecks() -> None:
    record = topology_artifact_record(
        exact_topology_preflight_6_26(np.zeros((2, 2, 2), dtype=bool)),
        campaign_hash="a" * 64,
        sequential_recheck=True,
    )
    assert record["topology_mode"] == "conservative_local_6_26"
    assert record["exact_preflight"]["foreground_6_components"] == 0
    assert record["exact_preflight"]["input_sha256"]
    assert record["campaign_hash"] == "a" * 64
    assert record["sequential_recheck"] is True


def test_conservative_campaign_reports_separate_false_accepts_rejects_hash_and_recall() -> None:
    from sasto.topology_campaign import run_campaign

    result = run_campaign(neighborhoods=2_000, data_root=None)
    assert result["campaign_hash"]
    assert result["random_local_neighborhoods"]["cases"] == 2_000
    assert result["false_accepts"] == 0
    assert result["exact_only_false_rejects"] >= 0
    assert 0.0 <= result["recall"] <= 1.0
    assert result["sequential_recheck"] is True
    assert result["sequential_batch"]["accepted"] == 2
    assert result["sequential_batch"]["false_accepts"] == 0
