"""Executable V&V for the canonical T0 Hex8 verifier.

Every assertion is a scientific contract, not a diagnostic printout.
"""
from __future__ import annotations

import numpy as np
import pytest


def _beam(length: int = 4, width: int = 2, height: int = 2) -> np.ndarray:
    return np.ones((length, width, height), dtype=bool)


def _config(**overrides: object) -> object:
    from sasto.voxel_fea import VoxelFEAConfig

    values: dict[str, object] = {
        "voxel_size": (0.05, 0.05, 0.05),
        "youngs_modulus_pa": 25.0e9,
        "poisson_ratio": 0.20,
        "density_kg_m3": 2400.0,
        "gravity_m_s2": (0.0, 0.0, -9.81),
        "include_self_weight": False,
        "fixed_total_force_n": (0.0, 0.0, -100.0),
        "relative_tolerance": 1e-10,
        "maximum_iterations": 10_000,
    }
    values.update(overrides)
    return VoxelFEAConfig(**values)


def test_default_residual_bound_is_preregistered_at_two_e_minus_eight() -> None:
    from sasto.voxel_fea import VoxelFEAConfig

    assert VoxelFEAConfig().relative_tolerance == 2e-8


def test_hex8_element_is_symmetric_psd_and_has_six_rigid_modes() -> None:
    from sasto.voxel_fea import element_stiffness

    stiffness, _, _, _ = element_stiffness(25.0e9, 0.20, (0.01, 0.01, 0.01))
    assert stiffness.shape == (24, 24)
    assert np.allclose(stiffness, stiffness.T, rtol=0.0, atol=np.linalg.norm(stiffness) * 1e-12)
    eigenvalues = np.linalg.eigvalsh(stiffness)
    assert eigenvalues[0] >= -np.linalg.norm(stiffness) * 1e-11
    assert np.count_nonzero(np.abs(eigenvalues) <= np.linalg.norm(stiffness) * 1e-10) == 6

    coordinates = np.array(
        [[x, y, z] for x in (0.0, 0.01) for y in (0.0, 0.01) for z in (0.0, 0.01)], dtype=float
    )
    translations = [np.tile(axis, 8) for axis in np.eye(3)]
    rotations = [
        np.column_stack((np.zeros(8), -coordinates[:, 2], coordinates[:, 1])).ravel(),
        np.column_stack((coordinates[:, 2], np.zeros(8), -coordinates[:, 0])).ravel(),
        np.column_stack((-coordinates[:, 1], coordinates[:, 0], np.zeros(8))).ravel(),
    ]
    for mode in translations + rotations:
        assert np.linalg.norm(stiffness @ mode) <= np.linalg.norm(stiffness) * np.linalg.norm(mode) * 1e-10


def test_integrated_self_weight_equals_density_gravity_and_occupied_volume() -> None:
    from sasto.voxel_fea import assemble_voxel_system

    occupancy = _beam()
    config = _config(include_self_weight=True, fixed_total_force_n=(0.0, 0.0, 0.0))
    assembled = assemble_voxel_system(occupancy, config)
    expected = np.asarray(config.gravity_m_s2) * config.density_kg_m3 * occupancy.sum() * np.prod(config.voxel_size)
    assert np.allclose(assembled.body_force_sum_n, expected, rtol=1e-12, atol=1e-15)


def test_fixed_force_is_design_independent_and_exactly_preserved() -> None:
    from sasto.voxel_fea import solve_voxels

    force = (12.5, -7.0, -100.25)
    first = solve_voxels(_beam(4, 2, 2), _config(fixed_total_force_n=force))
    second = solve_voxels(_beam(6, 3, 2), _config(fixed_total_force_n=force))
    assert first["status"] == second["status"] == "success"
    assert first["fixed_force_sum_n"] == pytest.approx(force, rel=0.0, abs=1e-12)
    assert second["fixed_force_sum_n"] == pytest.approx(force, rel=0.0, abs=1e-12)
    assert first["loaded_node_count"] != second["loaded_node_count"]


def test_iterative_and_independent_direct_small_solves_agree() -> None:
    from sasto.voxel_fea import solve_voxels

    iterative = solve_voxels(_beam(), _config(include_displacement_field=True))
    direct = solve_voxels(_beam(), _config(solver_mode="direct", include_displacement_field=True))
    assert iterative["status"] == direct["status"] == "success"
    assert iterative["relative_residual"] <= 1e-10
    assert direct["relative_residual"] <= 1e-12
    assert np.allclose(iterative["displacement_field_m"], direct["displacement_field_m"], rtol=1e-7, atol=1e-14)
    assert iterative["max_von_mises_pa"] == pytest.approx(direct["max_von_mises_pa"], rel=1e-7)


def test_material_and_load_scaling_contracts() -> None:
    from sasto.voxel_fea import solve_voxels

    occupancy = _beam()
    external = solve_voxels(occupancy, _config())
    external_e2 = solve_voxels(occupancy, _config(youngs_modulus_pa=50.0e9))
    external_f2 = solve_voxels(occupancy, _config(fixed_total_force_n=(0.0, 0.0, -200.0)))
    self_weight = solve_voxels(occupancy, _config(include_self_weight=True, fixed_total_force_n=(0.0, 0.0, 0.0)))
    self_weight_e2 = solve_voxels(occupancy, _config(include_self_weight=True, fixed_total_force_n=(0.0, 0.0, 0.0), youngs_modulus_pa=50.0e9))
    self_weight_rho2 = solve_voxels(occupancy, _config(include_self_weight=True, fixed_total_force_n=(0.0, 0.0, 0.0), density_kg_m3=4800.0))
    for record in (external, external_e2, external_f2, self_weight, self_weight_e2, self_weight_rho2):
        assert record["status"] == "success", record
    assert external_e2["max_displacement_m"] / external["max_displacement_m"] == pytest.approx(0.5, rel=2e-6)
    assert external_e2["compliance_j"] / external["compliance_j"] == pytest.approx(0.5, rel=2e-6)
    assert external_f2["max_displacement_m"] / external["max_displacement_m"] == pytest.approx(2.0, rel=2e-6)
    assert external_f2["max_von_mises_pa"] / external["max_von_mises_pa"] == pytest.approx(2.0, rel=2e-6)
    assert external_f2["compliance_j"] / external["compliance_j"] == pytest.approx(4.0, rel=2e-6)
    assert self_weight_e2["max_displacement_m"] / self_weight["max_displacement_m"] == pytest.approx(0.5, rel=2e-6)
    assert self_weight_e2["compliance_j"] / self_weight["compliance_j"] == pytest.approx(0.5, rel=2e-6)
    assert self_weight_e2["max_von_mises_pa"] / self_weight["max_von_mises_pa"] == pytest.approx(1.0, rel=2e-6)
    assert self_weight_rho2["max_displacement_m"] / self_weight["max_displacement_m"] == pytest.approx(2.0, rel=2e-6)
    assert self_weight_rho2["max_von_mises_pa"] / self_weight["max_von_mises_pa"] == pytest.approx(2.0, rel=2e-6)
    assert self_weight_rho2["compliance_j"] / self_weight["compliance_j"] == pytest.approx(4.0, rel=2e-6)


def test_disconnected_and_unstable_load_geometry_fail_closed_without_mutation() -> None:
    from sasto.voxel_fea import solve_voxels

    occupancy = _beam()
    disconnected = occupancy.copy()
    disconnected[-1, -1, -1] = False
    disconnected = np.pad(disconnected, ((0, 2), (0, 0), (0, 0)))
    disconnected[-1, -1, -1] = True
    before = disconnected.copy()
    rejected = solve_voxels(disconnected, _config())
    assert rejected["status"] == "failure"
    assert rejected["reason"] == "disconnected_occupancy"
    assert np.array_equal(disconnected, before)
    unstable = solve_voxels(_beam(), _config(expected_loaded_node_count=1))
    assert unstable["status"] == "failure"
    assert unstable["reason"] == "unstable_load_node_set"
    same_count_different_nodes = solve_voxels(
        _beam(), _config(expected_loaded_node_coordinates=((4, 0, 0),) * 9)
    )
    assert same_count_different_nodes["status"] == "failure"
    assert same_count_different_nodes["reason"] == "unstable_load_node_set"


def test_nonconvergence_and_repeat_digest_fail_closed_and_remain_deterministic() -> None:
    from sasto.voxel_fea import solve_voxels

    rejected = solve_voxels(_beam(), _config(maximum_iterations=1, relative_tolerance=1e-16))
    assert rejected["status"] == "failure"
    assert rejected["reason"] in {"iterative_nonconvergence", "relative_residual_exceeds_tolerance"}
    assert rejected["relative_residual"] is None
    first = solve_voxels(_beam(), _config())
    second = solve_voxels(_beam(), _config())
    assert first["status"] == second["status"] == "success"
    assert first["scientific_digest"] == second["scientific_digest"]
    assert first["scientific_digest"] != first["timing"]["wall_seconds"]


def test_invalid_occupancy_and_malformed_configuration_are_rejected_as_records() -> None:
    from sasto.voxel_fea import solve_voxels

    invalid = solve_voxels(np.ones((3, 2, 2), dtype=np.uint8), _config())
    empty = solve_voxels(np.zeros((3, 2, 2), dtype=bool), _config())
    bad_config = solve_voxels(_beam(), _config(poisson_ratio=0.5))
    assert invalid["status"] == "failure" and invalid["reason"] == "occupancy_must_be_boolean"
    assert empty["status"] == "failure" and empty["reason"] == "empty_occupancy"
    assert bad_config["status"] == "failure" and bad_config["reason"] == "invalid_configuration"


def test_fixed_load_cantilever_refinement_moves_toward_beam_theory() -> None:
    from sasto.voxel_fea import solve_voxels

    force = 100.0
    coarse = solve_voxels(
        np.ones((4, 1, 1), dtype=bool),
        _config(voxel_size=(0.05, 0.05, 0.05), fixed_total_force_n=(0.0, 0.0, -force)),
    )
    fine = solve_voxels(
        np.ones((8, 2, 2), dtype=bool),
        _config(voxel_size=(0.025, 0.025, 0.025), fixed_total_force_n=(0.0, 0.0, -force)),
    )
    assert coarse["status"] == fine["status"] == "success"
    length, breadth, depth = 0.20, 0.05, 0.05
    inertia = breadth * depth**3 / 12.0
    analytical_tip_displacement = force * length**3 / (3.0 * 25.0e9 * inertia)
    coarse_relative_error = abs(coarse["max_displacement_m"] / analytical_tip_displacement - 1.0)
    fine_relative_error = abs(fine["max_displacement_m"] / analytical_tip_displacement - 1.0)
    assert fine_relative_error <= coarse_relative_error
    assert fine_relative_error <= 0.55
