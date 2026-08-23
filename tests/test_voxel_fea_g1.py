"""Executable V&V for the canonical T0 Hex8 verifier.

Every assertion is a scientific contract, not a diagnostic printout.
"""
from __future__ import annotations

import json

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


def test_fixed_force_is_assembled_only_on_physical_maximum_x_face() -> None:
    from sasto.voxel_fea import assemble_voxel_system

    assembled = assemble_voxel_system(_beam(4, 2, 2), _config())
    nodal_force = assembled.force.reshape(-1, 3)
    nonzero_nodes = np.flatnonzero(np.any(nodal_force != 0.0, axis=1))
    nonzero_coordinates = assembled.node_coordinates[nonzero_nodes]
    loaded_x_values = sorted(set(nonzero_coordinates[:, 0].tolist()))
    assert loaded_x_values == [4]
    assert loaded_x_values != [3]
    assert not np.any(nonzero_coordinates[:, 0] < 4)
    assert {tuple(point) for point in nonzero_coordinates} == {
        (4, y, z) for y in range(3) for z in range(3)
    }


def test_sparse_asymmetric_geometry_locks_exact_maximum_face_coordinates_even_when_count_matches_old_face() -> None:
    from sasto.voxel_fea import VoxelFEAConfig, assemble_voxel_system, solve_voxels

    occupancy = np.zeros((5, 3, 1), dtype=bool)
    occupancy[(0, 0, 0)] = True
    occupancy[(1, 0, 0)] = True
    occupancy[(1, 1, 0)] = True
    occupancy[(2, 1, 0)] = True
    occupancy[(3, 1, 0)] = True
    occupancy[(4, 1, 0)] = True
    expected = tuple((5, y, z) for y in (1, 2) for z in (0, 1))
    config = _config(expected_loaded_node_count=4, expected_loaded_node_coordinates=expected)
    assembled = assemble_voxel_system(occupancy, config)
    nodal_force = assembled.force.reshape(-1, 3)
    force_coordinates = {
        tuple(point) for point in assembled.node_coordinates[np.flatnonzero(np.any(nodal_force != 0.0, axis=1))]
    }
    assert force_coordinates == set(expected)
    assert len(force_coordinates) == 4
    stale_face_coordinates = tuple((4, y, z) for y in (1, 2) for z in (0, 1))
    stale_guard = VoxelFEAConfig(**{**config.__dict__, "expected_loaded_node_coordinates": stale_face_coordinates})
    rejected = solve_voxels(occupancy, stale_guard)
    assert rejected["status"] == "failure"
    assert rejected["reason"] == "unstable_load_node_set"


def _reference_gauss_von_mises(element_displacements: np.ndarray, voxel_size: tuple[float, float, float], elasticity: np.ndarray) -> np.ndarray:
    """Independent Hex8 stress evaluator used only to verify postprocessing."""
    dx, dy, dz = voxel_size
    gauss = 1.0 / np.sqrt(3.0)
    values = []
    for xi in (-gauss, gauss):
        for eta in (-gauss, gauss):
            for zeta in (-gauss, gauss):
                derivatives_reference = np.array([
                    [-(1 - eta) * (1 - zeta), -(1 - eta) * (1 + zeta), -(1 + eta) * (1 - zeta), -(1 + eta) * (1 + zeta),
                     (1 - eta) * (1 - zeta), (1 - eta) * (1 + zeta), (1 + eta) * (1 - zeta), (1 + eta) * (1 + zeta)],
                    [-(1 - xi) * (1 - zeta), -(1 - xi) * (1 + zeta), (1 - xi) * (1 - zeta), (1 - xi) * (1 + zeta),
                     -(1 + xi) * (1 - zeta), -(1 + xi) * (1 + zeta), (1 + xi) * (1 - zeta), (1 + xi) * (1 + zeta)],
                    [-(1 - xi) * (1 - eta), (1 - xi) * (1 - eta), -(1 - xi) * (1 + eta), (1 - xi) * (1 + eta),
                     -(1 + xi) * (1 - eta), (1 + xi) * (1 - eta), -(1 + xi) * (1 + eta), (1 + xi) * (1 + eta)],
                ], dtype=float) / 8.0
                derivatives = np.array((2.0 / dx, 2.0 / dy, 2.0 / dz))[:, None] * derivatives_reference
                b_matrix = np.zeros((6, 24), dtype=float)
                for node, (x, y, z) in enumerate(derivatives.T):
                    base = 3 * node
                    b_matrix[0, base] = x
                    b_matrix[1, base + 1] = y
                    b_matrix[2, base + 2] = z
                    b_matrix[3, base + 1] = z
                    b_matrix[3, base + 2] = y
                    b_matrix[4, base] = z
                    b_matrix[4, base + 2] = x
                    b_matrix[5, base] = y
                    b_matrix[5, base + 1] = x
                stresses = elasticity @ (b_matrix @ element_displacements)
                sxx, syy, szz, syz, sxz, sxy = stresses
                values.append(np.sqrt(max(0.0, 0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                                          + 3.0 * (sxy ** 2 + syz ** 2 + sxz ** 2))))
    return np.asarray(values)


def test_declared_full_gauss_stress_outputs_match_independent_nonuniform_fixture_evaluation() -> None:
    from sasto.voxel_fea import assemble_voxel_system, solve_voxels

    occupancy = np.ones((3, 2, 2), dtype=bool)
    occupancy[0, 1, 1] = False
    config = _config(solver_mode="direct", include_displacement_field=True)
    record = solve_voxels(occupancy, config)
    assert record["status"] == "success"
    assembled = assemble_voxel_system(occupancy, config)
    displacement = np.asarray(record["displacement_field_m"], dtype=float).reshape(-1, 3).ravel()
    all_gauss_values = np.concatenate([
        _reference_gauss_von_mises(displacement[element_dofs], config.voxel_size, assembled.elasticity)
        for element_dofs in assembled.element_dofs
    ])
    assert record["stress_sampling"] == "eight_full_integration_gauss_points_per_element"
    assert record["stress_sample_count"] == 8 * record["element_count"] == len(all_gauss_values)
    assert record["max_gauss_von_mises_pa"] == pytest.approx(float(np.max(all_gauss_values)), rel=1e-11)
    assert record["p99_gauss_von_mises_pa"] == pytest.approx(float(np.percentile(all_gauss_values, 99.0)), rel=1e-11)
    assert record["max_von_mises_pa"] == record["max_gauss_von_mises_pa"]
    assert record["p99_von_mises_pa"] == record["p99_gauss_von_mises_pa"]


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
    assert same_count_different_nodes["reason"] == "invalid_configuration"


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


@pytest.mark.parametrize(
    "overrides",
    (
        {"voxel_size": (0.05, 0.05)},
        {"voxel_size": [0.05, 0.05, 0.05]},
        {"voxel_size": np.asarray((0.05, 0.05, 0.05))},
        {"voxel_size": (0.05, float("inf"), 0.05)},
        {"gravity_m_s2": (0.0, 0.0)},
        {"gravity_m_s2": (0.0, 0.0, float("nan"))},
        {"fixed_total_force_n": (0.0, -100.0)},
        {"fixed_total_force_n": (0.0, 0.0, float("inf"))},
        {"youngs_modulus_pa": float("nan")},
        {"poisson_ratio": float("inf")},
        {"density_kg_m3": float("nan")},
        {"relative_tolerance": float("inf")},
        {"include_self_weight": 1},
        {"include_displacement_field": 0},
        {"maximum_iterations": True},
        {"direct_max_dof": True},
        {"expected_loaded_node_count": True},
        {"expected_loaded_node_coordinates": [(4, 0, 0)]},
        {"expected_loaded_node_coordinates": ((4, 0, 0), (4, 0, 0))},
        {"expected_loaded_node_coordinates": ((-1, 0, 0),)},
        {"expected_loaded_node_count": 2, "expected_loaded_node_coordinates": ((4, 0, 0),)},
    ),
)
def test_every_configuration_field_rejects_malformed_values_as_canonical_failure_records(overrides: dict[str, object]) -> None:
    from sasto.voxel_fea import solve_voxels

    record = solve_voxels(_beam(), _config(**overrides))
    assert record["status"] == "failure"
    assert record["reason"] == "invalid_configuration"
    json.dumps(record, sort_keys=True, allow_nan=False)


def test_failure_helper_is_safe_for_malformed_config_and_occupancy() -> None:
    from sasto.voxel_fea import VoxelFEAConfig, _failure

    malformed = VoxelFEAConfig(youngs_modulus_pa=float("nan"), gravity_m_s2=(0.0, 0.0))
    record = _failure("invalid_configuration", malformed, object(), relative_residual=float("nan"))
    assert record == {
        "status": "failure", "reason": "invalid_configuration", "config_digest": None,
        "input_digest": None, "relative_residual": None, "outputs": None,
    }
    json.dumps(record, sort_keys=True, allow_nan=False)


def test_iterative_target_is_one_tenth_of_frozen_admission_bound_and_direct_has_no_target(monkeypatch: pytest.MonkeyPatch) -> None:
    from scipy.sparse import linalg as sparse_linalg
    from sasto import voxel_fea

    requested: list[float] = []

    def exact_cg(matrix: object, force: object, **kwargs: object) -> tuple[np.ndarray, int]:
        requested.append(float(kwargs["rtol"]))
        return sparse_linalg.spsolve(matrix, force), 0

    monkeypatch.setattr(voxel_fea.sparse_linalg, "cg", exact_cg)
    iterative = voxel_fea.solve_voxels(_beam(), _config(relative_tolerance=2e-8))
    direct = voxel_fea.solve_voxels(_beam(), _config(relative_tolerance=2e-8, solver_mode="direct"))
    assert requested == [2e-9]
    assert iterative["status"] == direct["status"] == "success"
    assert iterative["iterative_requested_rtol"] == 2e-9
    assert direct["iterative_requested_rtol"] is None
    assert iterative["relative_residual"] <= 2e-8


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
    # Corrected maximum-face traction gives a 6.60% refined-beam error; freeze
    # a 7% allowance rather than retaining the stale 55% interior-face bound.
    assert fine_relative_error <= 0.07
