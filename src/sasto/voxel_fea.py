"""Canonical T0 regular-Hex8 linear-elastic voxel verifier.

Array occupancy uses ``occupancy[a0, a1, a2]`` mapped directly to physical
``(x, y, z)``.  Node displacement degrees of freedom are contiguous physical
``(x, y, z)`` triples.  This module intentionally has no executable legacy
imports: the element integration and sparse assembly are owned here.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg


@dataclass(frozen=True)
class VoxelFEAConfig:
    """Frozen T0 contract; SI values and physical coordinates are explicit."""

    voxel_size: tuple[float, float, float] = (0.01, 0.01, 0.01)
    youngs_modulus_pa: float = 25.0e9
    poisson_ratio: float = 0.20
    density_kg_m3: float = 2400.0
    gravity_m_s2: tuple[float, float, float] = (0.0, 0.0, -9.81)
    include_self_weight: bool = True
    fixed_total_force_n: tuple[float, float, float] = (0.0, 0.0, 0.0)
    relative_tolerance: float = 2e-8
    maximum_iterations: int = 50_000
    expected_loaded_node_count: int | None = None
    expected_loaded_node_coordinates: tuple[tuple[int, int, int], ...] | None = None
    solver_mode: str = "iterative"
    direct_max_dof: int = 2_000
    include_displacement_field: bool = False


@dataclass(frozen=True)
class AssembledVoxelSystem:
    stiffness: sparse.csr_matrix
    force: np.ndarray
    body_force_sum_n: np.ndarray
    fixed_force_sum_n: np.ndarray
    element_dofs: np.ndarray
    fixed_dofs: np.ndarray
    free_dofs: np.ndarray
    node_coordinates: np.ndarray
    b_gauss: np.ndarray
    elasticity: np.ndarray
    n_elements: int
    n_nodes: int
    loaded_node_count: int
    fixed_node_count: int


def _canonical_json_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalise_scientific(value: object) -> object:
    """Retain 11 significant digits, above deterministic solver roundoff."""
    if isinstance(value, float):
        return float(format(value, ".11g"))
    if isinstance(value, list):
        return [_normalise_scientific(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalise_scientific(item) for key, item in value.items()}
    return value


def _scientific_digest(value: object) -> str:
    """Hash declared state, omitting measured iterative-residual noise and timing."""
    if not isinstance(value, dict):
        raise TypeError("scientific digest requires a record")
    stable = {key: item for key, item in value.items() if key not in {"relative_residual", "timing"}}
    return _canonical_json_digest(_normalise_scientific(stable))


def _config_payload(config: VoxelFEAConfig) -> dict[str, object]:
    return dataclasses.asdict(config)


def _safe_config_digest(config: object) -> str | None:
    """Return a config digest only when its exact value has canonical JSON."""
    if not isinstance(config, VoxelFEAConfig):
        return None
    try:
        return _canonical_json_digest(_config_payload(config))
    except Exception:
        return None


def _input_digest(occupancy: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(occupancy.shape).encode("ascii"))
    digest.update(occupancy.dtype.str.encode("ascii"))
    digest.update(np.ascontiguousarray(occupancy).tobytes())
    return digest.hexdigest()


def _safe_input_digest(occupancy: object) -> str | None:
    try:
        if isinstance(occupancy, np.ndarray) and occupancy.dtype == np.bool_:
            return _input_digest(occupancy)
    except Exception:
        pass
    return None


def _is_finite_real(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and bool(np.isfinite(value))


def _is_finite_vector(value: object) -> bool:
    return isinstance(value, tuple) and len(value) == 3 and all(_is_finite_real(item) for item in value)


def _validate_config(config: object) -> VoxelFEAConfig:
    if not isinstance(config, VoxelFEAConfig):
        raise ValueError("invalid_configuration")
    expected_coordinates = config.expected_loaded_node_coordinates
    expected_coordinates_valid = (
        expected_coordinates is None
        or (isinstance(expected_coordinates, tuple) and all(
            isinstance(point, tuple) and len(point) == 3
            and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in point)
            for point in expected_coordinates
        ) and len(set(expected_coordinates)) == len(expected_coordinates))
    )
    if (not all(_is_finite_real(value) for value in (
                config.youngs_modulus_pa, config.poisson_ratio, config.density_kg_m3, config.relative_tolerance))
            or not _is_finite_vector(config.voxel_size) or not _is_finite_vector(config.gravity_m_s2)
            or not _is_finite_vector(config.fixed_total_force_n)
            or config.youngs_modulus_pa <= 0.0 or config.density_kg_m3 <= 0.0
            or not (-1.0 < config.poisson_ratio < 0.5) or min(config.voxel_size) <= 0.0
            or config.relative_tolerance <= 0.0
            or not isinstance(config.include_self_weight, bool) or not isinstance(config.include_displacement_field, bool)
            or not isinstance(config.maximum_iterations, int) or isinstance(config.maximum_iterations, bool) or config.maximum_iterations < 1
            or not isinstance(config.solver_mode, str) or config.solver_mode not in {"iterative", "direct"}
            or not isinstance(config.direct_max_dof, int) or isinstance(config.direct_max_dof, bool) or config.direct_max_dof < 1
            or (config.expected_loaded_node_count is not None and (
                not isinstance(config.expected_loaded_node_count, int) or isinstance(config.expected_loaded_node_count, bool)
                or config.expected_loaded_node_count < 1))
            or not expected_coordinates_valid):
        raise ValueError("invalid_configuration")
    if (config.expected_loaded_node_count is not None and expected_coordinates is not None
            and config.expected_loaded_node_count != len(expected_coordinates)):
        raise ValueError("invalid_configuration")
    return config


def _validate_occupancy(occupancy: object) -> np.ndarray:
    if not isinstance(occupancy, np.ndarray) or occupancy.dtype != np.bool_:
        raise ValueError("occupancy_must_be_boolean")
    if occupancy.ndim != 3:
        raise ValueError("occupancy_must_be_three_dimensional")
    if not occupancy.any():
        raise ValueError("empty_occupancy")
    if int(occupancy.sum()) < 2:
        raise ValueError("occupancy_too_small")
    return occupancy


def _is_face_connected(occupancy: np.ndarray) -> bool:
    start = tuple(np.argwhere(occupancy)[0])
    seen = {start}
    stack = [start]
    shape = occupancy.shape
    while stack:
        x, y, z = stack.pop()
        for dx, dy, dz in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)):
            candidate = (x + dx, y + dy, z + dz)
            if (0 <= candidate[0] < shape[0] and 0 <= candidate[1] < shape[1] and 0 <= candidate[2] < shape[2]
                    and occupancy[candidate] and candidate not in seen):
                seen.add(candidate)
                stack.append(candidate)
    return len(seen) == int(occupancy.sum())


def _element_nodes(occupied: np.ndarray, shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    n0, n1, n2 = (dimension + 1 for dimension in shape)
    del n0
    x, y, z = occupied.T
    def node_index(a0: np.ndarray, a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
        return a0 * n1 * n2 + a1 * n2 + a2
    global_nodes = np.stack((
        node_index(x, y, z), node_index(x, y, z + 1), node_index(x, y + 1, z), node_index(x, y + 1, z + 1),
        node_index(x + 1, y, z), node_index(x + 1, y, z + 1), node_index(x + 1, y + 1, z), node_index(x + 1, y + 1, z + 1),
    ), axis=1)
    unique, inverse = np.unique(global_nodes, return_inverse=True)
    return inverse.reshape(global_nodes.shape), unique


def assemble_voxel_system(occupancy: object, config: VoxelFEAConfig) -> AssembledVoxelSystem:
    """Assemble only after strict validation; caller occupancy is never mutated."""
    config = _validate_config(config)
    volume = _validate_occupancy(occupancy)
    if not _is_face_connected(volume):
        raise ValueError("disconnected_occupancy")
    occupied = np.argwhere(volume)
    n_elements = len(occupied)
    nodes, unique_nodes = _element_nodes(occupied, volume.shape)
    n_nodes = len(unique_nodes)
    element_dofs = np.empty((n_elements, 24), dtype=np.int64)
    for local in range(8):
        element_dofs[:, 3 * local:3 * local + 3] = 3 * nodes[:, local, None] + np.arange(3, dtype=np.int64)
    stiffness_element, body_force_element, b_gauss, elasticity = element_stiffness(
        config.youngs_modulus_pa, config.poisson_ratio, config.voxel_size
    )
    rows = np.broadcast_to(element_dofs[:, :, None], (n_elements, 24, 24)).ravel()
    columns = np.broadcast_to(element_dofs[:, None, :], (n_elements, 24, 24)).ravel()
    values = np.broadcast_to(stiffness_element, (n_elements, 24, 24)).ravel()
    stiffness = sparse.coo_matrix((values, (rows, columns)), shape=(3 * n_nodes, 3 * n_nodes)).tocsr()
    body_force = np.zeros(3 * n_nodes, dtype=np.float64)
    if config.include_self_weight:
        element_load = body_force_element @ (config.density_kg_m3 * np.asarray(config.gravity_m_s2, dtype=np.float64))
        np.add.at(body_force, element_dofs.ravel(), np.broadcast_to(element_load, (n_elements, 24)).ravel())
    node_coordinates = np.column_stack((
        unique_nodes // ((volume.shape[1] + 1) * (volume.shape[2] + 1)),
        (unique_nodes // (volume.shape[2] + 1)) % (volume.shape[1] + 1),
        unique_nodes % (volume.shape[2] + 1),
    )).astype(np.int64)
    min_x = int(occupied[:, 0].min())
    max_face_x = int(occupied[:, 0].max()) + 1
    fixed_nodes = np.flatnonzero(node_coordinates[:, 0] == min_x)
    loaded_nodes = np.flatnonzero(node_coordinates[:, 0] == max_face_x)
    if not len(fixed_nodes):
        raise ValueError("support_free_geometry")
    if not len(loaded_nodes) or np.intersect1d(fixed_nodes, loaded_nodes).size:
        raise ValueError("load_free_geometry")
    if config.expected_loaded_node_count is not None and len(loaded_nodes) != config.expected_loaded_node_count:
        raise ValueError("unstable_load_node_set")
    actual_loaded_coordinates = tuple(tuple(int(value) for value in node_coordinates[node]) for node in loaded_nodes)
    if config.expected_loaded_node_coordinates is not None and actual_loaded_coordinates != config.expected_loaded_node_coordinates:
        raise ValueError("unstable_load_node_set")
    fixed_force = np.zeros(3 * n_nodes, dtype=np.float64)
    per_node = np.asarray(config.fixed_total_force_n, dtype=np.float64) / len(loaded_nodes)
    for node in loaded_nodes:
        fixed_force[3 * node:3 * node + 3] += per_node
    fixed_dofs = np.sort(np.concatenate((3 * fixed_nodes, 3 * fixed_nodes + 1, 3 * fixed_nodes + 2)))
    all_dofs = np.arange(3 * n_nodes, dtype=np.int64)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs, assume_unique=True)
    return AssembledVoxelSystem(
        stiffness=stiffness, force=body_force + fixed_force,
        body_force_sum_n=np.array((body_force[0::3].sum(), body_force[1::3].sum(), body_force[2::3].sum())),
        fixed_force_sum_n=np.array((fixed_force[0::3].sum(), fixed_force[1::3].sum(), fixed_force[2::3].sum())),
        element_dofs=element_dofs, fixed_dofs=fixed_dofs, free_dofs=free_dofs, node_coordinates=node_coordinates,
        b_gauss=b_gauss, elasticity=elasticity, n_elements=n_elements, n_nodes=n_nodes,
        loaded_node_count=len(loaded_nodes), fixed_node_count=len(fixed_nodes),
    )


def _failure(
    reason: str, config: object, occupancy: object, *, relative_residual: float | None = None
) -> dict[str, object]:
    stable_reason = reason if isinstance(reason, str) and reason else "failure_record_construction_error"
    safe_residual = relative_residual if _is_finite_real(relative_residual) else None
    record = {
        "status": "failure", "reason": stable_reason, "config_digest": _safe_config_digest(config),
        "input_digest": _safe_input_digest(occupancy), "relative_residual": safe_residual, "outputs": None,
    }
    try:
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return record
    except Exception:
        fallback = {
            "status": "failure", "reason": "failure_record_construction_error", "config_digest": None,
            "input_digest": None, "relative_residual": None, "outputs": None,
        }
        json.dumps(fallback, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return fallback


def solve_voxels(occupancy: object, config: VoxelFEAConfig) -> dict[str, object]:
    """Return an append-only serializable canonical solve record, failing closed."""
    started = time.perf_counter()
    try:
        assembled = assemble_voxel_system(occupancy, config)
    except ValueError as error:
        return _failure(str(error), config, occupancy)
    except Exception:
        return _failure("assembly_failure", config, occupancy)
    try:
        reduced_stiffness = assembled.stiffness[assembled.free_dofs][:, assembled.free_dofs]
        reduced_force = assembled.force[assembled.free_dofs]
        if not np.any(np.abs(reduced_force) > 0.0):
            return _failure("load_free_geometry", config, occupancy)
        if config.solver_mode == "direct":
            if len(assembled.free_dofs) > config.direct_max_dof:
                return _failure("direct_solver_fixture_too_large", config, occupancy)
            displacement_free = sparse_linalg.spsolve(reduced_stiffness, reduced_force)
            status, iterations, preconditioner = 0, 0, "none_direct_reference"
            solver_identity = "scipy.sparse.linalg.spsolve"
            iterative_requested_rtol: float | None = None
        else:
            try:
                import pyamg
                preconditioner = pyamg.smoothed_aggregation_solver(reduced_stiffness).aspreconditioner(cycle="V")
                preconditioner_identity = "pyamg.smoothed_aggregation_solver@{}".format(pyamg.__version__)
            except Exception:
                return _failure("preconditioner_unavailable", config, occupancy)
            iterations_box = [0]
            def count_iteration(_: np.ndarray) -> None:
                iterations_box[0] += 1
            # The frozen relative_tolerance remains the acceptance bound;
            # request one order of magnitude tighter from the iterative solver.
            iterative_requested_rtol = config.relative_tolerance / 10.0
            displacement_free, status = sparse_linalg.cg(
                reduced_stiffness, reduced_force, M=preconditioner, rtol=iterative_requested_rtol,
                atol=0.0, maxiter=config.maximum_iterations, callback=count_iteration,
            )
            iterations, preconditioner = iterations_box[0], preconditioner_identity
            solver_identity = "scipy.sparse.linalg.cg"
            if status != 0:
                return _failure("iterative_nonconvergence", config, occupancy)
        residual_denominator = float(np.linalg.norm(reduced_force))
        relative_residual = float(np.linalg.norm(reduced_stiffness @ displacement_free - reduced_force) / residual_denominator)
        if not np.isfinite(relative_residual) or relative_residual > config.relative_tolerance:
            return _failure("relative_residual_exceeds_tolerance", config, occupancy, relative_residual=relative_residual)
        displacement = np.zeros(assembled.n_nodes * 3, dtype=np.float64)
        displacement[assembled.free_dofs] = displacement_free
        if not np.all(np.isfinite(displacement)):
            return _failure("nonfinite_displacement", config, occupancy)
        element_displacements = displacement[assembled.element_dofs]
        strains = np.einsum("gij,ej->gei", assembled.b_gauss, element_displacements)
        stresses = np.einsum("ij,gej->gei", assembled.elasticity, strains)
        sxx, syy, szz, syz, sxz, sxy = (stresses[..., index] for index in range(6))
        vm_squared = 0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * (sxy ** 2 + syz ** 2 + sxz ** 2)
        von_mises = np.sqrt(np.maximum(vm_squared, 0.0))
        compliance = float(displacement @ assembled.force)
        max_displacement = float(np.max(np.linalg.norm(displacement.reshape(-1, 3), axis=1)))
        if (not np.all(np.isfinite(von_mises)) or not np.isfinite(compliance) or not np.isfinite(max_displacement)
                or compliance <= 0.0 or max_displacement <= 0.0):
            return _failure("invalid_postprocessing", config, occupancy)
        record: dict[str, object] = {
            "status": "success", "reason": None,
            "stress_sampling": "eight_full_integration_gauss_points_per_element",
            "stress_sample_count": int(von_mises.size),
            "max_gauss_von_mises_pa": float(np.max(von_mises)),
            "p99_gauss_von_mises_pa": float(np.percentile(von_mises, 99.0)),
            "stress_metric_aliases": {
                "max_von_mises_pa": "max_gauss_von_mises_pa",
                "p99_von_mises_pa": "p99_gauss_von_mises_pa",
            },
            "max_von_mises_pa": float(np.max(von_mises)),
            "p99_von_mises_pa": float(np.percentile(von_mises, 99.0)), "max_displacement_m": max_displacement,
            "compliance_j": compliance, "element_count": assembled.n_elements, "node_count": assembled.n_nodes,
            "dof_count": int(3 * assembled.n_nodes), "free_dof_count": int(len(assembled.free_dofs)),
            "loaded_node_count": assembled.loaded_node_count, "fixed_node_count": assembled.fixed_node_count,
            "body_force_sum_n": assembled.body_force_sum_n.tolist(), "fixed_force_sum_n": assembled.fixed_force_sum_n.tolist(),
            "applied_force_sum_n": (assembled.body_force_sum_n + assembled.fixed_force_sum_n).tolist(),
            "iterations": iterations, "relative_residual": relative_residual, "solver_identity": solver_identity,
            "iterative_requested_rtol": iterative_requested_rtol,
            "preconditioner_identity": preconditioner, "config_digest": _safe_config_digest(config),
            "input_digest": _input_digest(occupancy),
        }
        if config.include_displacement_field:
            record["displacement_field_m"] = displacement.reshape(-1, 3).tolist()
        scientific = dict(record)
        record["scientific_digest"] = _scientific_digest(scientific)
        record["timing"] = {"wall_seconds": min(max(0.0, time.perf_counter() - started), 86_400.0)}
        return record
    except Exception:
        return _failure("solver_or_postprocessing_failure", config, occupancy)


def _elasticity_matrix(youngs_modulus: float, poisson_ratio: float) -> np.ndarray:
    lam = youngs_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    matrix = np.zeros((6, 6), dtype=np.float64)
    matrix[:3, :3] = lam
    matrix[range(3), range(3)] = lam + 2.0 * mu
    matrix[3, 3] = matrix[4, 4] = matrix[5, 5] = mu
    return matrix


def _strain_displacement(derivatives: np.ndarray) -> np.ndarray:
    """Return engineering-strain B for eight nodes in the canonical node order."""
    result = np.zeros((6, 24), dtype=np.float64)
    for node in range(8):
        x, y, z = derivatives[:, node]
        base = 3 * node
        result[0, base] = x
        result[1, base + 1] = y
        result[2, base + 2] = z
        result[3, base + 1] = z
        result[3, base + 2] = y
        result[4, base] = z
        result[4, base + 2] = x
        result[5, base] = y
        result[5, base + 1] = x
    return result


def _gauss_strain_displacement_matrices(voxel_size: Sequence[float]) -> np.ndarray:
    """Return the eight regular Hex8 B matrices at 2x2x2 Gauss points."""
    dx, dy, dz = (float(value) for value in voxel_size)
    gauss = 1.0 / np.sqrt(3.0)
    inverse_jacobian = np.array((2.0 / dx, 2.0 / dy, 2.0 / dz), dtype=np.float64)[:, None]
    matrices = []
    for xi in (-gauss, gauss):
        for eta in (-gauss, gauss):
            for zeta in (-gauss, gauss):
                derivatives_reference = np.array(
                    [
                        [-(1 - eta) * (1 - zeta), -(1 - eta) * (1 + zeta), -(1 + eta) * (1 - zeta), -(1 + eta) * (1 + zeta),
                         (1 - eta) * (1 - zeta), (1 - eta) * (1 + zeta), (1 + eta) * (1 - zeta), (1 + eta) * (1 + zeta)],
                        [-(1 - xi) * (1 - zeta), -(1 - xi) * (1 + zeta), (1 - xi) * (1 - zeta), (1 - xi) * (1 + zeta),
                         -(1 + xi) * (1 - zeta), -(1 + xi) * (1 + zeta), (1 + xi) * (1 - zeta), (1 + xi) * (1 + zeta)],
                        [-(1 - xi) * (1 - eta), (1 - xi) * (1 - eta), -(1 - xi) * (1 + eta), (1 - xi) * (1 + eta),
                         -(1 + xi) * (1 - eta), (1 + xi) * (1 - eta), -(1 + xi) * (1 + eta), (1 + xi) * (1 + eta)],
                    ], dtype=np.float64,
                ) / 8.0
                matrices.append(_strain_displacement(inverse_jacobian * derivatives_reference))
    return np.asarray(matrices)


def element_stiffness(
    youngs_modulus: float, poisson_ratio: float, voxel_size: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate a full-integration regular Hex8 stiffness and unit body-force map.

    The returned body-force map turns a physical force density ``(N/m^3)`` into
    the element's 24-DOF equivalent nodal load. The eight returned B matrices
    are the declared locations for all canonical stress statistics.
    """
    dx, dy, dz = (float(value) for value in voxel_size)
    if not all(np.isfinite((youngs_modulus, poisson_ratio, dx, dy, dz))):
        raise ValueError("element parameters must be finite")
    if youngs_modulus <= 0.0 or not (-1.0 < poisson_ratio < 0.5) or min(dx, dy, dz) <= 0.0:
        raise ValueError("invalid element parameters")
    matrix = _elasticity_matrix(float(youngs_modulus), float(poisson_ratio))
    gauss = 1.0 / np.sqrt(3.0)
    points = tuple((x, y, z) for x in (-gauss, gauss) for y in (-gauss, gauss) for z in (-gauss, gauss))
    determinant = dx * dy * dz / 8.0
    b_gauss = _gauss_strain_displacement_matrices((dx, dy, dz))
    stiffness = np.zeros((24, 24), dtype=np.float64)
    body_force = np.zeros((24, 3), dtype=np.float64)
    for (xi, eta, zeta), b_matrix in zip(points, b_gauss, strict=True):
        stiffness += b_matrix.T @ matrix @ b_matrix * determinant
        shapes = np.array(
            ((1-xi)*(1-eta)*(1-zeta), (1-xi)*(1-eta)*(1+zeta),
             (1-xi)*(1+eta)*(1-zeta), (1-xi)*(1+eta)*(1+zeta),
             (1+xi)*(1-eta)*(1-zeta), (1+xi)*(1-eta)*(1+zeta),
             (1+xi)*(1+eta)*(1-zeta), (1+xi)*(1+eta)*(1+zeta)), dtype=np.float64,
        ) / 8.0
        for node, shape in enumerate(shapes):
            body_force[3 * node : 3 * node + 3, :] += np.eye(3) * shape * determinant
    return stiffness, body_force, b_gauss, matrix
