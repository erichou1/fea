#!/usr/bin/env python3
"""
SIMP Baseline Benchmark -- Standard density-based topology optimization

Runs SIMP (Solid Isotropic Material with Penalization) on 10 selected geometries
at 64^3 resolution to provide an empirical comparison against SASTO.

SIMP setup:
  - Density-based formulation with penalization p=3
  - OC (Optimality Criteria) update with move limit
  - Density filter (radius = 1.5 * voxel edge, uniform_filter)
  - Compliance minimization subject to volume fraction constraint
  - Same BCs as SASTO (fixed at min-x face, gravity loading)
  - Target volume fraction = 1 - target_reduction for each design

The benchmark compares:
  - Final volume reduction achieved
  - Compliance ratio (optimized / baseline)
  - Wall-clock runtime (and number of FEA evaluations)
"""
import sys, os, json, time, argparse, gc, traceback, random
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, spsolve
from scipy.ndimage import uniform_filter
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from run_fea_validation_100 import hex8_Ke_and_B


# ---------------------------------------------------------------- #
#                    Vectorized SIMP FEA solver                     #
# ---------------------------------------------------------------- #
def downsample_occ(occ, target=64):
    """Block-average downsample from 128^3 to target^3."""
    f = occ.shape[0] // target
    if f <= 1:
        return occ.copy()
    return (occ.reshape(target, f, target, f, target, f)
              .mean(axis=(1, 3, 5)) > 0.5).astype(np.uint8)


def _build_mesh(occ):
    """Return element coords, compact DOFs, and metadata for an occupancy grid."""
    a0, a1, a2 = np.nonzero(occ)
    n_elem = len(a0)
    D0, D1, D2 = occ.shape
    N0, N1, N2 = D0+1, D1+1, D2+1

    def nidx(i0, i1, i2):
        return i0*N1*N2 + i1*N2 + i2

    elem_nodes_global = np.stack([
        nidx(a0,a1,a2),     nidx(a0,a1,a2+1),
        nidx(a0,a1+1,a2+1), nidx(a0,a1+1,a2),
        nidx(a0+1,a1,a2),   nidx(a0+1,a1,a2+1),
        nidx(a0+1,a1+1,a2+1), nidx(a0+1,a1+1,a2),
    ], axis=1)

    unique_nodes = np.unique(elem_nodes_global)
    n_nodes = len(unique_nodes)
    n_dof = n_nodes * 3

    node_compact = np.empty(unique_nodes.max()+1, dtype=np.int32)
    node_compact[unique_nodes] = np.arange(n_nodes, dtype=np.int32)
    elem_nodes = node_compact[elem_nodes_global]

    elem_dofs = np.empty((n_elem, 24), dtype=np.int32)
    for j in range(8):
        elem_dofs[:, 3*j]   = 3*elem_nodes[:, j]
        elem_dofs[:, 3*j+1] = 3*elem_nodes[:, j] + 1
        elem_dofs[:, 3*j+2] = 3*elem_nodes[:, j] + 2

    # BC: fix at min a0 face
    a0_of_node = unique_nodes // (N1*N2)
    a0_min = a0.min()
    bc_compact = np.where(a0_of_node == a0_min)[0]
    bc_dofs = np.concatenate([3*bc_compact, 3*bc_compact+1, 3*bc_compact+2])
    bc_dofs.sort()
    free_mask = np.ones(n_dof, dtype=bool)
    free_mask[bc_dofs] = False
    free_dofs = np.where(free_mask)[0]

    return {
        'a0': a0, 'a1': a1, 'a2': a2,
        'n_elem': n_elem, 'n_dof': n_dof, 'n_nodes': n_nodes,
        'elem_dofs': elem_dofs,
        'free_dofs': free_dofs,
        'bc_dofs': bc_dofs,
        'unique_nodes': unique_nodes,
        'N1': N1, 'N2': N2,
        # Mapping from global DOF -> free-DOF index (-1 if fixed)
        'dof_to_free': _build_dof_to_free(n_dof, free_dofs),
    }

def _build_dof_to_free(n_dof, free_dofs):
    """Build mapping from global DOF index to free-DOF index."""
    m = np.full(n_dof, -1, dtype=np.int32)
    m[free_dofs] = np.arange(len(free_dofs), dtype=np.int32)
    return m


def simp_fea(mesh, rho_elem, Ke, fe_grav_elem, penal=3.0, rho_min=1e-3):
    """
    Assemble and solve with SIMP-penalized densities (vectorized).
    Assembles directly into free-DOF space to avoid expensive submatrix extraction.
    """
    n_elem = mesh['n_elem']
    n_dof = mesh['n_dof']
    elem_dofs = mesh['elem_dofs']
    free_dofs = mesh['free_dofs']
    dof_to_free = mesh['dof_to_free']
    n_free = len(free_dofs)

    rho_pen = np.maximum(rho_elem, rho_min) ** penal   # E(rho) = rho^p * E0

    # Map element DOFs to free-DOF indices
    elem_free = dof_to_free[elem_dofs]   # (n_elem, 24), -1 for fixed DOFs

    # -- Vectorized assembly directly into free-DOF space --
    row_idx = np.repeat(elem_free[:, :, np.newaxis], 24, axis=2)   # (n,24,24)
    col_idx = np.repeat(elem_free[:, np.newaxis, :], 24, axis=1)   # (n,24,24)
    val = Ke[np.newaxis, :, :] * rho_pen[:, np.newaxis, np.newaxis]  # (n,24,24)

    # Mask out entries involving fixed DOFs
    mask = (row_idx >= 0) & (col_idx >= 0)
    K_ff = sparse.coo_matrix(
        (val.ravel()[mask.ravel()],
         (row_idx.ravel()[mask.ravel()], col_idx.ravel()[mask.ravel()])),
        shape=(n_free, n_free)
    ).tocsc()

    # Force vector: gravity proportional to density (mass = rho * V)
    fe_scaled = fe_grav_elem[np.newaxis, :] * rho_elem[:, np.newaxis]  # (n,24)
    f_global = np.zeros(n_dof, dtype=np.float64)
    np.add.at(f_global, elem_dofs.ravel(), fe_scaled.ravel())
    f_f = f_global[free_dofs]

    # Direct solver — fast for 64^3 resolution (~12K DOFs)
    try:
        u_f = spsolve(K_ff, f_f)
    except Exception:
        K_ff_csr = K_ff.tocsr()
        diag = K_ff_csr.diagonal().copy()
        diag[diag <= 0] = 1.0
        M_pre = sparse.diags(1.0/diag, format='csr')
        u_f, info = cg(K_ff_csr, f_f, M=M_pre, rtol=1e-6, maxiter=5000)
        if info != 0:
            u_f, _ = cg(K_ff_csr, f_f, rtol=1e-4, maxiter=10000)

    u_global = np.zeros(n_dof)
    u_global[free_dofs] = u_f

    compliance = float(f_global @ u_global)

    # Per-element compliance: c_e = u_e^T Ke u_e   (without rho factor)
    u_elem = u_global[elem_dofs]                          # (n,24)
    elem_comp = np.einsum('ei,ij,ej->e', u_elem, Ke, u_elem)  # (n,)

    return compliance, u_global, elem_comp


# ---------------------------------------------------------------- #
#                      SIMP main loop (OC)                          #
# ---------------------------------------------------------------- #
def run_simp(occ, voxel_size, target_vf, E=25e9, nu=0.2, density=2400.0,
             penal=3.0, max_iter=200, move=0.2, tol=0.01, filt_r=1.5,
             verbose=True):
    """
    Density-based SIMP with OC update and density filter.
    Returns dict of results (or dict with 'error' on failure).
    """
    from scipy.ndimage import label as ndlabel
    N = occ.shape[0]
    rho_min = 1e-3
    dx = dy = dz = voxel_size

    # Largest connected component touching BC face
    a0_raw_min = np.nonzero(occ)[0].min() if occ.any() else 0
    labeled, nc = ndlabel(occ)
    if nc > 1:
        bc_labels = set(np.unique(labeled[a0_raw_min, :, :])) - {0}
        if bc_labels:
            keep = np.isin(labeled, list(bc_labels))
            occ = (occ & keep).astype(np.uint8)

    mesh = _build_mesh(occ)
    n_elem = mesh['n_elem']
    if n_elem < 10:
        return {'error': 'too few elements'}

    Ke, fe_grav_ref, B_c, Dmat = hex8_Ke_and_B(E, nu, dx, dy, dz)
    fe_grav_elem = fe_grav_ref * (-density * 9.81)

    a0, a1, a2 = mesh['a0'], mesh['a1'], mesh['a2']
    target_vol = target_vf * n_elem

    # Density field stored per-element; initial = target_vf (uniform)
    rho_elem = np.full(n_elem, target_vf)

    history = []
    n_fea = 0
    t_total = time.time()
    change = 1.0

    if verbose:
        print(f"  SIMP: N={N}, elems={n_elem}, target_vf={target_vf:.3f}")
        sys.stdout.flush()

    for it in range(max_iter):
        t_it = time.time()

        result = simp_fea(mesh, rho_elem, Ke, fe_grav_elem, penal=penal, rho_min=rho_min)
        n_fea += 1
        if result is None:
            return {'error': f'FEA failed at iter {it}'}
        compliance, u_global, elem_comp = result

        # Sensitivity: dc/drho_e = -p * rho^{p-1} * u_e^T Ke u_e
        rho_safe = np.maximum(rho_elem, rho_min)
        dc = -penal * rho_safe**(penal-1) * elem_comp

        # Density filter via 3-D uniform_filter on the lattice
        # Map sensitivities back to grid, filter, read back
        dc_grid = np.zeros((N, N, N), dtype=np.float64)
        rho_grid = np.zeros((N, N, N), dtype=np.float64)
        dc_grid[a0, a1, a2] = dc * rho_elem            # numerator
        rho_grid[a0, a1, a2] = rho_elem                 # denominator
        ks = int(np.ceil(filt_r)) * 2 + 1
        dc_filt = uniform_filter(dc_grid, size=ks, mode='constant')
        rho_filt = uniform_filter(rho_grid, size=ks, mode='constant')
        rho_filt = np.maximum(rho_filt, 1e-12)
        dc_tilde = dc_filt[a0, a1, a2] / (rho_filt[a0, a1, a2] * rho_safe)

        # OC update (bisection on Lagrange multiplier)
        l1, l2 = 1e-30, 1e12
        for _ in range(200):
            lmid = 0.5 * (l1 + l2)
            B_e = np.sqrt(np.maximum(-dc_tilde / lmid, 1e-30))
            rho_new = np.maximum(rho_min,
                      np.maximum(rho_elem - move,
                      np.minimum(1.0,
                      np.minimum(rho_elem + move, rho_elem * B_e))))
            if rho_new.sum() > target_vol:
                l1 = lmid
            else:
                l2 = lmid
            if (l2 - l1) / (l1 + l2 + 1e-30) < 1e-9:
                break

        change = float(np.max(np.abs(rho_new - rho_elem)))
        rho_elem = rho_new

        vf_cur = float(rho_elem.sum() / n_elem)
        dt = time.time() - t_it
        history.append({'iter': it+1, 'compliance': compliance, 'vf': vf_cur, 'time': dt, 'change': change})

        if verbose and (it < 5 or (it+1) % 10 == 0):
            print(f"    It {it+1:3d}: C={compliance:.4g}, VF={vf_cur:.4f}, chg={change:.4f}, dt={dt:.1f}s")
            sys.stdout.flush()

        if it > 10 and change < tol:
            if verbose:
                print(f"    Converged at iter {it+1} (change={change:.6f})")
            break
        # Secondary convergence: compliance change < 0.1% over last 10 iterations
        if it >= 20 and len(history) >= 10:
            c_recent = [h['compliance'] for h in history[-10:]]
            c_range = max(c_recent) - min(c_recent)
            if c_range < 0.01 * abs(c_recent[-1]):
                if verbose:
                    print(f"    Converged (compliance stable) at iter {it+1}")
                break

    total_time = time.time() - t_total

    # Final evaluation with thresholded design rho>=0.5 -> 1, else 0
    rho_thresh = np.where(rho_elem >= 0.5, 1.0, rho_min)
    result_thresh = simp_fea(mesh, rho_thresh, Ke, fe_grav_elem, penal=1.0, rho_min=rho_min)
    n_fea += 1
    # Baseline: all rho=1
    result_base = simp_fea(mesh, np.ones(n_elem), Ke, fe_grav_elem, penal=1.0, rho_min=rho_min)
    n_fea += 1

    final_vf = float((rho_thresh >= 0.5).sum() / n_elem)
    final_reduction = (1.0 - final_vf) * 100

    if result_thresh is not None and result_base is not None:
        comp_ratio = result_thresh[0] / result_base[0]
    else:
        comp_ratio = float('inf')

    return {
        'n_fea_evaluations': n_fea,
        'n_iterations': len(history),
        'converged': change < tol,
        'final_compliance': result_thresh[0] if result_thresh else None,
        'baseline_compliance': result_base[0] if result_base else None,
        'comp_ratio': comp_ratio,
        'final_vf': final_vf,
        'volume_reduction_pct': final_reduction,
        'target_vf': target_vf,
        'total_time_s': total_time,
        'time_per_fea': total_time / n_fea if n_fea > 0 else 0,
        'history': history,
        'resolution': N,
    }


# ---------------------------------------------------------------- #
#                  Geometry selection and main                      #
# ---------------------------------------------------------------- #
def select_benchmark_geometries(batch_dir, data_dir, splits_path, n=10, seed=42):
    """3 high-reduction, 4 near-boundary, 3 easy."""
    with open(splits_path) as f:
        splits = json.load(f)
    test_ids = set(p.split('/')[-1] for p in splits['test'])

    samples = []
    for d in sorted(os.listdir(batch_dir)):
        sp = os.path.join(batch_dir, d, 'optimization_summary.json')
        op = os.path.join(batch_dir, d, 'optimized_occ.npz')
        if not os.path.isfile(sp) or not os.path.isfile(op):
            continue
        with open(sp) as f:
            s = json.load(f)
        if s['sample_id'] not in test_ids or not s.get('success', True):
            continue
        if not s.get('constraints_satisfied', False):
            continue
        s['_dir'] = os.path.join(batch_dir, d)
        samples.append(s)

    samples.sort(key=lambda x: x.get('volume_reduction_pct', 0), reverse=True)

    high = samples[:3]
    used = {s['sample_id'] for s in high}

    remaining = [s for s in samples if s['sample_id'] not in used]
    for s in remaining:
        s['_max_util'] = max(s.get('vm_utilization', 0), s.get('comp_utilization', 0))
    remaining.sort(key=lambda x: x['_max_util'], reverse=True)
    boundary = remaining[:4]
    used.update(s['sample_id'] for s in boundary)

    remaining2 = [s for s in samples if s['sample_id'] not in used]
    remaining2.sort(key=lambda x: x.get('_max_util', x.get('volume_reduction_pct', 0)))
    easy = remaining2[:3]

    selected = []
    for s in high:
        selected.append(('high_reduction', s))
    for s in boundary:
        selected.append(('near_boundary', s))
    for s in easy:
        selected.append(('easy', s))
    return selected


def main():
    parser = argparse.ArgumentParser(description='SIMP Baseline Benchmark')
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--max-iter', type=int, default=60)
    parser.add_argument('--n-designs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    BATCH_DIR = Path('runs/v3/batch_results_all')
    DATA_DIR  = Path('data/runs_real_128')
    RUNS_V3   = Path('runs/v3')
    OUT_FILE  = RUNS_V3 / 'simp_benchmark.json'
    LOG_FILE  = RUNS_V3 / 'simp_benchmark_out.txt'

    log_f = open(LOG_FILE, 'w', encoding='utf-8', errors='replace')
    class Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, data):
            for s in self.streams: s.write(data); s.flush()
        def flush(self):
            for s in self.streams: s.flush()
    sys.stdout = Tee(sys.__stdout__, log_f)

    print(f'SIMP Baseline Benchmark  resolution={args.resolution}  max_iter={args.max_iter}')
    print()

    selected = select_benchmark_geometries(
        str(BATCH_DIR), str(DATA_DIR), str(RUNS_V3 / 'splits.json'),
        n=args.n_designs, seed=args.seed
    )

    # Resume: load existing results if any
    results = []
    done_ids = set()
    if OUT_FILE.exists():
        try:
            results = json.load(open(OUT_FILE))
            done_ids = {r['sample_id'] for r in results}
            print(f"  Resuming: {len(results)} designs already done")
        except Exception:
            pass

    for i, (group, s) in enumerate(selected):
        sid = s['sample_id']
        if sid in done_ids:
            print(f"\n[{i+1}/{len(selected)}] Sample {sid} -- SKIPPING (already done)")
            continue
        sasto_red = s.get('volume_reduction_pct', 0)
        target_vf = max(0.05, 1.0 - sasto_red / 100.0)

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(selected)}] Sample {sid} ({group})")
        print(f"  SASTO red={sasto_red:.1f}%, target VF={target_vf:.3f}")

        try:
            meta_path = Path(DATA_DIR) / sid.zfill(5) / 'meta.json'
            with open(meta_path) as f:
                meta = json.load(f)
            vs_orig = meta['voxel_size']

            occ_128 = np.load(Path(DATA_DIR) / sid.zfill(5) / 'occ.npz')['data'].astype(np.uint8)
            if args.resolution < 128:
                occ = downsample_occ(occ_128, args.resolution)
                vs = vs_orig * (128.0 / args.resolution)
            else:
                occ = occ_128
                vs = vs_orig

            print(f"  Grid {occ.shape[0]}^3, active={int(occ.sum())}, vs={vs:.4f}m")

            t0 = time.time()
            res = run_simp(occ, vs, target_vf,
                           E=meta.get('E', 25e9), nu=meta.get('nu', 0.2),
                           density=meta.get('density', 2400.0),
                           max_iter=args.max_iter, verbose=True)
            dt = time.time() - t0

            res['sample_id'] = sid
            res['group'] = group
            res['sasto_reduction_pct'] = sasto_red
            res['sasto_time_s'] = s.get('optimization_time', 0)
            res['sasto_comp_ratio'] = s.get('comp_utilization', 0)

            if 'error' not in res:
                print(f"  SIMP: red={res['volume_reduction_pct']:.1f}%, "
                      f"C_ratio={res['comp_ratio']:.4f}, "
                      f"FEA_evals={res['n_fea_evaluations']}, time={dt:.1f}s")
                print(f"  SASTO: red={sasto_red:.1f}%, time={s.get('optimization_time',0):.1f}s")
            else:
                print(f"  SIMP FAILED: {res['error']}")

            results.append(res)
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            results.append({'sample_id': sid, 'group': group, 'error': str(e),
                            'sasto_reduction_pct': sasto_red})

        with open(OUT_FILE, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        gc.collect()

    # ---- Summary ----
    print(f"\n{'='*70}")
    print('SIMP vs SASTO BENCHMARK SUMMARY')
    print(f"{'='*70}")
    valid = [r for r in results if 'error' not in r]
    if valid:
        hdr = f"{'Sample':>8} {'Group':>15} {'SIMP%':>6} {'SASTO%':>7} {'C_rat':>7} {'SIMP_s':>7} {'SASTO_s':>8} {'Ratio':>6}"
        print(hdr)
        print('-' * len(hdr))
        for r in valid:
            sasto_t = r.get('sasto_time_s', 0)
            simp_t = r['total_time_s']
            ratio = simp_t / sasto_t if sasto_t > 0 else float('inf')
            print(f"{r['sample_id']:>8} {r['group']:>15} "
                  f"{r['volume_reduction_pct']:>5.1f}% {r['sasto_reduction_pct']:>6.1f}% "
                  f"{r['comp_ratio']:>7.4f} {simp_t:>6.0f}s {sasto_t:>7.1f}s {ratio:>5.0f}x")

        simp_times = [r['total_time_s'] for r in valid]
        sasto_times = [r.get('sasto_time_s', 0) for r in valid if r.get('sasto_time_s', 0) > 0]
        if sasto_times:
            ratios = [r['total_time_s']/r['sasto_time_s'] for r in valid if r.get('sasto_time_s',0)>0]
            print(f"\nMedian SIMP/SASTO time ratio: {np.median(ratios):.0f}x")
        print(f"SIMP median wall-clock: {np.median(simp_times):.0f}s")

    log_f.close()


if __name__ == '__main__':
    main()
