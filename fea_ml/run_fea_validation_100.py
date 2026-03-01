#!/usr/bin/env python3
"""
Ground-Truth FEA Validation — Same-Method Voxel Baseline vs Optimized (100 designs)

For each design:
  1. Run hex8 voxel FEA on the BASELINE geometry
  2. Run hex8 voxel FEA on the OPTIMIZED geometry
  3. Compute compliance ratio C_opt / C_base  (must be ≤ 1.15)

Uses AMG preconditioning (pyamg) for ~10-50x faster convergence than Jacobi CG.

Sampling strategy:
  - 30 highest-reduction (35-45%)
  - 40 near-boundary (constraint utilization > 90%)
  - 30 random from remaining CS geometries
"""
import sys, os, json, argparse, time, gc, random, traceback
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, LinearOperator
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def hex8_Ke_and_B(E, nu, dx, dy, dz):
    """24x24 element stiffness matrix for regular hex8 brick."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu  = E / (2.0 * (1 + nu))
    D = np.zeros((6, 6), dtype=np.float64)
    D[0,0] = D[1,1] = D[2,2] = lam + 2*mu
    D[0,1] = D[0,2] = D[1,0] = D[1,2] = D[2,0] = D[2,1] = lam
    D[3,3] = D[4,4] = D[5,5] = mu

    detJ = (dx * dy * dz) / 8.0
    inv2 = [2.0/dx, 2.0/dy, 2.0/dz]

    gp = 1.0 / np.sqrt(3.0)
    gauss_pts = np.array([
        [-gp,-gp,-gp],[gp,-gp,-gp],[-gp,gp,-gp],[gp,gp,-gp],
        [-gp,-gp, gp],[gp,-gp, gp],[-gp,gp, gp],[gp,gp, gp],
    ])

    Ke = np.zeros((24, 24), dtype=np.float64)
    fe_grav = np.zeros(24, dtype=np.float64)

    for xi, eta, zeta in gauss_pts:
        dN = np.array([
            [-(1-eta)*(1-zeta),  (1-eta)*(1-zeta),
              (1+eta)*(1-zeta), -(1+eta)*(1-zeta),
             -(1-eta)*(1+zeta),  (1-eta)*(1+zeta),
              (1+eta)*(1+zeta), -(1+eta)*(1+zeta)],
            [-(1-xi)*(1-zeta), -(1+xi)*(1-zeta),
              (1+xi)*(1-zeta),  (1-xi)*(1-zeta),
             -(1-xi)*(1+zeta), -(1+xi)*(1+zeta),
              (1+xi)*(1+zeta),  (1-xi)*(1+zeta)],
            [-(1-xi)*(1-eta), -(1+xi)*(1-eta),
             -(1+xi)*(1+eta), -(1-xi)*(1+eta),
              (1-xi)*(1-eta),  (1+xi)*(1-eta),
              (1+xi)*(1+eta),  (1-xi)*(1+eta)],
        ]) / 8.0
        dNdx = np.array(inv2).reshape(3,1) * dN
        B = np.zeros((6, 24), dtype=np.float64)
        for i in range(8):
            c = 3*i
            B[0,c]   = dNdx[0,i]; B[1,c+1] = dNdx[1,i]; B[2,c+2] = dNdx[2,i]
            B[3,c+1] = dNdx[2,i]; B[3,c+2] = dNdx[1,i]
            B[4,c]   = dNdx[2,i]; B[4,c+2] = dNdx[0,i]
            B[5,c]   = dNdx[1,i]; B[5,c+1] = dNdx[0,i]
        Ke += B.T @ D @ B * detJ
        N = np.array([
            (1-xi)*(1-eta)*(1-zeta), (1+xi)*(1-eta)*(1-zeta),
            (1+xi)*(1+eta)*(1-zeta), (1-xi)*(1+eta)*(1-zeta),
            (1-xi)*(1-eta)*(1+zeta), (1+xi)*(1-eta)*(1+zeta),
            (1+xi)*(1+eta)*(1+zeta), (1-xi)*(1+eta)*(1+zeta),
        ]) / 8.0
        for i in range(8):
            fe_grav[3*i] += N[i] * detJ

    # Centroid B-matrix for stress recovery
    dN_c = np.array([
        [-1,1,1,-1,-1,1,1,-1],[-1,-1,1,1,-1,-1,1,1],[-1,-1,-1,-1,1,1,1,1],
    ], dtype=np.float64) / 8.0
    dNdx_c = np.array(inv2).reshape(3,1) * dN_c
    B_c = np.zeros((6, 24), dtype=np.float64)
    for i in range(8):
        c = 3*i
        B_c[0,c]   = dNdx_c[0,i]; B_c[1,c+1] = dNdx_c[1,i]; B_c[2,c+2] = dNdx_c[2,i]
        B_c[3,c+1] = dNdx_c[2,i]; B_c[3,c+2] = dNdx_c[1,i]
        B_c[4,c]   = dNdx_c[2,i]; B_c[4,c+2] = dNdx_c[0,i]
        B_c[5,c]   = dNdx_c[1,i]; B_c[5,c+1] = dNdx_c[0,i]
    return Ke, fe_grav, B_c, D


def voxel_fea(occ, voxel_size, E=25e9, nu=0.20, rho=2400.0, g=9.81, use_amg=True):
    """
    Hex8 FEA with AMG-preconditioned CG solver (falls back to Jacobi if AMG unavailable).
    Returns dict with max_von_mises, compliance, etc., or None if too few elements.
    """
    from scipy.ndimage import label as ndlabel

    # Keep only largest connected component touching BC face
    a0_min_raw = np.nonzero(occ)[0].min() if occ.any() else 0
    labeled, n_comp = ndlabel(occ)
    if n_comp > 1:
        bc_labels = set(np.unique(labeled[a0_min_raw, :, :])) - {0}
        if bc_labels:
            keep_mask = np.isin(labeled, list(bc_labels))
            n_removed = int(occ.sum()) - int(keep_mask.sum())
            if n_removed > 0:
                occ = (occ & keep_mask).astype(np.uint8)

    dx = dy = dz = voxel_size
    Ke, fe_grav_ref, B_c, Dmat = hex8_Ke_and_B(E, nu, dx, dy, dz)
    fe_grav_elem = fe_grav_ref * (-rho * g)

    a0, a1, a2 = np.nonzero(occ)
    n_elem = len(a0)
    if n_elem < 5:
        return None

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

    # Element DOFs
    elem_dofs = np.empty((n_elem, 24), dtype=np.int32)
    for j in range(8):
        elem_dofs[:, 3*j]   = 3*elem_nodes[:,j]
        elem_dofs[:, 3*j+1] = 3*elem_nodes[:,j]+1
        elem_dofs[:, 3*j+2] = 3*elem_nodes[:,j]+2

    # Assembly
    row_idx = np.repeat(elem_dofs[:,:,np.newaxis], 24, axis=2)
    col_idx = np.repeat(elem_dofs[:,np.newaxis,:], 24, axis=1)
    val = np.broadcast_to(Ke[np.newaxis,:,:], (n_elem,24,24)).copy()
    K = sparse.coo_matrix(
        (val.ravel(), (row_idx.ravel(), col_idx.ravel())),
        shape=(n_dof, n_dof)
    ).tocsr()
    del row_idx, col_idx, val

    f_global = np.zeros(n_dof, dtype=np.float64)
    np.add.at(f_global, elem_dofs.ravel(),
              np.broadcast_to(fe_grav_elem[np.newaxis,:], (n_elem,24)).ravel())

    # BCs: fix at min a0 face
    a0_of_node = unique_nodes // (N1*N2)
    a0_min_struct = a0.min()
    bc_compact = np.where(a0_of_node == a0_min_struct)[0]
    bc_dofs = np.concatenate([3*bc_compact, 3*bc_compact+1, 3*bc_compact+2])
    bc_dofs.sort()

    if len(bc_dofs) == 0:
        return None

    free_mask = np.ones(n_dof, dtype=bool)
    free_mask[bc_dofs] = False
    free_dofs = np.where(free_mask)[0]

    K_ff = K[free_dofs][:, free_dofs]
    f_f = f_global[free_dofs]
    n_free = len(free_dofs)

    del K
    gc.collect()

    # Solve with AMG preconditioning (much faster than Jacobi for structured problems)
    t_solve = time.time()
    solve_iters = [0]
    def callback(xk):
        solve_iters[0] += 1

    amg_used = False
    if use_amg:
        try:
            import pyamg
            # For 3D elasticity, provide near-null space (6 rigid body modes)
            # This is critical for AMG performance on vector-valued PDEs
            B_null = np.zeros((n_free, 6), dtype=np.float64)
            # Get coordinates of free nodes for rigid body modes
            free_node_ids = free_dofs // 3
            free_dof_comp = free_dofs % 3  # 0=x, 1=y, 2=z
            
            # Translation modes
            for d in range(3):
                mask_d = (free_dof_comp == d)
                B_null[mask_d, d] = 1.0
            
            # Rotation modes (cross product with position)
            unique_a0 = unique_nodes // (N1*N2)
            unique_a1 = (unique_nodes % (N1*N2)) // N2
            unique_a2 = unique_nodes % N2
            
            node_coords = np.column_stack([unique_a0, unique_a1, unique_a2]).astype(np.float64)
            node_coords *= voxel_size  # physical coordinates
            
            # Map free DOFs to compact node IDs (vectorized)
            free_compact_node = free_dofs // 3
            
            # Rotation around z: [-y, x, 0]
            B_null[free_dof_comp == 0, 3] = -node_coords[free_compact_node[free_dof_comp == 0], 1]
            B_null[free_dof_comp == 1, 3] =  node_coords[free_compact_node[free_dof_comp == 1], 0]
            # Rotation around y: [z, 0, -x]
            B_null[free_dof_comp == 0, 4] =  node_coords[free_compact_node[free_dof_comp == 0], 2]
            B_null[free_dof_comp == 2, 4] = -node_coords[free_compact_node[free_dof_comp == 2], 0]
            # Rotation around x: [0, -z, y]
            B_null[free_dof_comp == 1, 5] = -node_coords[free_compact_node[free_dof_comp == 1], 2]
            B_null[free_dof_comp == 2, 5] =  node_coords[free_compact_node[free_dof_comp == 2], 1]
            
            ml = pyamg.smoothed_aggregation_solver(K_ff, B=B_null, max_coarse=500,
                                                    smooth='energy')
            M_amg = ml.aspreconditioner(cycle='V')
            u_f, info = cg(K_ff, f_f, M=M_amg, rtol=1e-5, maxiter=10000, callback=callback)
            amg_used = True
        except Exception as e:
            print(f"    AMG failed ({e}), falling back to Jacobi")
            amg_used = False

    if not amg_used:
        # Fallback: Jacobi CG
        diag_K = K_ff.diagonal().copy()
        diag_K[diag_K == 0] = 1.0
        M_pre = sparse.diags(1.0/diag_K, format="csr")
        u_f, info = cg(K_ff, f_f, M=M_pre, rtol=1e-5, maxiter=50000, callback=callback)

    solve_time = time.time() - t_solve

    u_global = np.zeros(n_dof, dtype=np.float64)
    u_global[free_dofs] = u_f

    del K_ff, f_f
    gc.collect()

    # Post-process
    u3 = u_global.reshape(-1, 3)
    u_mag = np.linalg.norm(u3, axis=1)
    max_displacement = float(np.max(u_mag))
    compliance = float(u_global @ f_global)

    # Von Mises stress
    u_elem = u_global[elem_dofs]
    strain_all = B_c @ u_elem.T
    stress_all = Dmat @ strain_all
    sxx, syy, szz = stress_all[0], stress_all[1], stress_all[2]
    syz, sxz, sxy = stress_all[3], stress_all[4], stress_all[5]
    vm_sq = 0.5*((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2) + 3.0*(sxy**2+syz**2+sxz**2)
    vm = np.sqrt(np.maximum(vm_sq, 0.0))
    max_vm = float(np.max(vm))
    vm_p99 = float(np.percentile(vm, 99))
    vm_p95 = float(np.percentile(vm, 95))

    return {
        "max_von_mises": max_vm,
        "vm_p99": vm_p99,
        "vm_p95": vm_p95,
        "max_displacement": max_displacement,
        "compliance": compliance,
        "n_elements": n_elem,
        "n_dof": n_dof,
        "n_free_dof": n_free,
        "solve_iters": solve_iters[0],
        "solve_time_s": solve_time,
        "amg_used": amg_used,
    }


def select_samples(batch_dir, data_dir, splits_path, n_high=30, n_boundary=40, n_random=30, seed=42):
    """Select stratified samples for FEA validation."""
    with open(splits_path) as f:
        splits = json.load(f)
    test_ids = set(p.split('/')[-1] for p in splits['test'])

    samples = []
    for d in sorted(os.listdir(batch_dir)):
        summary_p = os.path.join(batch_dir, d, 'optimization_summary.json')
        occ_p = os.path.join(batch_dir, d, 'optimized_occ.npz')
        if not os.path.isfile(summary_p) or not os.path.isfile(occ_p):
            continue
        with open(summary_p) as f:
            s = json.load(f)
        sid = s['sample_id']
        if sid not in test_ids:
            continue
        if not s.get('success', True):
            continue
        s['_dir'] = os.path.join(batch_dir, d)
        samples.append(s)

    cs = [s for s in samples if s.get('constraints_satisfied', False)]
    cs.sort(key=lambda x: x.get('volume_reduction_pct', 0), reverse=True)

    print(f"Total test samples: {len(samples)}")
    print(f"Constraint-satisfying: {len(cs)}")

    selected = []
    used_ids = set()

    # Stratum 1: highest reduction
    for s in cs[:n_high]:
        selected.append(('high_reduction', s))
        used_ids.add(s['sample_id'])

    # Stratum 2: near-boundary (highest constraint utilization)
    remaining = [s for s in cs if s['sample_id'] not in used_ids]
    for s in remaining:
        vm_util = s.get('vm_utilization', 0)
        comp_util = s.get('comp_utilization', 0)
        s['_max_util'] = max(vm_util, comp_util)
    remaining.sort(key=lambda x: x['_max_util'], reverse=True)
    for s in remaining[:n_boundary]:
        selected.append(('near_boundary', s))
        used_ids.add(s['sample_id'])

    # Stratum 3: random from remaining
    remaining2 = [s for s in cs if s['sample_id'] not in used_ids]
    rng = random.Random(seed)
    rng.shuffle(remaining2)
    for s in remaining2[:n_random]:
        selected.append(('random', s))
        used_ids.add(s['sample_id'])

    print(f"\nSelected {len(selected)} samples:")
    print(f"  High reduction: {sum(1 for g,_ in selected if g=='high_reduction')}")
    print(f"  Near boundary:  {sum(1 for g,_ in selected if g=='near_boundary')}")
    print(f"  Random:         {sum(1 for g,_ in selected if g=='random')}")

    return selected


def run_single_design(args_tuple):
    """Run FEA for one design (both baseline and optimized). Designed for multiprocessing."""
    idx, total, group, sample_info, data_dir, use_amg = args_tuple
    sid = sample_info['sample_id']
    
    try:
        # Load meta
        meta_path = Path(data_dir) / sid.zfill(5) / "meta.json"
        if not meta_path.exists():
            return {"sample_id": sid, "error": f"no meta.json at {meta_path}"}
        with open(meta_path) as f:
            meta = json.load(f)

        voxel_size = meta['voxel_size']
        mat_E = meta.get('E', 25e9)
        mat_nu = meta.get('nu', 0.2)
        mat_rho = meta.get('density', 2400.0)

        # Load optimized occupancy
        occ_path = Path(sample_info['_dir']) / "optimized_occ.npz"
        occ_opt = np.load(occ_path)['data'].astype(np.uint8)

        # Load baseline occupancy
        baseline_occ_path = Path(data_dir) / sid.zfill(5) / "occ.npz"
        occ_base = np.load(baseline_occ_path)['data'].astype(np.uint8)

        n_opt = int(occ_opt.sum())
        n_base = int(occ_base.sum())

        # Run BASELINE voxel FEA
        t0 = time.time()
        res_base = voxel_fea(occ_base, voxel_size, E=mat_E, nu=mat_nu, rho=mat_rho, use_amg=use_amg)
        dt_base = time.time() - t0

        if res_base is None:
            return {"sample_id": sid, "error": "baseline FEA failed (too few elements)"}

        # Run OPTIMIZED voxel FEA
        t0 = time.time()
        res_opt = voxel_fea(occ_opt, voxel_size, E=mat_E, nu=mat_nu, rho=mat_rho, use_amg=use_amg)
        dt_opt = time.time() - t0

        if res_opt is None:
            return {"sample_id": sid, "error": "optimized FEA failed (too few elements)"}

        # Compute relative metrics (THE KEY METRIC)
        comp_ratio = res_opt['compliance'] / res_base['compliance'] if res_base['compliance'] > 0 else float('inf')
        vm_ratio = res_opt['max_von_mises'] / res_base['max_von_mises'] if res_base['max_von_mises'] > 0 else float('inf')
        disp_ratio = res_opt['max_displacement'] / res_base['max_displacement'] if res_base['max_displacement'] > 0 else float('inf')

        # Surrogate predictions
        baseline_targets = sample_info.get('baseline_targets', {})

        r = {
            "sample_id": sid,
            "group": group,
            "volume_reduction_pct": sample_info.get('volume_reduction_pct', 0),
            "n_voxels_opt": n_opt,
            "n_voxels_base": n_base,
            # Surrogate predictions
            "surrogate_vm_mean": sample_info.get('pred_mean', [0,0,0])[0],
            "surrogate_comp_mean": sample_info.get('pred_mean', [0,0,0])[2],
            "surrogate_vm_cons": sample_info.get('vm_conservative', 0),
            "surrogate_comp_cons": sample_info.get('comp_conservative', 0),
            # Baseline voxel FEA
            "voxel_base_vm": res_base['max_von_mises'],
            "voxel_base_vm_p99": res_base['vm_p99'],
            "voxel_base_comp": res_base['compliance'],
            "voxel_base_disp": res_base['max_displacement'],
            "voxel_base_n_elem": res_base['n_elements'],
            "voxel_base_iters": res_base['solve_iters'],
            "voxel_base_time_s": dt_base,
            # Optimized voxel FEA
            "voxel_opt_vm": res_opt['max_von_mises'],
            "voxel_opt_vm_p99": res_opt['vm_p99'],
            "voxel_opt_vm_p95": res_opt['vm_p95'],
            "voxel_opt_comp": res_opt['compliance'],
            "voxel_opt_disp": res_opt['max_displacement'],
            "voxel_opt_n_elem": res_opt['n_elements'],
            "voxel_opt_iters": res_opt['solve_iters'],
            "voxel_opt_time_s": dt_opt,
            # CRITICAL: Same-method relative metrics
            "comp_ratio": comp_ratio,  # C_opt / C_base (must be ≤ 1.15)
            "vm_ratio": vm_ratio,
            "disp_ratio": disp_ratio,
            # Feasibility under same-method comparison
            "comp_ratio_ok": comp_ratio <= 1.15,
            "vm_ratio_ok": vm_ratio <= 2.0,  # stress should not more than double
            "total_time_s": dt_base + dt_opt,
            "amg_used": res_opt.get('amg_used', False),
        }

        return r

    except Exception as e:
        return {"sample_id": sid, "error": str(e), "traceback": traceback.format_exc()}


def main():
    parser = argparse.ArgumentParser(description="Same-method FEA validation (100 designs)")
    parser.add_argument("--n-high", type=int, default=30)
    parser.add_argument("--n-boundary", type=int, default=40)
    parser.add_argument("--n-random", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--no-amg", action="store_true", help="Disable AMG (use Jacobi)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing results")
    args = parser.parse_args()

    BATCH_DIR = Path("runs/v3/batch_results_all")
    DATA_DIR = Path("data/runs_real_128")
    RUNS_V3 = Path("runs/v3")
    OUT_FILE = RUNS_V3 / "fea_validation_100.json"

    # Load existing results if resuming
    existing = {}
    if args.resume and OUT_FILE.exists():
        with open(OUT_FILE) as f:
            data = json.load(f)
            for r in data:
                if 'error' not in r and 'comp_ratio' in r:
                    existing[r['sample_id']] = r
        print(f"Resuming: {len(existing)} valid results found")

    # Select samples
    selected = select_samples(
        str(BATCH_DIR), str(DATA_DIR),
        str(RUNS_V3 / "splits.json"),
        n_high=args.n_high,
        n_boundary=args.n_boundary,
        n_random=args.n_random,
        seed=args.seed,
    )

    # Filter out already completed
    to_run = []
    results = list(existing.values())
    for i, (group, s) in enumerate(selected):
        sid = s['sample_id']
        if sid in existing:
            continue
        to_run.append((i+1, len(selected), group, s, str(DATA_DIR), not args.no_amg))

    print(f"\nTo run: {len(to_run)} (skipping {len(existing)} already done)")
    use_amg = not args.no_amg
    print(f"AMG preconditioning: {'enabled' if use_amg else 'disabled'}")
    print(f"Workers: {args.workers}")
    sys.stdout.flush()

    t_start = time.time()

    if args.workers > 1:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_single_design, a): a for a in to_run}
            for future in as_completed(futures):
                r = future.result()
                sid = r['sample_id']
                if 'error' in r:
                    print(f"  FAILED {sid}: {r['error']}")
                else:
                    print(f"  Done {sid} ({r['group']}, "
                          f"red={r['volume_reduction_pct']:.1f}%, "
                          f"C_ratio={r['comp_ratio']:.4f}, "
                          f"ok={'Y' if r['comp_ratio_ok'] else 'N'}, "
                          f"t={r['total_time_s']:.0f}s)")
                results.append(r)
                # Save incrementally
                with open(OUT_FILE, 'w') as f:
                    json.dump(results, f, indent=2)
                sys.stdout.flush()
    else:
        # Sequential execution with detailed output
        for args_tuple in to_run:
            idx, total, group, s, data_dir_str, amg = args_tuple
            sid = s['sample_id']
            print(f"\n{'='*60}")
            print(f"[{idx}/{total}] Sample {sid} ({group}, "
                  f"reduction={s.get('volume_reduction_pct',0):.1f}%)")
            sys.stdout.flush()

            r = run_single_design(args_tuple)

            if 'error' in r:
                print(f"  FAILED: {r['error']}")
            else:
                print(f"  Baseline:  {r['n_voxels_base']:,} voxels, "
                      f"VM={r['voxel_base_vm']:.4g} Pa, "
                      f"C={r['voxel_base_comp']:.4g} J, "
                      f"t={r['voxel_base_time_s']:.1f}s")
                print(f"  Optimized: {r['n_voxels_opt']:,} voxels, "
                      f"VM={r['voxel_opt_vm']:.4g} Pa, "
                      f"C={r['voxel_opt_comp']:.4g} J, "
                      f"t={r['voxel_opt_time_s']:.1f}s")
                print(f"  C_opt/C_base = {r['comp_ratio']:.4f} "
                      f"({'PASS' if r['comp_ratio_ok'] else 'FAIL'} <= 1.15)")
                print(f"  VM_opt/VM_base = {r['vm_ratio']:.4f}")
                print(f"  Total time: {r['total_time_s']:.1f}s "
                      f"(AMG={'Y' if r.get('amg_used', False) else 'N'})")

            results.append(r)
            # Save incrementally
            with open(OUT_FILE, 'w') as f:
                json.dump(results, f, indent=2)
            sys.stdout.flush()
            gc.collect()

    elapsed = time.time() - t_start

    # Save final
    with open(OUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    valid = [r for r in results if 'error' not in r and 'comp_ratio' in r]
    errors = [r for r in results if 'error' in r]

    print(f"\n{'='*70}")
    print(f"SAME-METHOD FEA VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total: {len(results)}  Valid: {len(valid)}  Errors: {len(errors)}")
    print(f"Elapsed: {elapsed/3600:.2f} hours")

    if len(valid) < 3:
        print("Too few valid results for statistics")
        return

    from scipy.stats import spearmanr

    comp_ratios = np.array([r['comp_ratio'] for r in valid])
    vm_ratios = np.array([r['vm_ratio'] for r in valid])
    survival = np.array([r['comp_ratio_ok'] for r in valid])

    print(f"\n--- COMPLIANCE RATIO (C_opt/C_base) ---")
    print(f"  Mean:   {np.mean(comp_ratios):.4f}")
    print(f"  Median: {np.median(comp_ratios):.4f}")
    print(f"  Std:    {np.std(comp_ratios):.4f}")
    print(f"  Min:    {np.min(comp_ratios):.4f}")
    print(f"  Max:    {np.max(comp_ratios):.4f}")
    print(f"  <= 1.15: {survival.sum()}/{len(valid)} ({100*survival.mean():.1f}%)")

    print(f"\n--- VM STRESS RATIO (sigma_opt/sigma_base) ---")
    print(f"  Mean:   {np.mean(vm_ratios):.4f}")
    print(f"  Median: {np.median(vm_ratios):.4f}")
    print(f"  Max:    {np.max(vm_ratios):.4f}")

    # Per-group breakdown
    for group in ['high_reduction', 'near_boundary', 'random']:
        subset = [r for r in valid if r['group'] == group]
        if not subset:
            continue
        cr = np.array([r['comp_ratio'] for r in subset])
        sv = np.array([r['comp_ratio_ok'] for r in subset])
        print(f"\n--- {group.upper()} (n={len(subset)}) ---")
        print(f"  C_ratio: {np.mean(cr):.4f} +/- {np.std(cr):.4f}")
        print(f"  Survival: {sv.sum()}/{len(subset)} ({100*sv.mean():.1f}%)")

    # Spearman correlation: surrogate compliance vs voxel FEA compliance
    surr_comp = np.array([r['surrogate_comp_mean'] for r in valid])
    fea_comp = np.array([r['voxel_opt_comp'] for r in valid])
    mask = (surr_comp > 0) & (fea_comp > 0)
    if mask.sum() >= 3:
        rho_s, p_val = spearmanr(surr_comp[mask], fea_comp[mask])
        print(f"\n--- RANKING FIDELITY ---")
        print(f"  Spearman rho (surr vs voxel comp): {rho_s:.4f} (p={p_val:.2e})")

    # False positive analysis
    false_positives = [r for r in valid if not r['comp_ratio_ok']]
    if false_positives:
        print(f"\n--- FALSE POSITIVES (C_ratio > 1.15) ---")
        print(f"  Count: {len(false_positives)}")
        fp_reds = [r['volume_reduction_pct'] for r in false_positives]
        print(f"  Mean reduction: {np.mean(fp_reds):.1f}%")
        print(f"  Groups: ", end="")
        from collections import Counter
        gc_counts = Counter(r['group'] for r in false_positives)
        print(", ".join(f"{g}: {c}" for g, c in gc_counts.most_common()))

        # Top-5 worst violations
        false_positives.sort(key=lambda r: r['comp_ratio'], reverse=True)
        print(f"  Worst violations:")
        for fp in false_positives[:5]:
            print(f"    {fp['sample_id']}: C_ratio={fp['comp_ratio']:.4f}, "
                  f"red={fp['volume_reduction_pct']:.1f}%, "
                  f"group={fp['group']}")

    print(f"\n--- TIMING ---")
    times = [r['total_time_s'] for r in valid]
    print(f"  Per-design (opt+base): {np.mean(times):.0f}s +/- {np.std(times):.0f}s")
    print(f"  Total: {np.sum(times)/3600:.2f} hours")


if __name__ == "__main__":
    main()
