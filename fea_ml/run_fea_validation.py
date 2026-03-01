#!/usr/bin/env python3
"""
Ground-Truth FEA Validation of Optimized Designs (Optimized Solver)

Runs hex8 voxel FEA on a stratified sample of optimized geometries
and compares against surrogate predictions.

Sampling strategy (per reviewer):
  - 30 highest-reduction (35-45%)
  - 30 near-boundary (constraint utilization > 90%)
  - 30 random from remaining CS geometries

Uses ILU-preconditioned CG for ~10x faster convergence than Jacobi.
"""
import sys, os, json, argparse, time, gc, random
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg
from pathlib import Path


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

    # Centroid B-matrix
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


def voxel_fea_fast(occ, voxel_size, E=25e9, nu=0.20, rho=2400.0, g=9.81):
    """
    Hex8 FEA with ILU-preconditioned CG solver.
    Returns {max_von_mises, max_displacement, compliance, vm_p99, solve_iters}.
    """
    from scipy.ndimage import label as ndlabel

    # Keep only largest component touching BC face
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

    # BCs: fix at min a0
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

    # Jacobi-preconditioned CG (ILU factorization too slow for 100k+ DOFs)
    t_solve = time.time()
    solve_iters = [0]
    def callback(xk):
        solve_iters[0] += 1

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

    # Von Mises
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
    }


def select_samples(batch_dir, data_dir, splits_path, n_high=30, n_boundary=30, n_random=30, seed=42):
    """
    Select stratified samples for FEA validation.
    Returns list of dicts with sample info.
    """
    # Load splits
    with open(splits_path) as f:
        splits = json.load(f)
    test_ids = set(p.split('/')[-1] for p in splits['test'])

    # Load all test summaries
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

    # Constraint-satisfying only
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
    # Compute max utilization across VM, disp, compliance
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


def main():
    parser = argparse.ArgumentParser(description="Ground-truth FEA validation")
    parser.add_argument("--n-high", type=int, default=30)
    parser.add_argument("--n-boundary", type=int, default=30)
    parser.add_argument("--n-random", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume from existing results")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline FEA (faster)")
    args = parser.parse_args()

    BATCH_DIR = Path("runs/v3/batch_results_all")
    DATA_DIR = Path("data/runs_real_128")
    RUNS_V3 = Path("runs/v3")
    OUT_FILE = RUNS_V3 / "fea_validation_results.json"

    # Load existing results if resuming
    existing = {}
    if args.resume and OUT_FILE.exists():
        with open(OUT_FILE) as f:
            for r in json.load(f):
                existing[r['sample_id']] = r
        print(f"Resuming: {len(existing)} already completed")

    # Select samples
    selected = select_samples(
        str(BATCH_DIR), str(DATA_DIR),
        str(RUNS_V3 / "splits.json"),
        n_high=args.n_high,
        n_boundary=args.n_boundary,
        n_random=args.n_random,
        seed=args.seed,
    )

    results = list(existing.values())
    n_total = len(selected)
    n_skip = 0

    for i, (group, s) in enumerate(selected):
        sid = s['sample_id']

        if sid in existing:
            n_skip += 1
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{n_total}] Sample {sid} ({group}, "
              f"reduction={s.get('volume_reduction_pct',0):.1f}%)")

        # Load meta
        meta_path = DATA_DIR / sid.zfill(5) / "meta.json"
        if not meta_path.exists():
            print(f"  SKIP: no meta.json at {meta_path}")
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        # Load optimized occupancy
        occ_path = Path(s['_dir']) / "optimized_occ.npz"
        occ = np.load(occ_path)['data'].astype(np.uint8)

        voxel_size = meta['voxel_size']
        mat_E = meta.get('E', 25e9)
        mat_nu = meta.get('nu', 0.2)
        mat_rho = meta.get('density', 2400.0)

        print(f"  Filled voxels: {occ.sum():,}, voxel_size={voxel_size:.5f} m")

        # Also load baseline (original) occupancy for same-method comparison
        baseline_occ_path = DATA_DIR / sid.zfill(5) / "occ.npz"
        baseline_occ = None
        if not args.skip_baseline and baseline_occ_path.exists():
            baseline_occ = np.load(baseline_occ_path)['data'].astype(np.uint8)

        t0 = time.time()
        res = voxel_fea_fast(occ, voxel_size, E=mat_E, nu=mat_nu, rho=mat_rho)
        dt = time.time() - t0

        # Run baseline voxel FEA for relative comparison
        res_baseline = None
        dt_baseline = 0
        if baseline_occ is not None:
            print(f"  Running baseline FEA ({baseline_occ.sum():,} voxels)...")
            t0b = time.time()
            res_baseline = voxel_fea_fast(baseline_occ, voxel_size, E=mat_E, nu=mat_nu, rho=mat_rho)
            dt_baseline = time.time() - t0b
            gc.collect()

        if res is None:
            print(f"  FAILED ({dt:.1f}s)")
            continue

        # Constraint limits from SASTO
        comp_limit = s.get('comp_limit', 0)
        baseline = s.get('baseline_targets', {})
        baseline_vm = baseline.get('max_von_mises', 0)
        baseline_disp = baseline.get('max_displacement', 0)
        baseline_comp = baseline.get('compliance', 0)

        r = {
            "sample_id": sid,
            "group": group,
            "volume_reduction_pct": s.get('volume_reduction_pct', 0),
            "surrogate_vm_mean": s['pred_mean'][0],
            "surrogate_disp_mean": s['pred_mean'][1],
            "surrogate_comp_mean": s['pred_mean'][2],
            "surrogate_vm_cons": s.get('vm_conservative', 0),
            "surrogate_disp_cons": s.get('disp_conservative', 0),
            "surrogate_comp_cons": s.get('comp_conservative', 0),
            "fea_vm": res['max_von_mises'],
            "fea_vm_p99": res['vm_p99'],
            "fea_vm_p95": res['vm_p95'],
            "fea_disp": res['max_displacement'],
            "fea_comp": res['compliance'],
            "baseline_vm": baseline_vm,
            "baseline_disp": baseline_disp,
            "baseline_comp": baseline_comp,
            "comp_limit": comp_limit,
            "fea_n_elements": res['n_elements'],
            "fea_n_dof": res['n_dof'],
            "fea_solve_iters": res['solve_iters'],
            "fea_solve_time_s": res['solve_time_s'],
            "fea_total_time_s": dt,
        }

        # Add baseline voxel FEA results if available
        if res_baseline is not None:
            r["voxel_baseline_vm"] = res_baseline['max_von_mises']
            r["voxel_baseline_disp"] = res_baseline['max_displacement']
            r["voxel_baseline_comp"] = res_baseline['compliance']
            r["voxel_baseline_time_s"] = dt_baseline
            # Relative change: optimized vs baseline under same FEA
            r["rel_vm_change"] = res['max_von_mises'] / res_baseline['max_von_mises'] if res_baseline['max_von_mises'] > 0 else float('inf')
            r["rel_disp_change"] = res['max_displacement'] / res_baseline['max_displacement'] if res_baseline['max_displacement'] > 0 else float('inf')
            r["rel_comp_change"] = res['compliance'] / res_baseline['compliance'] if res_baseline['compliance'] > 0 else float('inf')

        # Check true feasibility (absolute)
        VM_LIMIT = 5.0e6   # 5 MPa
        DISP_LIMIT = 1.0   # 1 m
        r['fea_vm_ok'] = res['max_von_mises'] <= VM_LIMIT
        r['fea_disp_ok'] = res['max_displacement'] <= DISP_LIMIT
        r['fea_comp_ok'] = res['compliance'] <= comp_limit if comp_limit > 0 else True
        r['fea_all_ok'] = r['fea_vm_ok'] and r['fea_disp_ok'] and r['fea_comp_ok']

        # Relative feasibility check (same-method: voxel FEA opt vs voxel FEA base)
        if res_baseline is not None and res_baseline['compliance'] > 0:
            r['rel_comp_ok'] = res['compliance'] <= 1.15 * res_baseline['compliance']
        else:
            r['rel_comp_ok'] = None

        # Conservative bound coverage
        r['cons_covers_vm'] = s.get('vm_conservative', 0) >= res['max_von_mises']
        r['cons_covers_disp'] = s.get('disp_conservative', 0) >= res['max_displacement']
        r['cons_covers_comp'] = s.get('comp_conservative', 0) >= res['compliance']

        results.append(r)
        existing[sid] = r

        print(f"  Surrogate: VM={r['surrogate_vm_mean']:.4g}  "
              f"disp={r['surrogate_disp_mean']:.4g}  "
              f"comp={r['surrogate_comp_mean']:.4g}")
        print(f"  FEA:       VM={r['fea_vm']:.4g}  "
              f"disp={r['fea_disp']:.4g}  "
              f"comp={r['fea_comp']:.4g}")
        print(f"  Baseline:  VM={r['baseline_vm']:.4g}  "
              f"disp={r['baseline_disp']:.4g}  "
              f"comp={r['baseline_comp']:.4g}")
        print(f"  FEA OK: VM={'Y' if r['fea_vm_ok'] else 'N'}  "
              f"disp={'Y' if r['fea_disp_ok'] else 'N'}  "
              f"comp={'Y' if r['fea_comp_ok'] else 'N'}  "
              f"ALL={'Y' if r['fea_all_ok'] else 'N'}")
        print(f"  Time: {dt:.1f}s (solve: {res['solve_time_s']:.1f}s, "
              f"iters: {res['solve_iters']})")

        # Save incrementally
        with open(OUT_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        gc.collect()

    print(f"\n{'='*60}")
    print(f"Completed: {len(results)} | Skipped (resume): {n_skip}")

    # Save final
    with open(OUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    # ── Summary statistics ──
    if len(results) < 5:
        print("Too few results for statistics")
        return

    print_summary(results)


def print_summary(results):
    """Print comprehensive FEA validation summary."""
    from scipy.stats import spearmanr

    print("\n" + "="*70)
    print("GROUND-TRUTH FEA VALIDATION SUMMARY")
    print(f"N = {len(results)} optimized designs")
    print("="*70)

    # Per-group breakdown
    for group in ['high_reduction', 'near_boundary', 'random', 'ALL']:
        if group == 'ALL':
            subset = results
        else:
            subset = [r for r in results if r['group'] == group]
        if not subset:
            continue

        n = len(subset)
        n_ok = sum(1 for r in subset if r['fea_all_ok'])
        n_vm_ok = sum(1 for r in subset if r['fea_vm_ok'])
        n_disp_ok = sum(1 for r in subset if r['fea_disp_ok'])
        n_comp_ok = sum(1 for r in subset if r['fea_comp_ok'])

        print(f"\n--- {group.upper()} (n={n}) ---")
        print(f"  True FEA feasibility: {n_ok}/{n} ({100*n_ok/n:.1f}%)")
        print(f"    VM OK:   {n_vm_ok}/{n}  Disp OK: {n_disp_ok}/{n}  Comp OK: {n_comp_ok}/{n}")

        # Surrogate vs FEA metrics
        for name, sk, ck, fk in [
            ("VM stress", "surrogate_vm_mean", "surrogate_vm_cons", "fea_vm"),
            ("Displacement", "surrogate_disp_mean", "surrogate_disp_cons", "fea_disp"),
            ("Compliance", "surrogate_comp_mean", "surrogate_comp_cons", "fea_comp"),
        ]:
            sv = np.array([r[sk] for r in subset])
            cv = np.array([r[ck] for r in subset])
            fv = np.array([r[fk] for r in subset])
            mask = (fv > 0) & (sv > 0) & np.isfinite(fv) & np.isfinite(sv)
            if mask.sum() < 3:
                continue

            sv_m, cv_m, fv_m = sv[mask], cv[mask], fv[mask]
            ratio_mean = np.mean(sv_m / fv_m)
            ratio_std = np.std(sv_m / fv_m)
            mape = np.mean(np.abs(fv_m - sv_m) / fv_m) * 100
            bias = np.mean((sv_m - fv_m) / fv_m) * 100
            rho_s, _ = spearmanr(sv_m, fv_m)

            # Conservative coverage
            n_covered = np.sum(cv_m >= fv_m)

            print(f"  {name}:")
            print(f"    Surr/FEA ratio: {ratio_mean:.3f} ± {ratio_std:.3f}")
            print(f"    MAPE: {mape:.1f}%  Bias: {bias:+.1f}%")
            print(f"    Spearman ρ: {rho_s:.3f}")
            print(f"    Conservative coverage (μ+kσ ≥ FEA): {n_covered}/{mask.sum()} "
                  f"({100*n_covered/mask.sum():.0f}%)")

    # Worst-case violations
    print(f"\n--- WORST-CASE ANALYSIS ---")
    vm_violations = [(r['sample_id'], r['fea_vm'], r['surrogate_vm_cons'])
                     for r in results if not r['fea_vm_ok']]
    comp_violations = [(r['sample_id'], r['fea_comp'], r['comp_limit'])
                       for r in results if not r['fea_comp_ok']]

    if vm_violations:
        print(f"  VM violations: {len(vm_violations)}")
        for sid, fv, cv in sorted(vm_violations, key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {sid}: FEA={fv:.4g} Pa (limit=5e6)")
    else:
        print(f"  VM violations: 0")

    if comp_violations:
        print(f"  Compliance violations: {len(comp_violations)}")
        for sid, fv, lim in sorted(comp_violations, key=lambda x: x[1]/x[2] if x[2]>0 else 0, reverse=True)[:5]:
            print(f"    {sid}: FEA={fv:.4g} J (limit={lim:.4g}, ratio={fv/lim:.3f})")
    else:
        print(f"  Compliance violations: 0")

    # FEA timing
    times = [r['fea_total_time_s'] for r in results]
    print(f"\n--- FEA TIMING ---")
    print(f"  Mean: {np.mean(times):.1f}s  Median: {np.median(times):.1f}s")
    print(f"  Min: {np.min(times):.1f}s  Max: {np.max(times):.1f}s")
    print(f"  Total: {np.sum(times)/3600:.1f} hours")


if __name__ == "__main__":
    main()
