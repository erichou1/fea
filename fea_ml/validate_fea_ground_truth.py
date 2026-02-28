#!/usr/bin/env python3
"""
Ground-Truth FEA Validation of Optimized Designs

Runs actual hex8 FEA on a stratified sample of optimized voxel geometries
and compares results with surrogate predictions.

Coordinate convention (from training data):
  array[a0, a1, a2]  →  physical (x, y, z)
  a0 → physical x,  a1 → physical y,  a2 → physical z (HEIGHT)

Original FEA setup:
  - Gravity: [0, 0, -ρg] = −z physical
  - Fixed BC: axis=0 side="min" = minimum physical x face

Element node ordering makes element-local-x = a2 direction = physical z,
element-local-y = a1 = physical y, element-local-z = a0 = physical x.
So: DOF 0 per node = phys z, DOF 1 = phys y, DOF 2 = phys x.

Usage:
    cd fea_ml
    python validate_fea_ground_truth.py [--n-samples 40]
"""
import sys, os, json, argparse, time, gc
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, cg, spilu, LinearOperator
from pathlib import Path


def hex8_Ke_and_B(E, nu, dx, dy, dz):
    """
    24×24 element stiffness matrix for a regular hex8 brick (dx×dy×dz).
    Also returns gravity load vector and centroid B-matrix.
    """
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2.0 * (1 + nu))
    D = np.zeros((6, 6), dtype=np.float64)
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2 * mu
    D[0, 1] = D[0, 2] = D[1, 0] = D[1, 2] = D[2, 0] = D[2, 1] = lam
    D[3, 3] = D[4, 4] = D[5, 5] = mu

    detJ = (dx * dy * dz) / 8.0
    inv2 = [2.0 / dx, 2.0 / dy, 2.0 / dz]

    gp = 1.0 / np.sqrt(3.0)
    gauss_pts = np.array([
        [-gp, -gp, -gp], [gp, -gp, -gp], [-gp, gp, -gp], [gp, gp, -gp],
        [-gp, -gp,  gp], [gp, -gp,  gp], [-gp, gp,  gp], [gp, gp,  gp],
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

        dNdx = np.array(inv2).reshape(3, 1) * dN

        B = np.zeros((6, 24), dtype=np.float64)
        for i in range(8):
            c = 3 * i
            B[0, c]     = dNdx[0, i]
            B[1, c + 1] = dNdx[1, i]
            B[2, c + 2] = dNdx[2, i]
            B[3, c + 1] = dNdx[2, i]
            B[3, c + 2] = dNdx[1, i]
            B[4, c]     = dNdx[2, i]
            B[4, c + 2] = dNdx[0, i]
            B[5, c]     = dNdx[1, i]
            B[5, c + 1] = dNdx[0, i]

        Ke += B.T @ D @ B * detJ

        N = np.array([
            (1-xi)*(1-eta)*(1-zeta), (1+xi)*(1-eta)*(1-zeta),
            (1+xi)*(1+eta)*(1-zeta), (1-xi)*(1+eta)*(1-zeta),
            (1-xi)*(1-eta)*(1+zeta), (1+xi)*(1-eta)*(1+zeta),
            (1+xi)*(1+eta)*(1+zeta), (1-xi)*(1+eta)*(1+zeta),
        ]) / 8.0

        # Gravity on DOF 0 per node (element-local-x = physical z = height)
        for i in range(8):
            fe_grav[3 * i] += N[i] * detJ

    # Centroid B-matrix
    dN_c = np.array([
        [-1,  1,  1, -1, -1,  1,  1, -1],
        [-1, -1,  1,  1, -1, -1,  1,  1],
        [-1, -1, -1, -1,  1,  1,  1,  1],
    ], dtype=np.float64) / 8.0
    dNdx_c = np.array(inv2).reshape(3, 1) * dN_c
    B_c = np.zeros((6, 24), dtype=np.float64)
    for i in range(8):
        c = 3 * i
        B_c[0, c]     = dNdx_c[0, i]
        B_c[1, c + 1] = dNdx_c[1, i]
        B_c[2, c + 2] = dNdx_c[2, i]
        B_c[3, c + 1] = dNdx_c[2, i]
        B_c[3, c + 2] = dNdx_c[1, i]
        B_c[4, c]     = dNdx_c[2, i]
        B_c[4, c + 2] = dNdx_c[0, i]
        B_c[5, c]     = dNdx_c[1, i]
        B_c[5, c + 1] = dNdx_c[0, i]

    return Ke, fe_grav, B_c, D


def voxel_fea(occ, voxel_size, E=25e9, nu=0.20, rho=2400.0, g=9.81):
    """
    Run hex8 FEA on a binary voxel occupancy grid.
    Vectorized assembly. Returns {max_von_mises, max_displacement, compliance}.
    Keeps only the largest connected component touching the BC face.
    """
    from scipy.ndimage import label as ndlabel

    # ── 0. Keep only the largest component connected to the BC face ──
    # BC face is at minimum a0. Find connected components, keep the one(s)
    # that touch the a0_min layer.
    a0_min_raw = np.nonzero(occ)[0].min() if occ.any() else 0
    labeled, n_comp = ndlabel(occ)
    if n_comp > 1:
        # Find which labels touch the a0_min face
        bc_labels = set(np.unique(labeled[a0_min_raw, :, :])) - {0}
        if bc_labels:
            keep_mask = np.isin(labeled, list(bc_labels))
            n_removed = int(occ.sum()) - int(keep_mask.sum())
            if n_removed > 0:
                occ = (occ & keep_mask).astype(np.uint8)
                print(f"    Removed {n_removed} disconnected voxels ({n_comp} components → {len(bc_labels)})")

    # dx = dy = dz = voxel_size (isotropic cubic voxels)
    dx = dy = dz = voxel_size

    Ke, fe_grav_ref, B_c, Dmat = hex8_Ke_and_B(E, nu, dx, dy, dz)
    # Gravity force: −ρg in DOF 0 direction (element-local-x = physical z = height)
    fe_grav_elem = fe_grav_ref * (-rho * g)

    # ── 1. Filled voxels ──
    a0, a1, a2 = np.nonzero(occ)   # a0=phys_x, a1=phys_y, a2=phys_z
    n_elem = len(a0)
    if n_elem < 5:
        return None

    print(f"    Elements: {n_elem:,}, voxel_size={voxel_size:.5f} m")

    D0, D1, D2 = occ.shape          # grid dimensions
    N0, N1, N2 = D0 + 1, D1 + 1, D2 + 1  # node grid dimensions

    # ── 2. Node indices per element (vectorized) ──
    # Node numbering: node at grid pos (i0,i1,i2) → index = i0*N1*N2 + i1*N2 + i2
    # Element at voxel (a0,a1,a2) has 8 corner nodes.
    # Node ordering must match hex8_Ke_and_B:
    #   n0: (a0,a1,a2)         = local (0,0,0)
    #   n1: (a0,a1,a2+1)       = local (1,0,0)   ← element-local-x = a2 = phys z
    #   n2: (a0,a1+1,a2+1)     = local (1,1,0)
    #   n3: (a0,a1+1,a2)       = local (0,1,0)
    #   n4: (a0+1,a1,a2)       = local (0,0,1)   ← element-local-z = a0 = phys x
    #   n5: (a0+1,a1,a2+1)     = local (1,0,1)
    #   n6: (a0+1,a1+1,a2+1)   = local (1,1,1)
    #   n7: (a0+1,a1+1,a2)     = local (0,1,1)
    def nidx(i0, i1, i2):
        return i0 * N1 * N2 + i1 * N2 + i2

    n_0 = nidx(a0,   a1,   a2)
    n_1 = nidx(a0,   a1,   a2+1)
    n_2 = nidx(a0,   a1+1, a2+1)
    n_3 = nidx(a0,   a1+1, a2)
    n_4 = nidx(a0+1, a1,   a2)
    n_5 = nidx(a0+1, a1,   a2+1)
    n_6 = nidx(a0+1, a1+1, a2+1)
    n_7 = nidx(a0+1, a1+1, a2)

    elem_nodes_global = np.stack([n_0, n_1, n_2, n_3, n_4, n_5, n_6, n_7], axis=1)

    # ── 3. Compact node numbering ──
    unique_nodes = np.unique(elem_nodes_global)
    n_nodes = len(unique_nodes)
    n_dof = n_nodes * 3

    node_compact = np.empty(unique_nodes.max() + 1, dtype=np.int32)
    node_compact[unique_nodes] = np.arange(n_nodes, dtype=np.int32)
    elem_nodes = node_compact[elem_nodes_global]

    print(f"    Nodes: {n_nodes:,}, DOFs: {n_dof:,}")

    # ── 4. Element DOF array ──
    elem_dofs = np.empty((n_elem, 24), dtype=np.int32)
    for j in range(8):
        elem_dofs[:, 3*j]     = 3 * elem_nodes[:, j]
        elem_dofs[:, 3*j + 1] = 3 * elem_nodes[:, j] + 1
        elem_dofs[:, 3*j + 2] = 3 * elem_nodes[:, j] + 2

    # ── 5. Vectorized assembly ──
    print("    Assembling...")
    t_asm = time.time()

    row_idx = np.repeat(elem_dofs[:, :, np.newaxis], 24, axis=2)
    col_idx = np.repeat(elem_dofs[:, np.newaxis, :], 24, axis=1)
    val = np.broadcast_to(Ke[np.newaxis, :, :], (n_elem, 24, 24)).copy()

    K = sparse.coo_matrix(
        (val.ravel(), (row_idx.ravel(), col_idx.ravel())),
        shape=(n_dof, n_dof),
    ).tocsr()
    del row_idx, col_idx, val
    gc.collect()

    f_global = np.zeros(n_dof, dtype=np.float64)
    np.add.at(f_global, elem_dofs.ravel(),
              np.broadcast_to(fe_grav_elem[np.newaxis, :], (n_elem, 24)).ravel())

    print(f"    Assembly: {time.time()-t_asm:.1f}s, nnz={K.nnz:,}")

    # ── 6. Boundary conditions ──
    # Original FEA: fix at axis=0 min → minimum physical x → minimum a0
    # Extract a0 component of each unique node's global index
    a0_of_node = unique_nodes // (N1 * N2)
    a0_min_struct = a0.min()   # min a0 among filled voxels

    bc_compact = np.where(a0_of_node == a0_min_struct)[0]
    bc_dofs = np.concatenate([3*bc_compact, 3*bc_compact+1, 3*bc_compact+2])
    bc_dofs.sort()

    print(f"    Fixed DOFs: {len(bc_dofs)} (a0={a0_min_struct} face, phys x-min)")
    if len(bc_dofs) == 0:
        print("    WARNING: No BCs!")
        return None

    # ── 7. Apply BCs via elimination and solve ──
    free_mask = np.ones(n_dof, dtype=bool)
    free_mask[bc_dofs] = False
    free_dofs = np.where(free_mask)[0]

    print(f"    Extracting submatrix ({len(free_dofs)} free DOFs)...")
    t_solve = time.time()

    # Efficient sparse submatrix extraction (row slice then column slice)
    K_ff = K[free_dofs][:, free_dofs]
    f_f = f_global[free_dofs]

    print(f"    Submatrix: nnz={K_ff.nnz:,}")

    # Always use CG with Jacobi preconditioner — direct solver LU factorization
    # is too memory-intensive for 100k+ DOF 3D elasticity problems.
    n_free = len(free_dofs)
    try:
        print(f"    CG solver (n={n_free:,})...")
        diag_K = K_ff.diagonal().copy()
        diag_K[diag_K == 0] = 1.0
        M = sparse.diags(1.0 / diag_K, format="csr")

        # Track convergence
        iter_count = [0]
        def callback(xk):
            iter_count[0] += 1
            if iter_count[0] % 500 == 0:
                print(f"      CG iter {iter_count[0]}...")

        u_f, info = cg(K_ff, f_f, M=M, rtol=1e-6, maxiter=50000, callback=callback)
        print(f"    CG finished: info={info}, iters={iter_count[0]}")
        if info != 0:
            print(f"    CG did not converge (info={info}), using partial result")
    except Exception as e:
        print(f"    Solver failed: {e}")
        return None

    u_global = np.zeros(n_dof, dtype=np.float64)
    u_global[free_dofs] = u_f

    print(f"    Solved in {time.time()-t_solve:.1f}s")

    del K, K_ff, f_f
    gc.collect()

    # ── 8. Post-process ──
    u3 = u_global.reshape(-1, 3)
    u_mag = np.linalg.norm(u3, axis=1)
    max_displacement = float(np.max(u_mag))

    compliance = float(u_global @ f_global)

    # Von Mises (vectorized)
    print("    Computing stresses...")
    u_elem = u_global[elem_dofs]          # (n_elem, 24)
    strain_all = B_c @ u_elem.T           # (6, n_elem)
    stress_all = Dmat @ strain_all        # (6, n_elem)

    sxx, syy, szz = stress_all[0], stress_all[1], stress_all[2]
    syz, sxz, sxy = stress_all[3], stress_all[4], stress_all[5]

    vm_sq = 0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2) \
            + 3.0 * (sxy**2 + syz**2 + sxz**2)
    vm = np.sqrt(np.maximum(vm_sq, 0.0))
    max_vm = float(np.max(vm))

    print(f"    VM={max_vm:.4g} Pa, disp={max_displacement:.4g} m, C={compliance:.4g} J")

    return {
        "max_von_mises": max_vm,
        "max_displacement": max_displacement,
        "compliance": compliance,
    }


# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=40)
    args = parser.parse_args()

    BATCH_DIR = Path("runs/v3/batch_results_all")
    DATA_DIR = Path("data/runs_real_128")
    RUNS_V3 = Path("runs/v3")

    # Load optimization summaries
    print("Loading optimization summaries...")
    summaries = []
    for sd in sorted(BATCH_DIR.iterdir()):
        summary_path = sd / "optimization_summary.json"
        occ_path = sd / "optimized_occ.npz"
        if not summary_path.exists() or not occ_path.exists():
            continue
        with open(summary_path) as f:
            s = json.load(f)
        if not s.get("success", False):
            continue
        s["_dir"] = str(sd)
        summaries.append(s)

    print(f"Found {len(summaries)} successful optimizations")

    # Stratify by volume reduction: pick from constraint-satisfying samples
    ok = [s for s in summaries if s.get("constraints_satisfied", False)]
    ok.sort(key=lambda x: x.get("volume_reduction_pct", 0))
    n = min(args.n_samples, len(ok))

    n_q = n // 4
    n_rem = n - 3 * n_q
    selected = []

    lo = [s for s in ok if 5 <= s.get("volume_reduction_pct", 0) <= 15]
    selected.extend(lo[:n_q])
    md = [s for s in ok if 15 < s.get("volume_reduction_pct", 0) <= 30]
    selected.extend(md[:n_q])
    hi = [s for s in ok if 30 < s.get("volume_reduction_pct", 0) <= 45]
    selected.extend(hi[:n_q])
    bnd = sorted(ok, key=lambda x: x.get("volume_reduction_pct", 0), reverse=True)
    selected.extend([s for s in bnd if s not in selected][:n_rem])

    print(f"\nSelected {len(selected)} samples:")
    print(f"  Low (5-15%):  {sum(1 for s in selected if 5<=s.get('volume_reduction_pct',0)<=15)}")
    print(f"  Med (15-30%): {sum(1 for s in selected if 15<s.get('volume_reduction_pct',0)<=30)}")
    print(f"  Hi (30-45%):  {sum(1 for s in selected if 30<s.get('volume_reduction_pct',0)<=45)}")

    results = []
    for i, s in enumerate(selected):
        sid = s["sample_id"]
        sd = Path(s["_dir"])
        print(f"\n[{i+1}/{len(selected)}] Sample {sid} "
              f"(reduction: {s.get('volume_reduction_pct',0):.1f}%)")

        # Load meta for per-sample voxel_size
        meta_path = DATA_DIR / str(sid).zfill(5) / "meta.json"
        if not meta_path.exists():
            print(f"    SKIP: no meta.json")
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        voxel_size = meta["voxel_size"]
        mat_E = meta.get("E", 25e9)
        mat_nu = meta.get("nu", 0.2)
        mat_rho = meta.get("density", 2400.0)

        occ = np.load(sd / "optimized_occ.npz")["data"].astype(np.uint8)

        t0 = time.time()
        res = voxel_fea(occ, voxel_size=voxel_size, E=mat_E, nu=mat_nu, rho=mat_rho)
        dt = time.time() - t0

        if res is None:
            print("    SKIPPED (FEA failed)")
            continue

        r = {
            "sample_id": sid,
            "volume_reduction_pct": s.get("volume_reduction_pct", 0),
            "voxel_size_m": voxel_size,
            "surrogate_vm": s["pred_mean"][0],
            "surrogate_disp": s["pred_mean"][1],
            "surrogate_comp": s["pred_mean"][2],
            "surrogate_vm_cons": s.get("vm_conservative", 0),
            "surrogate_disp_cons": s.get("disp_conservative", 0),
            "surrogate_comp_cons": s.get("comp_conservative", 0),
            "fea_vm": res["max_von_mises"],
            "fea_disp": res["max_displacement"],
            "fea_comp": res["compliance"],
            "fea_time_s": dt,
            "baseline_vm": s.get("baseline_targets", {}).get("max_von_mises", 0),
            "baseline_disp": s.get("baseline_targets", {}).get("max_displacement", 0),
            "baseline_comp": s.get("baseline_targets", {}).get("compliance", 0),
        }
        results.append(r)

        print(f"    Surr: VM={r['surrogate_vm']:.4g}  disp={r['surrogate_disp']:.4g}  "
              f"comp={r['surrogate_comp']:.4g}")
        print(f"    FEA:  VM={r['fea_vm']:.4g}  disp={r['fea_disp']:.4g}  "
              f"comp={r['fea_comp']:.4g}")
        print(f"    Base: VM={r['baseline_vm']:.4g}  disp={r['baseline_disp']:.4g}  "
              f"comp={r['baseline_comp']:.4g}")
        print(f"    Time: {dt:.1f}s")

        # Save incrementally
        with open(RUNS_V3 / "ground_truth_fea_validation.json", "w") as f:
            json.dump(results, f, indent=2)
        gc.collect()

    # Final save
    out = RUNS_V3 / "ground_truth_fea_validation.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results → {out}")

    # ── Summary ──
    if len(results) < 3:
        print("Too few results for statistics.")
        return

    from scipy.stats import spearmanr, pearsonr

    print("\n" + "=" * 70)
    print("GROUND-TRUTH FEA VALIDATION SUMMARY")
    print("=" * 70)

    for name, sk, fk, bk, unit in [
        ("Von Mises",  "surrogate_vm",  "fea_vm",  "baseline_vm",  "Pa"),
        ("Displacement","surrogate_disp","fea_disp","baseline_disp","m"),
        ("Compliance",  "surrogate_comp","fea_comp","baseline_comp","J"),
    ]:
        sv = np.array([r[sk] for r in results])
        fv = np.array([r[fk] for r in results])
        bv = np.array([r[bk] for r in results])

        mask = (fv > 0) & (sv > 0) & np.isfinite(fv) & np.isfinite(sv)
        if mask.sum() < 3:
            print(f"\n{name}: insufficient data")
            continue

        sv, fv, bv = sv[mask], fv[mask], bv[mask]

        ss_res = np.sum((fv - sv) ** 2)
        ss_tot = np.sum((fv - np.mean(fv)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-30)
        rho_s, pval = spearmanr(sv, fv)
        pr, _ = pearsonr(sv, fv)
        mape = np.mean(np.abs((fv - sv) / fv)) * 100
        ratio = sv / fv

        # Also compare FEA-optimized vs baseline (original FEA)
        mask_b = bv > 0
        if mask_b.sum() > 0:
            ratio_b = fv[mask_b] / bv[mask_b]
            fea_change = (1 - ratio_b.mean()) * 100
        else:
            fea_change = float('nan')

        print(f"\n{name} ({unit}):")
        print(f"  R²            = {r2:.4f}")
        print(f"  Pearson r     = {pr:.4f}")
        print(f"  Spearman ρ    = {rho_s:.4f} (p={pval:.2e})")
        print(f"  MAPE          = {mape:.1f}%")
        print(f"  Surr/FEA      = {np.mean(ratio):.3f} ± {np.std(ratio):.3f}")
        print(f"  Surr mean     = {np.mean(sv):.4g}")
        print(f"  FEA mean      = {np.mean(fv):.4g}")
        print(f"  FEA vs base   = {fea_change:+.1f}% change (opt vs original)")

    # Coverage analysis
    print("\n" + "-" * 40)
    print("CONSERVATIVE ESTIMATE COVERAGE (μ+kσ ≥ FEA):")
    for name, ck, fk in [
        ("Von Mises",  "surrogate_vm_cons",  "fea_vm"),
        ("Displacement","surrogate_disp_cons","fea_disp"),
        ("Compliance",  "surrogate_comp_cons","fea_comp"),
    ]:
        cv = np.array([r[ck] for r in results])
        fv = np.array([r[fk] for r in results])
        mask = (fv > 0) & (cv > 0)
        if mask.sum() < 3:
            continue
        covered = np.sum(cv[mask] >= fv[mask])
        total = mask.sum()
        print(f"  {name}: {covered}/{total} ({100*covered/total:.0f}%)")

    # True feasibility
    MAX_VM = 5.0e6
    MAX_DISP = 0.028
    n_vm = sum(1 for r in results if r["fea_vm"] <= MAX_VM)
    n_di = sum(1 for r in results if r["fea_disp"] <= MAX_DISP)
    n_ok = sum(1 for r in results if r["fea_vm"] <= MAX_VM and r["fea_disp"] <= MAX_DISP)
    N = len(results)
    print(f"\n  True FEA feasibility:")
    print(f"    VM ≤ 5 MPa:  {n_vm}/{N} ({100*n_vm/N:.0f}%)")
    print(f"    Disp ≤ L/360: {n_di}/{N} ({100*n_di/N:.0f}%)")
    print(f"    All OK:       {n_ok}/{N} ({100*n_ok/N:.0f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()
