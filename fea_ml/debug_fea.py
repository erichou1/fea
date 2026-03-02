#!/usr/bin/env python3
"""Quick FEA debug: profile where time is spent."""
import sys, os, time, gc
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "runs_real_128")
BATCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "v3", "batch_results_all")

sid = "04203"
occ_path = os.path.join(BATCH_DIR, sid, "optimized_occ.npz")
meta_path = os.path.join(DATA_DIR, sid, "meta.json")

import json
occ = np.load(occ_path)['data'].astype(np.uint8)
with open(meta_path) as f:
    meta = json.load(f)

voxel_size = meta['voxel_size']

print(f"Sample {sid}, elements={int(occ.sum()):,}, voxel_size={voxel_size:.5f}")

# Import FEA components
from run_fea_validation import hex8_Ke_and_B
from scipy import sparse
from scipy.sparse.linalg import cg, spilu, LinearOperator
from scipy.ndimage import label as ndlabel

t0 = time.time()

# Component cleanup
a0_min_raw = np.nonzero(occ)[0].min()
labeled, n_comp = ndlabel(occ)
print(f"[{time.time()-t0:.1f}s] Components: {n_comp}")
if n_comp > 1:
    bc_labels = set(np.unique(labeled[a0_min_raw, :, :])) - {0}
    keep_mask = np.isin(labeled, list(bc_labels))
    n_removed = int(occ.sum()) - int(keep_mask.sum())
    print(f"  Removing {n_removed} disconnected voxels")
    occ = (occ & keep_mask).astype(np.uint8)

dx = dy = dz = voxel_size
E, nu, rho, g = 25e9, 0.20, 2400.0, 9.81
Ke, fe_grav_ref, B_c, Dmat = hex8_Ke_and_B(E, nu, dx, dy, dz)
fe_grav_elem = fe_grav_ref * (-rho * g)
print(f"[{time.time()-t0:.1f}s] Ke computed")

a0, a1, a2 = np.nonzero(occ)
n_elem = len(a0)
print(f"[{time.time()-t0:.1f}s] Elements after cleanup: {n_elem:,}")

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
print(f"[{time.time()-t0:.1f}s] Nodes: {n_nodes:,}, DOFs: {n_dof:,}")

node_compact = np.empty(unique_nodes.max()+1, dtype=np.int32)
node_compact[unique_nodes] = np.arange(n_nodes, dtype=np.int32)
elem_nodes = node_compact[elem_nodes_global]

elem_dofs = np.empty((n_elem, 24), dtype=np.int32)
for j in range(8):
    elem_dofs[:, 3*j]   = 3*elem_nodes[:,j]
    elem_dofs[:, 3*j+1] = 3*elem_nodes[:,j]+1
    elem_dofs[:, 3*j+2] = 3*elem_nodes[:,j]+2

print(f"[{time.time()-t0:.1f}s] DOF mapping done")

# Assembly
row_idx = np.repeat(elem_dofs[:,:,np.newaxis], 24, axis=2)
col_idx = np.repeat(elem_dofs[:,np.newaxis,:], 24, axis=1)
val = np.broadcast_to(Ke[np.newaxis,:,:], (n_elem,24,24)).copy()
K = sparse.coo_matrix(
    (val.ravel(), (row_idx.ravel(), col_idx.ravel())),
    shape=(n_dof, n_dof)
).tocsr()
del row_idx, col_idx, val
print(f"[{time.time()-t0:.1f}s] Assembly done, K shape: {K.shape}, nnz: {K.nnz:,}")

f_global = np.zeros(n_dof, dtype=np.float64)
np.add.at(f_global, elem_dofs.ravel(),
          np.broadcast_to(fe_grav_elem[np.newaxis,:], (n_elem,24)).ravel())

# BCs
a0_of_node = unique_nodes // (N1*N2)
a0_min_struct = a0.min()
bc_compact = np.where(a0_of_node == a0_min_struct)[0]
bc_dofs = np.concatenate([3*bc_compact, 3*bc_compact+1, 3*bc_compact+2])
bc_dofs.sort()

free_mask = np.ones(n_dof, dtype=bool)
free_mask[bc_dofs] = False
free_dofs = np.where(free_mask)[0]
n_free = len(free_dofs)
print(f"[{time.time()-t0:.1f}s] BCs: {len(bc_dofs)} fixed, {n_free} free DOFs")

K_ff = K[free_dofs][:, free_dofs]
f_f = f_global[free_dofs]
del K
gc.collect()
print(f"[{time.time()-t0:.1f}s] K_ff extracted, shape: {K_ff.shape}")

# Try ILU
if n_free < 300000:
    print(f"[{time.time()-t0:.1f}s] Computing ILU factorization (n_free={n_free:,})...")
    try:
        K_ff_csc = K_ff.tocsc()
        print(f"[{time.time()-t0:.1f}s] Converted to CSC")
        ilu = spilu(K_ff_csc, drop_tol=1e-3, fill_factor=5)
        M_pre = LinearOperator((n_free, n_free), matvec=ilu.solve)
        print(f"[{time.time()-t0:.1f}s] ILU done!")
    except Exception as e:
        print(f"[{time.time()-t0:.1f}s] ILU failed: {e}, falling back to Jacobi")
        diag_K = K_ff.diagonal().copy()
        diag_K[diag_K == 0] = 1.0
        M_pre = sparse.diags(1.0/diag_K, format="csr")
else:
    print(f"[{time.time()-t0:.1f}s] Using Jacobi (n_free={n_free:,} > 300k)")
    diag_K = K_ff.diagonal().copy()
    diag_K[diag_K == 0] = 1.0
    M_pre = sparse.diags(1.0/diag_K, format="csr")

# CG solve
solve_iters = [0]
def callback(xk):
    solve_iters[0] += 1
    if solve_iters[0] % 100 == 0:
        print(f"  CG iter {solve_iters[0]}...")

print(f"[{time.time()-t0:.1f}s] Starting CG solve...")
u_f, info = cg(K_ff, f_f, M=M_pre, rtol=1e-5, maxiter=20000, callback=callback)
print(f"[{time.time()-t0:.1f}s] CG done! info={info}, iters={solve_iters[0]}")
print(f"Total wall-clock: {time.time()-t0:.1f}s")
