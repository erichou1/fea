#!/usr/bin/env python3
"""Compare BC configurations: a0_min (left wall) vs a2_min (foundation)."""
import json, numpy as np, time, sys, gc
sys.path.insert(0, ".")
from scipy import sparse
from scipy.sparse.linalg import cg

from validate_fea_ground_truth import hex8_Ke_and_B

sample = "00000"
p = f"data/runs_real_128/{sample}"

with open(f"{p}/meta.json") as f:
    meta = json.load(f)
with open(f"{p}/targets.json") as f:
    targets = json.load(f)

occ = np.load(f"{p}/occ.npz")["data"].astype(np.uint8)

print(f"Sample {sample}")
print(f"  Known targets: VM={targets['max_von_mises']:.4g}, "
      f"disp={targets['max_displacement']:.4g}, comp={targets['compliance']:.4g}")
print(f"  Voxel size: {meta['voxel_size']:.5f} m")
print(f"  Elements: {occ.sum()}")

# Element setup
voxel_size = meta['voxel_size']
E, nu, rho = meta['E'], meta['nu'], meta['density']
dx = dy = dz = voxel_size
Ke, fe_grav_ref, B_c, Dmat = hex8_Ke_and_B(E, nu, dx, dy, dz)
fe_grav_elem = fe_grav_ref * (-rho * 9.81)

a0, a1, a2 = np.nonzero(occ)
n_elem = len(a0)
D0, D1, D2 = occ.shape
N0, N1, N2 = D0+1, D1+1, D2+1

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

elem_nodes_global = np.stack([n_0,n_1,n_2,n_3,n_4,n_5,n_6,n_7], axis=1)
unique_nodes = np.unique(elem_nodes_global)
n_nodes = len(unique_nodes)
n_dof = n_nodes * 3

node_compact = np.empty(unique_nodes.max()+1, dtype=np.int32)
node_compact[unique_nodes] = np.arange(n_nodes, dtype=np.int32)
elem_nodes = node_compact[elem_nodes_global]

elem_dofs = np.empty((n_elem, 24), dtype=np.int32)
for j in range(8):
    elem_dofs[:, 3*j] = 3*elem_nodes[:, j]
    elem_dofs[:, 3*j+1] = 3*elem_nodes[:, j] + 1
    elem_dofs[:, 3*j+2] = 3*elem_nodes[:, j] + 2

# Assembly
print("  Assembling...")
row_idx = np.repeat(elem_dofs[:,:,np.newaxis], 24, axis=2)
col_idx = np.repeat(elem_dofs[:,np.newaxis,:], 24, axis=1)
val = np.broadcast_to(Ke[np.newaxis,:,:], (n_elem,24,24)).copy()
K = sparse.coo_matrix((val.ravel(), (row_idx.ravel(), col_idx.ravel())),
                       shape=(n_dof, n_dof)).tocsr()
del row_idx, col_idx, val; gc.collect()

f_global = np.zeros(n_dof)
np.add.at(f_global, elem_dofs.ravel(),
          np.broadcast_to(fe_grav_elem[np.newaxis,:], (n_elem,24)).ravel())

print(f"  Assembly done. nnz={K.nnz}")

# Precompute node coordinates in voxel space
a0_of_node = unique_nodes // (N1*N2)
a1_of_node = (unique_nodes % (N1*N2)) // N2
a2_of_node = unique_nodes % N2

# Stats
print(f"\n  Structure extent:")
print(f"    a0: [{a0.min()}, {a0.max()}]  (phys x)")
print(f"    a1: [{a1.min()}, {a1.max()}]  (phys y)")
print(f"    a2: [{a2.min()}, {a2.max()}]  (phys z=height)")

def solve_with_bc(bc_name, bc_axis_idx, bc_side="min"):
    """Solve with BCs on a specific face."""
    if bc_axis_idx == 0:
        node_coords = a0_of_node
        struct_min = a0.min()
        struct_max = a0.max()
    elif bc_axis_idx == 1:
        node_coords = a1_of_node
        struct_min = a1.min()
        struct_max = a1.max()
    else:
        node_coords = a2_of_node
        struct_min = a2.min()
        struct_max = a2.max()
    
    if bc_side == "min":
        bc_val = struct_min
    else:
        bc_val = struct_max + 1  # +1 because nodes go one beyond elements
    
    bc_compact = np.where(node_coords == bc_val)[0]
    bc_dofs = np.concatenate([3*bc_compact, 3*bc_compact+1, 3*bc_compact+2])
    bc_dofs.sort()
    
    print(f"\n  BC: {bc_name} (axis {bc_axis_idx}, {bc_side})")
    print(f"    Fixed nodes: {len(bc_compact)}, Fixed DOFs: {len(bc_dofs)}")
    
    if len(bc_dofs) == 0:
        print("    No BCs - skipping")
        return None
    
    free_mask = np.ones(n_dof, dtype=bool)
    free_mask[bc_dofs] = False
    free_dofs = np.where(free_mask)[0]
    
    K_ff = K[free_dofs][:, free_dofs]
    f_f = f_global[free_dofs]
    
    diag_K = K_ff.diagonal().copy()
    diag_K[diag_K == 0] = 1.0
    M = sparse.diags(1.0/diag_K, format="csr")
    
    iter_count = [0]
    def cb(xk):
        iter_count[0] += 1
    
    t0 = time.time()
    u_f, info = cg(K_ff, f_f, M=M, rtol=1e-8, maxiter=50000, callback=cb)
    dt = time.time() - t0
    
    # Residual
    r = K_ff @ u_f - f_f
    rel_res = np.linalg.norm(r) / np.linalg.norm(f_f)
    
    u_global = np.zeros(n_dof)
    u_global[free_dofs] = u_f
    
    u3 = u_global.reshape(-1, 3)
    u_mag = np.linalg.norm(u3, axis=1)
    max_disp = float(np.max(u_mag))
    compliance = float(u_global @ f_global)
    
    # Stresses
    u_elem = u_global[elem_dofs]
    strain_all = B_c @ u_elem.T
    stress_all = Dmat @ strain_all
    sxx, syy, szz = stress_all[0], stress_all[1], stress_all[2]
    syz, sxz, sxy = stress_all[3], stress_all[4], stress_all[5]
    vm_sq = 0.5*((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2) + 3*(sxy**2+syz**2+sxz**2)
    vm = np.sqrt(np.maximum(vm_sq, 0.0))
    max_vm = float(np.max(vm))
    
    print(f"    CG: info={info}, iters={iter_count[0]}, time={dt:.1f}s")
    print(f"    Residual ||Ku-f||/||f||: {rel_res:.3e}")
    print(f"    VM:   {max_vm:.4g} Pa  (target: {targets['max_von_mises']:.4g}, ratio: {max_vm/targets['max_von_mises']:.3f})")
    print(f"    Disp: {max_disp:.4g} m  (target: {targets['max_displacement']:.4g}, ratio: {max_disp/targets['max_displacement']:.3f})")
    print(f"    Comp: {compliance:.4g} J  (target: {targets['compliance']:.4g}, ratio: {compliance/targets['compliance']:.3f})")
    return max_disp

# Try different BC configurations
solve_with_bc("a0_min (phys x-min = left wall)", 0, "min")
solve_with_bc("a2_min (phys z-min = foundation)", 2, "min")
solve_with_bc("a0_min + a2_min (left + foundation)", -1, "both")

# For the combined BC case
print(f"\n  BC: a0_min + a2_min (both faces)")
bc_a0 = np.where(a0_of_node == a0.min())[0]
bc_a2 = np.where(a2_of_node == a2.min())[0]
bc_both = np.unique(np.concatenate([bc_a0, bc_a2]))
bc_dofs_both = np.concatenate([3*bc_both, 3*bc_both+1, 3*bc_both+2])
bc_dofs_both.sort()
print(f"    Fixed nodes: {len(bc_both)}, Fixed DOFs: {len(bc_dofs_both)}")

free_mask2 = np.ones(n_dof, dtype=bool)
free_mask2[bc_dofs_both] = False
free_dofs2 = np.where(free_mask2)[0]

K_ff2 = K[free_dofs2][:, free_dofs2]
f_f2 = f_global[free_dofs2]

diag_K2 = K_ff2.diagonal().copy()
diag_K2[diag_K2 == 0] = 1.0
M2 = sparse.diags(1.0/diag_K2, format="csr")

ic2 = [0]
def cb2(xk): ic2[0] += 1
t0 = time.time()
u_f2, info2 = cg(K_ff2, f_f2, M=M2, rtol=1e-8, maxiter=50000, callback=cb2)
dt2 = time.time() - t0

r2 = K_ff2 @ u_f2 - f_f2
rel_res2 = np.linalg.norm(r2) / np.linalg.norm(f_f2)

u_global2 = np.zeros(n_dof)
u_global2[free_dofs2] = u_f2

u32 = u_global2.reshape(-1,3)
u_mag2 = np.linalg.norm(u32, axis=1)
max_disp2 = float(np.max(u_mag2))
compliance2 = float(u_global2 @ f_global)

u_elem2 = u_global2[elem_dofs]
strain_all2 = B_c @ u_elem2.T
stress_all2 = Dmat @ strain_all2
s2 = stress_all2
vm_sq2 = 0.5*((s2[0]-s2[1])**2 + (s2[1]-s2[2])**2 + (s2[2]-s2[0])**2) + 3*(s2[5]**2+s2[3]**2+s2[4]**2)
vm2 = np.sqrt(np.maximum(vm_sq2, 0.0))
max_vm2 = float(np.max(vm2))

print(f"    CG: info={info2}, iters={ic2[0]}, time={dt2:.1f}s")
print(f"    Residual: {rel_res2:.3e}")
print(f"    VM:   {max_vm2:.4g} Pa  (ratio: {max_vm2/targets['max_von_mises']:.3f})")
print(f"    Disp: {max_disp2:.4g} m  (ratio: {max_disp2/targets['max_displacement']:.3f})")
print(f"    Comp: {compliance2:.4g} J  (ratio: {compliance2/targets['compliance']:.3f})")

print("\nDone!")
