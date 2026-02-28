#!/usr/bin/env python3
"""Quick test of the hex8 FEA implementation with simple known cases."""
import numpy as np
import time, sys
sys.path.insert(0, ".")
from validate_fea_ground_truth import hex8_Ke_and_B, voxel_fea
from scipy import sparse

# ============================================================
# Test 1: Element stiffness matrix symmetry and properties
# ============================================================
print("=" * 60)
print("Test 1: Element stiffness matrix properties")
E, nu, h = 25e9, 0.2, 0.01
Ke, fe_grav, B_c, D = hex8_Ke_and_B(E, nu, h, h, h)

print(f"  Ke shape: {Ke.shape}")
print(f"  Ke symmetric: {np.allclose(Ke, Ke.T, atol=1e-10)}")
print(f"  Ke positive semi-definite: {np.min(np.linalg.eigvalsh(Ke)):.6g}")

# Rigid body: Ke @ u_rigid should be zero
# Translation in x (DOF 0): u = [1,0,0, 1,0,0, ...]
u_trans = np.zeros(24)
u_trans[::3] = 1.0
f_trans = Ke @ u_trans
print(f"  Rigid translation DOF0: max |Ke*u| = {np.max(np.abs(f_trans)):.3e}")

# Translation in y (DOF 1)
u_trans_y = np.zeros(24)
u_trans_y[1::3] = 1.0
f_trans_y = Ke @ u_trans_y
print(f"  Rigid translation DOF1: max |Ke*u| = {np.max(np.abs(f_trans_y)):.3e}")

# Translation in z (DOF 2)
u_trans_z = np.zeros(24)
u_trans_z[2::3] = 1.0
f_trans_z = Ke @ u_trans_z
print(f"  Rigid translation DOF2: max |Ke*u| = {np.max(np.abs(f_trans_z)):.3e}")

# Gravity load: should integrate to ρ*g*V on DOF 0
rho_test, g_test = 2400.0, 9.81
fe_test = fe_grav * (-rho_test * g_test)
total_fe = np.sum(fe_test[::3])  # Only DOF 0 (gravity direction)
V = h**3
expected = -rho_test * g_test * V
print(f"\n  Total gravity force (DOF 0): {total_fe:.6g}")
print(f"  Expected: {expected:.6g}")
print(f"  Match: {np.isclose(total_fe, expected, rtol=1e-10)}")

# ============================================================
# Test 2: 3x3x3 solid cube, fixed at a0=0 face
# ============================================================
print("\n" + "=" * 60)
print("Test 2: 3x3x3 solid cube under gravity")
n_vox = 3
occ = np.zeros((n_vox + 10, n_vox + 10, n_vox + 10), dtype=np.uint8)
# Place cube at some offset to test coordinate handling
offset = 2
occ[offset:offset+n_vox, offset:offset+n_vox, offset:offset+n_vox] = 1

h_test = 0.01  # 1 cm voxels
result = voxel_fea(occ, voxel_size=h_test, E=25e9, nu=0.2, rho=2400.0)
if result:
    print(f"  VM:   {result['max_von_mises']:.4g} Pa")
    print(f"  Disp: {result['max_displacement']:.4g} m")
    print(f"  Comp: {result['compliance']:.4g} J")

# ============================================================
# Test 3: 5x1x1 cantilever beam under gravity, compare with analytical
# ============================================================
print("\n" + "=" * 60)
print("Test 3: 5x1x1 cantilever beam under gravity")
print("  (5 elements in a0 direction, 1 in a1, 1 in a2)")
occ2 = np.zeros((10, 10, 10), dtype=np.uint8)
# Beam along a0 direction, width in a1, height in a2
occ2[2:7, 4:5, 4:5] = 1  # 5 voxels in a0, 1 in a1, 1 in a2
h2 = 0.01  # 1 cm voxels

print(f"  Elements: {occ2.sum()}")
print(f"  Fixed at a0=2 (leftmost layer)")

result2 = voxel_fea(occ2, voxel_size=h2, E=25e9, nu=0.2, rho=2400.0)

if result2:
    # Analytical: cantilever beam under self-weight
    # Tip deflection δ = wL^4/(8EI) where w = ρgA, I = bd^3/12
    L = 5 * h2  # beam length
    b = h2  # width
    d = h2  # depth (in gravity direction = a2)
    w = 2400 * 9.81 * b * d  # N/m
    I = b * d**3 / 12
    delta_analytical = w * L**4 / (8 * 25e9 * I)
    
    print(f"  FEA VM:   {result2['max_von_mises']:.4g} Pa")
    print(f"  FEA Disp: {result2['max_displacement']:.4g} m")
    print(f"  FEA Comp: {result2['compliance']:.4g} J")
    print(f"  Analytical tip deflection (beam theory): {delta_analytical:.4g} m")
    print(f"  Ratio FEA/analytical: {result2['max_displacement']/delta_analytical:.4f}")

# ============================================================
# Test 4: Check residual for a moderate-size problem
# ============================================================
print("\n" + "=" * 60)
print("Test 4: Residual check on 10x10x10 cube")
occ3 = np.zeros((20, 20, 20), dtype=np.uint8)
occ3[3:13, 3:13, 3:13] = 1  # 10x10x10 cube

# Run FEA manually to check residual
from validate_fea_ground_truth import hex8_Ke_and_B
from scipy.ndimage import label as ndlabel
from scipy.sparse.linalg import cg

h3 = 0.01
dx = dy = dz = h3
Ke3, fe_grav_ref3, B_c3, D3 = hex8_Ke_and_B(25e9, 0.2, dx, dy, dz)
fe_grav_elem3 = fe_grav_ref3 * (-2400.0 * 9.81)

a0, a1, a2 = np.nonzero(occ3)
n_elem = len(a0)
D0, D1, D2 = occ3.shape
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

import gc
row_idx = np.repeat(elem_dofs[:,:,np.newaxis], 24, axis=2)
col_idx = np.repeat(elem_dofs[:,np.newaxis,:], 24, axis=1)
val = np.broadcast_to(Ke3[np.newaxis,:,:], (n_elem,24,24)).copy()
K = sparse.coo_matrix((val.ravel(), (row_idx.ravel(), col_idx.ravel())),
                       shape=(n_dof, n_dof)).tocsr()
del row_idx, col_idx, val; gc.collect()

f_global = np.zeros(n_dof)
np.add.at(f_global, elem_dofs.ravel(),
          np.broadcast_to(fe_grav_elem3[np.newaxis,:], (n_elem,24)).ravel())

# BCs at a0_min
a0_of_node = unique_nodes // (N1*N2)
a0_min = a0.min()
bc_compact = np.where(a0_of_node == a0_min)[0]
bc_dofs = np.concatenate([3*bc_compact, 3*bc_compact+1, 3*bc_compact+2])
bc_dofs.sort()

free_mask = np.ones(n_dof, dtype=bool)
free_mask[bc_dofs] = False
free_dofs = np.where(free_mask)[0]

K_ff = K[free_dofs][:, free_dofs]
f_f = f_global[free_dofs]

print(f"  Elements: {n_elem}, Nodes: {n_nodes}, DOFs: {n_dof}")
print(f"  Free DOFs: {len(free_dofs)}, BC DOFs: {len(bc_dofs)}")

# Solve with CG
diag_K = K_ff.diagonal().copy()
diag_K[diag_K == 0] = 1.0
M = sparse.diags(1.0/diag_K, format="csr")
u_f, info = cg(K_ff, f_f, M=M, rtol=1e-10, maxiter=50000)
print(f"  CG info: {info}")

# Check residual
r = K_ff @ u_f - f_f
rel_residual = np.linalg.norm(r) / np.linalg.norm(f_f)
print(f"  Relative residual ||Ku-f||/||f||: {rel_residual:.3e}")

u_global = np.zeros(n_dof)
u_global[free_dofs] = u_f

u3 = u_global.reshape(-1, 3)
u_mag = np.linalg.norm(u3, axis=1)
max_disp = np.max(u_mag)
compliance = u_global @ f_global

print(f"  Max displacement: {max_disp:.4g} m")
print(f"  Compliance: {compliance:.4g} J")
print(f"  Total force: {np.sum(f_global):.4g} N")
print(f"  Force DOF0: {np.sum(f_global[0::3]):.4g} N")
print(f"  Force DOF1: {np.sum(f_global[1::3]):.4g} N")
print(f"  Force DOF2: {np.sum(f_global[2::3]):.4g} N")

# Analytical estimate for a cantilever cube L=W=D=10h:
# δ ~ wL^4/(8EI), w = ρg*b*d, I = b*d^3/12
L_an = 10*h3
b_an = 10*h3
d_an = 10*h3  
w_an = 2400*9.81*b_an*d_an  # N/m
I_an = b_an*d_an**3/12
delta_an = w_an*L_an**4/(8*25e9*I_an)
print(f"\n  Analytical (beam): δ = {delta_an:.4g} m")
print(f"  FEA/analytical: {max_disp/delta_an:.4f}")

print("\nDone!")
