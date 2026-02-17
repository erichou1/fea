"""Convert optimized voxel grid to viewable STL file."""
import numpy as np
from pathlib import Path

# Load files
opt_dir = Path("runs/gb200_v1/optimization")
baseline_dir = Path("data/runs_real/00000")

# Load optimized occupancy
occ_opt = np.load(opt_dir / "optimized_occ.npz")["data"]
print(f"Optimized grid shape: {occ_opt.shape}")
print(f"Optimized voxels filled: {np.sum(occ_opt)}")

# Load baseline for comparison
occ_base = np.load(baseline_dir / "occ.npz")["data"]
print(f"Baseline voxels filled: {np.sum(occ_base)}")
print(f"Volume reduction: {1.0 - np.sum(occ_opt) / np.sum(occ_base):.1%}")

# Convert to STL using marching cubes
try:
    from skimage.measure import marching_cubes
    
    occ_float = occ_opt.astype(np.float32)
    if np.sum(occ_float) == 0:
        print("ERROR: Optimized grid is empty!")
    else:
        verts, faces, normals, _ = marching_cubes(occ_float, level=0.5)
        
        # Save as STL
        import struct
        stl_path = opt_dir / "optimized_house.stl"
        with open(stl_path, 'wb') as f:
            f.write(b'\0' * 80)  # header
            f.write(struct.pack('<I', len(faces)))
            for face in faces:
                v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normal = normal / norm
                f.write(struct.pack('<fff', *normal))
                f.write(struct.pack('<fff', *v0))
                f.write(struct.pack('<fff', *v1))
                f.write(struct.pack('<fff', *v2))
                f.write(struct.pack('<H', 0))
        
        print(f"\nSTL saved to: {stl_path}")
        print(f"Faces: {len(faces)}, Vertices: {len(verts)}")
        
        # Also save baseline as STL for comparison
        verts_b, faces_b, _, _ = marching_cubes(occ_base.astype(np.float32), level=0.5)
        stl_base = opt_dir / "baseline_house.stl"
        with open(stl_base, 'wb') as f:
            f.write(b'\0' * 80)
            f.write(struct.pack('<I', len(faces_b)))
            for face in faces_b:
                v0, v1, v2 = verts_b[face[0]], verts_b[face[1]], verts_b[face[2]]
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normal = normal / norm
                f.write(struct.pack('<fff', *normal))
                f.write(struct.pack('<fff', *v0))
                f.write(struct.pack('<fff', *v1))
                f.write(struct.pack('<fff', *v2))
                f.write(struct.pack('<H', 0))
        
        print(f"Baseline STL saved to: {stl_base}")
        print(f"\nOpen these files in any 3D viewer (Windows 3D Viewer, Blender, MeshLab, etc.)")

except ImportError:
    print("Installing scikit-image for marching cubes...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-image"])
    print("Installed! Re-run this script.")
