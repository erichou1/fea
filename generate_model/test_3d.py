import sys
sys.path.insert(0, './plan2scene/code/src')

import torch
import numpy as np
import json
import trimesh
from pathlib import Path

class Plan2Scene3DGenerator:
    """Generate 3D models from CubiCasa output using Plan2Scene"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}\n")
        
        # Your actual paths
        self.texture_synth_conf = Path("./plan2scene/conf/plan2scene/texture_synth_conf.yml")
        self.texture_synth_weights = Path("./plan2scene/checkpoints/loss-7.67493-epoch-750.ckpt")
        self.texture_prop_conf = Path("./plan2scene/conf/plan2scene/texture_prop_conf.json")
        self.texture_prop_weights = Path("./plan2scene/checkpoints/loss-0.51442-epoch-250.ckpt")
        
        # Input/Output
        self.input_dir = Path("./cubicasa_output")  # CubiCasa JSON outputs
        self.output_dir = Path("./final_3d_models")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.load_plan2scene_models()
        
        # 3D parameters
        self.scale = 0.01
        self.wall_height = 2.7
    
    def load_plan2scene_models(self):
        """Load Plan2Scene texture models"""
        print("Loading Plan2Scene models...")
        
        # Check if files exist
        if not self.texture_synth_weights.exists():
            print(f"  ⚠ Texture synth weights not found: {self.texture_synth_weights}")
        if not self.texture_prop_weights.exists():
            print(f"  ⚠ Texture prop weights not found: {self.texture_prop_weights}")
        
        try:
            self.texture_synth = torch.load(
                self.texture_synth_weights,
                map_location=self.device,
                weights_only=False
            )
            print(f"  ✓ Texture synthesis loaded")
            
            self.texture_prop = torch.load(
                self.texture_prop_weights,
                map_location=self.device,
                weights_only=False
            )
            print(f"  ✓ Texture propagation loaded")
            
            self.models_loaded = True
        except Exception as e:
            print(f"  ⚠ Plan2Scene models not loaded: {e}")
            print("  Continuing with geometry only")
            self.models_loaded = False
        
        print()
    
    def create_3d_from_json(self, json_path):
        """Generate 3D model from CubiCasa JSON"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        rooms = data.get('rooms', [])
        if not rooms:
            return None
        
        meshes = []
        
        for room in rooms:
            vertices_2d = room['vertices']
            if len(vertices_2d) < 3:
                continue
            
            points = np.array(vertices_2d) * self.scale
            n = len(points)
            
            # Floor
            floor_verts = np.column_stack([points[:, 0], points[:, 1], np.zeros(n)])
            floor_faces = []
            for i in range(1, n - 1):
                floor_faces.append([0, i, i + 1])
            
            if floor_faces:
                floor_mesh = trimesh.Trimesh(
                    vertices=floor_verts,
                    faces=np.array(floor_faces),
                    process=False
                )
                meshes.append(floor_mesh)
            
            # Walls (open top for viewing)
            for i in range(n):
                j = (i + 1) % n
                wall_verts = np.array([
                    [points[i, 0], points[i, 1], 0],
                    [points[j, 0], points[j, 1], 0],
                    [points[j, 0], points[j, 1], self.wall_height],
                    [points[i, 0], points[i, 1], self.wall_height]
                ])
                wall_faces = np.array([[0, 1, 2], [0, 2, 3]])
                wall_mesh = trimesh.Trimesh(
                    vertices=wall_verts,
                    faces=wall_faces,
                    process=False
                )
                meshes.append(wall_mesh)
        
        if meshes:
            combined = trimesh.util.concatenate(meshes)
            combined.merge_vertices()
            return combined
        
        return None
    
    def process_house(self, json_path):
        """Process single house"""
        house_id = json_path.stem
        
        try:
            mesh = self.create_3d_from_json(json_path)
            
            if mesh is None:
                return False
            
            # Save in multiple formats
            for fmt in ['obj', 'stl', 'ply']:
                output_path = self.output_dir / f"{house_id}.{fmt}"
                mesh.export(output_path)
            
            # Save stats
            stats = {
                'house_id': house_id,
                'vertices': int(len(mesh.vertices)),
                'faces': int(len(mesh.faces)),
                'bounds': mesh.bounds.tolist()
            }
            
            stats_path = self.output_dir / f"{house_id}_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"  ✗ {house_id}: {e}")
            return False
    
    def process_all(self):
        """Process all vectorized floor plans"""
        json_files = list(self.input_dir.glob("*.json"))
        
        if not json_files:
            print(f"✗ No JSON files in {self.input_dir.resolve()}")
            print("Run CubiCasa vectorization first!")
            return
        
        print(f"{'='*70}")
        print(f"PLAN2SCENE 3D GENERATION")
        print(f"{'='*70}")
        print(f"Input: {self.input_dir.resolve()}")
        print(f"Output: {self.output_dir.resolve()}")
        print(f"Processing {len(json_files)} houses...\n")
        
        success = 0
        for i, json_file in enumerate(json_files, 1):
            if self.process_house(json_file):
                success += 1
            
            if i % 20 == 0:
                print(f"  Progress: {i}/{len(json_files)} ({success} successful)")
        
        print(f"\n{'='*70}")
        print(f"COMPLETE: {success}/{len(json_files)} 3D models generated")
        print(f"Output: {self.output_dir.resolve()}")
        print(f"{'='*70}\n")
        print("Models saved as:")
        print("  - .obj (Blender, Windows 3D Viewer)")
        print("  - .stl (3D printing)")
        print("  - .ply (MeshLab)")

def main():
    generator = Plan2Scene3DGenerator()
    generator.process_all()

if __name__ == "__main__":
    main()
