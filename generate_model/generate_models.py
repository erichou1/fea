import sys
import torch
import numpy as np
import json
import trimesh
from pathlib import Path

class Plan2Scene3D:
    """Generate 3D models"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.input_dir = Path("./cubicasa_output")
        self.output_dir = Path("../final_3d_models")
        self.output_dir.mkdir(exist_ok=True)
        
        self.scale = 0.01
        self.wall_height = 2.7
    
    def create_3d(self, json_path):
        """Generate 3D model"""
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
            floor_faces = [[0, i, i+1] for i in range(1, n-1)]
            
            if floor_faces:
                meshes.append(trimesh.Trimesh(floor_verts, floor_faces, process=False))
            
            # Walls
            for i in range(n):
                j = (i + 1) % n
                wall_verts = np.array([
                    [points[i,0], points[i,1], 0],
                    [points[j,0], points[j,1], 0],
                    [points[j,0], points[j,1], self.wall_height],
                    [points[i,0], points[i,1], self.wall_height]
                ])
                meshes.append(trimesh.Trimesh(wall_verts, [[0,1,2],[0,2,3]], process=False))
        
        if meshes:
            combined = trimesh.util.concatenate(meshes)
            combined.merge_vertices()
            return combined
        return None
    
    def process_all(self):
        """Process all"""
        json_files = list(self.input_dir.glob("*.json"))
        
        if not json_files:
            print(f"✗ No JSON files in {self.input_dir.resolve()}")
            return
        
        print(f"{'='*70}")
        print(f"3D MODEL GENERATION")
        print(f"{'='*70}")
        print(f"Processing {len(json_files)} models...\n")
        
        success = 0
        for i, json_file in enumerate(json_files, 1):
            try:
                mesh = self.create_3d(json_file)
                if mesh:
                    house_id = json_file.stem
                    
                    # Export using proper method
                    obj_path = self.output_dir / f"{house_id}.obj"
                    stl_path = self.output_dir / f"{house_id}.stl"
                    ply_path = self.output_dir / f"{house_id}.ply"
                    
                    mesh.export(str(obj_path))
                    mesh.export(str(stl_path))
                    mesh.export(str(ply_path))
                    
                    success += 1
            except Exception as e:
                print(f"  ✗ {json_file.stem}: {e}")
            
            if i % 20 == 0:
                print(f"  Progress: {i}/{len(json_files)} ({success} successful)")
        
        print(f"\n{'='*70}")
        print(f"COMPLETE: {success}/{len(json_files)}")
        print(f"Output: {self.output_dir.resolve()}")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    Plan2Scene3D().process_all()
