# inference.py - Inference and CAD Export
import torch
import numpy as np
from models import FloorPlanTo3DPipeline
from PIL import Image
import torchvision.transforms as transforms
import os

class Inference:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load model
        self.model = FloorPlanTo3DPipeline(num_vertices=2048)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_image(self, image_path):
        img = Image.open(image_path).convert('RGB')
        return self.transform(img).unsqueeze(0)

    def predict(self, floor_plan_paths, exterior_paths):
        """
        Args:
            floor_plan_paths: List of paths to floor plan images
            exterior_paths: List of paths to exterior images
        """
        # Load images
        floor_plans = [self.load_image(fp).to(self.device) for fp in floor_plan_paths]
        exteriors = [self.load_image(ext).to(self.device) for ext in exterior_paths]

        # Inference
        with torch.no_grad():
            outputs = self.model(floor_plans, exteriors)

        return outputs

    def export_to_obj(self, outputs, output_path='output.obj'):
        """Export to Wavefront OBJ format"""
        vertices = outputs['vertices'][0].cpu().numpy()
        materials = outputs['materials'][0].cpu().numpy()

        with open(output_path, 'w') as f:
            # Write vertices
            f.write("# 3D Model generated from floor plans\n")
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")

            # Simple face generation (triangulate neighboring vertices)
            # This is simplified - use proper mesh generation in production
            num_verts = len(vertices)
            for i in range(0, num_verts-2, 3):
                f.write(f"f {i+1} {i+2} {i+3}\n")

        print(f"Model exported to {output_path}")

        # Save material properties separately
        material_path = output_path.replace('.obj', '_materials.txt')
        with open(material_path, 'w') as f:
            f.write("Vertex Materials (density, youngs_modulus, poisson, thickness, type)\n")
            for i, mat in enumerate(materials):
                f.write(f"Vertex {i}: {mat}\n")

        print(f"Materials exported to {material_path}")

    def export_to_stl(self, outputs, output_path='output.stl'):
        """Export to STL format (requires trimesh)"""
        try:
            import trimesh

            vertices = outputs['vertices'][0].cpu().numpy()

            # Simple triangulation
            faces = []
            num_verts = len(vertices)
            for i in range(0, num_verts-2, 3):
                faces.append([i, i+1, i+2])

            mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))
            mesh.export(output_path)
            print(f"Model exported to {output_path}")

        except ImportError:
            print("trimesh not installed. Install: pip install trimesh")

    def check_structural_soundness(self, outputs):
        """Analyze structural soundness"""
        is_sound = outputs['is_structurally_sound'][0].item()
        structural_props = outputs['structural_properties'][0].cpu().numpy()

        print("\n=== Structural Analysis ===")
        print(f"Structurally Sound: {'YES' if is_sound > 0.5 else 'NO'} ({is_sound:.2%} confidence)")
        print(f"Max Stress: {structural_props[0]:.2f} MPa")
        print(f"Max Displacement: {structural_props[1]:.4f} m")
        print(f"Safety Factor: {structural_props[2]:.2f}")
        print(f"Stability Score: {structural_props[3]:.2f}")
        print(f"Resonance Frequency: {structural_props[4]:.2f} Hz")
        print(f"Deflection: {structural_props[5]:.4f} m")

        return is_sound > 0.5

if __name__ == '__main__':
    # Example usage
    inferencer = Inference('checkpoints/checkpoint_epoch_99.pth')

    floor_plans = ['data/test/house_001/floor_plans/floor_0.jpg',
                   'data/test/house_001/floor_plans/floor_1.jpg']
    exteriors = ['data/test/house_001/exteriors/front.jpg',
                 'data/test/house_001/exteriors/back.jpg']

    outputs = inferencer.predict(floor_plans, exteriors)

    inferencer.export_to_obj(outputs, 'house_001.obj')
    inferencer.export_to_stl(outputs, 'house_001.stl')
    inferencer.check_structural_soundness(outputs)
