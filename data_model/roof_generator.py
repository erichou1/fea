# roof_generator.py - Automatic Roof Generation for Plan2Scene Models

import numpy as np
import trimesh
from scipy.spatial import ConvexHull
import json

class RoofGenerator:
    """Generate procedural roofs for building footprints"""

    def __init__(self, roof_type='hip', roof_height=3.0, overhang=0.5):
        """
        Args:
            roof_type: 'hip', 'gable', 'flat', or 'auto'
            roof_height: Height of roof peak above ceiling (meters)
            overhang: Roof overhang beyond walls (meters)
        """
        self.roof_type = roof_type
        self.roof_height = roof_height
        self.overhang = overhang

    def extract_building_footprint(self, scene_mesh):
        """Extract 2D building footprint from Plan2Scene 3D mesh"""
        vertices = scene_mesh.vertices

        # Get XY coordinates (ignore Z)
        xy_coords = vertices[:, :2]

        # Find convex hull of building footprint
        hull = ConvexHull(xy_coords)
        footprint = xy_coords[hull.vertices]

        # Get ceiling height (max Z coordinate)
        ceiling_height = vertices[:, 2].max()

        return footprint, ceiling_height

    def expand_footprint(self, footprint):
        """Expand footprint by overhang amount"""
        # Calculate centroid
        centroid = footprint.mean(axis=0)

        # Expand each point away from centroid
        expanded = []
        for point in footprint:
            direction = point - centroid
            dist = np.linalg.norm(direction)
            if dist > 0:
                direction = direction / dist
                expanded_point = point + direction * self.overhang
                expanded.append(expanded_point)
            else:
                expanded.append(point)

        return np.array(expanded)

    def generate_hip_roof(self, footprint, base_height):
        """Generate hip roof (slopes on all sides)"""
        vertices = []
        faces = []

        # Expand footprint for overhang
        expanded = self.expand_footprint(footprint)

        # Base vertices (at ceiling level)
        for point in expanded:
            vertices.append([point[0], point[1], base_height])

        # Calculate roof peak (center of footprint)
        center = expanded.mean(axis=0)
        peak_vertex = [center[0], center[1], base_height + self.roof_height]
        peak_idx = len(vertices)
        vertices.append(peak_vertex)

        # Create triangular faces from each edge to peak
        n = len(expanded)
        for i in range(n):
            # Triangle: base_edge[i] -> base_edge[i+1] -> peak
            faces.append([i, (i+1) % n, peak_idx])

        return np.array(vertices), np.array(faces)

    def generate_gable_roof(self, footprint, base_height):
        """Generate gable roof (ridge along longest axis)"""
        vertices = []
        faces = []

        # Expand footprint
        expanded = self.expand_footprint(footprint)

        # Find longest edge to determine ridge direction
        max_dist = 0
        ridge_idx1, ridge_idx2 = 0, 1

        n = len(expanded)
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(expanded[i] - expanded[j])
                if dist > max_dist:
                    max_dist = dist
                    ridge_idx1, ridge_idx2 = i, j

        # Base vertices
        for point in expanded:
            vertices.append([point[0], point[1], base_height])

        # Ridge vertices (at peak height)
        ridge1 = expanded[ridge_idx1]
        ridge2 = expanded[ridge_idx2]

        ridge1_peak = [ridge1[0], ridge1[1], base_height + self.roof_height]
        ridge2_peak = [ridge2[0], ridge2[1], base_height + self.roof_height]

        ridge1_idx = len(vertices)
        ridge2_idx = ridge1_idx + 1

        vertices.append(ridge1_peak)
        vertices.append(ridge2_peak)

        # Create roof faces
        # Connect base edges to ridge
        for i in range(n):
            next_i = (i + 1) % n

            # Determine which ridge point is closer
            dist1 = np.linalg.norm(expanded[i] - ridge1)
            dist2 = np.linalg.norm(expanded[i] - ridge2)

            if dist1 < dist2:
                # Closer to ridge1
                if i != ridge_idx1 and next_i != ridge_idx1:
                    faces.append([i, next_i, ridge1_idx])
            else:
                # Closer to ridge2
                if i != ridge_idx2 and next_i != ridge_idx2:
                    faces.append([i, next_i, ridge2_idx])

        # Connect the two ridge points
        faces.append([ridge_idx1, ridge_idx2, ridge1_idx])
        faces.append([ridge_idx2, ridge_idx1, ridge2_idx])

        return np.array(vertices), np.array(faces)

    def generate_flat_roof(self, footprint, base_height):
        """Generate flat roof (simple plane)"""
        vertices = []
        faces = []

        # Expand footprint
        expanded = self.expand_footprint(footprint)

        # Create vertices at roof height
        for point in expanded:
            vertices.append([point[0], point[1], base_height + 0.3])  # Slight height for drainage

        # Triangulate the roof surface
        n = len(expanded)
        center_idx = n

        # Add center vertex
        center = expanded.mean(axis=0)
        vertices.append([center[0], center[1], base_height + 0.3])

        # Create triangular faces
        for i in range(n):
            faces.append([i, (i+1) % n, center_idx])

        return np.array(vertices), np.array(faces)

    def generate_roof(self, scene_mesh):
        """Generate roof for a Plan2Scene mesh"""

        # Extract building footprint and height
        footprint, ceiling_height = self.extract_building_footprint(scene_mesh)

        print(f"Building footprint: {len(footprint)} points")
        print(f"Ceiling height: {ceiling_height:.2f}m")

        # Generate roof based on type
        if self.roof_type == 'hip':
            vertices, faces = self.generate_hip_roof(footprint, ceiling_height)
        elif self.roof_type == 'gable':
            vertices, faces = self.generate_gable_roof(footprint, ceiling_height)
        elif self.roof_type == 'flat':
            vertices, faces = self.generate_flat_roof(footprint, ceiling_height)
        elif self.roof_type == 'auto':
            # Auto-detect: use gable for rectangular, hip for complex shapes
            if len(footprint) == 4:
                vertices, faces = self.generate_gable_roof(footprint, ceiling_height)
            else:
                vertices, faces = self.generate_hip_roof(footprint, ceiling_height)
        else:
            raise ValueError(f"Unknown roof type: {self.roof_type}")

        # Create roof mesh
        roof_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        print(f"✓ Generated {self.roof_type} roof: {len(vertices)} vertices, {len(faces)} faces")

        return roof_mesh

    def combine_with_building(self, building_mesh, roof_mesh):
        """Combine building and roof into single mesh"""
        combined = trimesh.util.concatenate([building_mesh, roof_mesh])

        print(f"✓ Combined building + roof: {len(combined.vertices)} vertices")

        return combined


def add_roof_to_plan2scene_output(scene_json_path, output_path, roof_type='hip'):
    """
    Add roof to Plan2Scene generated 3D model

    Args:
        scene_json_path: Path to Plan2Scene .scene.json file
        output_path: Output path for final model with roof
        roof_type: 'hip', 'gable', 'flat', or 'auto'
    """
    print(f"\nAdding {roof_type} roof to Plan2Scene model...")
    print("="*60)

    # Load Plan2Scene output
    # Note: Plan2Scene outputs scene.json which needs to be converted to mesh
    # For now, assume we have the mesh

    # Load building mesh (you'll get this from Plan2Scene)
    building_mesh = trimesh.load(scene_json_path.replace('.scene.json', '.obj'))

    # Generate roof
    roof_gen = RoofGenerator(roof_type=roof_type, roof_height=3.0, overhang=0.5)
    roof_mesh = roof_gen.generate_roof(building_mesh)

    # Combine
    final_mesh = roof_gen.combine_with_building(building_mesh, roof_mesh)

    # Export
    final_mesh.export(output_path)

    print(f"\n✓ Saved final model with roof: {output_path}")
    print("="*60)

    return final_mesh


# Example usage
if __name__ == "__main__":
    # After running Plan2Scene, add roofs
    scene_path = "outputs/133006hlt.scene.json"
    output_path = "outputs/133006hlt_with_roof.obj"

    add_roof_to_plan2scene_output(scene_path, output_path, roof_type='auto')
