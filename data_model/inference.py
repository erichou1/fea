# inference.py - Generate 3D CAD models from your trained model
# FIXED VERSION - Generates proper meshes with faces

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os
import numpy as np
import struct

# ═══════════════════════════════════════════════════════════════════════
# Model Architecture (Same as training)
# ═══════════════════════════════════════════════════════════════════════

class FloorPlanSegmentationModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=6):
        super().__init__()
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        self.bottleneck = self._conv_block(512, 1024)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._conv_block(128, 64)
        self.out = nn.Conv2d(64, num_classes, 1)
        self.pool = nn.MaxPool2d(2)

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.upconv4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))
        return self.out(d1)

class ExteriorFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, feature_dim)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
                  nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        for _ in range(1, blocks):
            layers.extend([nn.Conv2d(out_channels, out_channels, 3, padding=1),
                          nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)])
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))

class MultiFloorFusionTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, max_vertices=2048):
        super().__init__()
        self.d_model, self.max_vertices = d_model, max_vertices
        self.floorplan_encoder = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(), nn.Linear(256 * 16, d_model)
        )
        self.floor_embedding = nn.Embedding(10, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, 1024), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, max_vertices * 3)
        )

    def forward(self, floor_plans, exterior_features):
        batch_size, device = floor_plans[0].shape[0], floor_plans[0].device
        floor_tokens = [self.floorplan_encoder(fp) + 
                       self.floor_embedding(torch.tensor([i] * batch_size, device=device)) 
                       for i, fp in enumerate(floor_plans)]
        fused = self.transformer(torch.cat([torch.stack(floor_tokens, dim=1), exterior_features], dim=1))
        vertices = self.reconstruction_head(fused.mean(dim=1)).view(batch_size, self.max_vertices, 3)
        return torch.tanh(vertices) * 10.0

class MeshGenerator(nn.Module):
    def __init__(self, num_vertices=2048):
        super().__init__()
        self.num_vertices = num_vertices
        self.material_predictor = nn.Sequential(
            nn.Linear(num_vertices * 3, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, num_vertices * 5)
        )

    def forward(self, vertices):
        materials = self.material_predictor(vertices.view(vertices.shape[0], -1))
        materials = materials.view(vertices.shape[0], self.num_vertices, 5)
        materials[:, :, 0] = torch.sigmoid(materials[:, :, 0]) * 2500 + 500
        materials[:, :, 1] = torch.sigmoid(materials[:, :, 1]) * 199e9 + 1e9
        materials[:, :, 2] = torch.sigmoid(materials[:, :, 2]) * 0.35 + 0.1
        materials[:, :, 3] = torch.sigmoid(materials[:, :, 3]) * 0.4 + 0.1
        return materials

class FEASurrogateModel(nn.Module):
    def __init__(self, num_vertices=2048):
        super().__init__()
        self.vertex_processor = nn.Sequential(
            nn.Linear(8, 256), nn.ReLU(), nn.LayerNorm(256), nn.Linear(256, 128), nn.ReLU()
        )
        self.attention = nn.MultiheadAttention(128, num_heads=8, batch_first=True)
        self.structural_analyzer = nn.Sequential(
            nn.Linear(num_vertices * 128, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 6)
        )
        self.soundness_classifier = nn.Sequential(
            nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, vertices, materials):
        processed = self.vertex_processor(torch.cat([vertices, materials], dim=-1))
        attended, _ = self.attention(processed, processed, processed)
        structural_props = self.structural_analyzer((attended + processed).reshape(vertices.shape[0], -1))
        return structural_props, self.soundness_classifier(structural_props)

class FloorPlanTo3DPipeline(nn.Module):
    def __init__(self, num_vertices=2048):
        super().__init__()
        self.segmentation_model = FloorPlanSegmentationModel()
        self.exterior_extractor = ExteriorFeatureExtractor(feature_dim=512)
        self.fusion_model = MultiFloorFusionTransformer(max_vertices=num_vertices)
        self.mesh_generator = MeshGenerator(num_vertices=num_vertices)
        self.fea_model = FEASurrogateModel(num_vertices=num_vertices)

    def forward(self, floor_plan_images, exterior_images):
        segmented_floors = [self.segmentation_model(fp) for fp in floor_plan_images]
        exterior_features = torch.stack([self.exterior_extractor(ext) for ext in exterior_images], dim=1)
        vertices = self.fusion_model(segmented_floors, exterior_features)
        materials = self.mesh_generator(vertices)
        structural_props, is_sound = self.fea_model(vertices, materials)
        return {'vertices': vertices, 'materials': materials, 
                'structural_properties': structural_props, 'is_structurally_sound': is_sound}

# ═══════════════════════════════════════════════════════════════════════
# Inference Functions
# ═══════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path, num_vertices=2048, device='cpu'):
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")

    model = FloorPlanTo3DPipeline(num_vertices=num_vertices)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully!")
    print(f"  Trained for {checkpoint.get('epoch', '?')} epochs")
    print(f"  {sum(p.numel() for p in model.parameters()):,} parameters")

    return model

def load_images(floor_plan_paths, exterior_paths, img_size=512):
    """Load and preprocess images"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    floor_plans = [transform(Image.open(p).convert('RGB')) for p in floor_plan_paths]
    exteriors = [transform(Image.open(p).convert('RGB')) for p in exterior_paths]

    return floor_plans, exteriors

def save_obj(vertices, filepath):
    """Save vertices as OBJ file with proper triangular mesh"""
    with open(filepath, 'w') as f:
        f.write("# Generated by Floor Plan to 3D CAD\n")
        f.write(f"# {len(vertices)} vertices, {len(vertices)//3} triangles\n\n")

        # Write all vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        f.write("\n# Triangular faces\n")

        # Create triangular faces (every 3 vertices = 1 triangle)
        num_triangles = len(vertices) // 3
        for i in range(num_triangles):
            v1 = i * 3 + 1  # OBJ indices start at 1
            v2 = i * 3 + 2
            v3 = i * 3 + 3
            f.write(f"f {v1} {v2} {v3}\n")

def save_stl(vertices, filepath):
    """Save as binary STL with triangular mesh"""
    triangles = []
    num_triangles = len(vertices) // 3

    for i in range(num_triangles):
        v1 = vertices[i * 3]
        v2 = vertices[i * 3 + 1]
        v3 = vertices[i * 3 + 2]

        # Calculate normal
        edge1 = v2 - v1
        edge2 = v3 - v1
        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        normal = normal / norm if norm > 1e-8 else np.array([0, 0, 1])

        triangles.append((normal, v1, v2, v3))

    # Write binary STL
    with open(filepath, 'wb') as f:
        header = b'Binary STL - Floor Plan to 3D' + b' ' * 50
        f.write(header[:80])
        f.write(struct.pack('<I', len(triangles)))

        for normal, v1, v2, v3 in triangles:
            f.write(struct.pack('<3f', *normal))
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            f.write(struct.pack('<3f', *v3))
            f.write(struct.pack('<H', 0))

def save_materials(materials, filepath):
    """Save material properties"""
    with open(filepath, 'w') as f:
        f.write("Material Properties Report\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total vertices: {len(materials)}\n\n")

        avg_density = materials[:, 0].mean()
        avg_youngs = materials[:, 1].mean()
        avg_poisson = materials[:, 2].mean()
        avg_thickness = materials[:, 3].mean()

        f.write("Average Properties:\n")
        f.write(f"  Density: {avg_density:.2f} kg/m³\n")
        f.write(f"  Young's Modulus: {avg_youngs:.2e} Pa\n")
        f.write(f"  Poisson's Ratio: {avg_poisson:.3f}\n")
        f.write(f"  Wall Thickness: {avg_thickness:.3f} m\n")

def save_analysis(struct_props, is_sound, filepath):
    """Save structural analysis report"""
    with open(filepath, 'w') as f:
        f.write("Structural Analysis Report (FEA Surrogate)\n")
        f.write("="*60 + "\n\n")

        f.write(f"Structurally Sound: {'YES ✓' if is_sound > 0.5 else 'NO ✗'} ({is_sound*100:.1f}%)\n\n")

        f.write("Predicted FEA Metrics:\n")
        f.write(f"  Max Stress: {struct_props[0]:.2f} MPa\n")
        f.write(f"  Max Displacement: {struct_props[1]:.6f} m\n")
        f.write(f"  Safety Factor: {struct_props[2]:.2f}\n")
        f.write(f"  Stability Score: {struct_props[3]:.3f}\n")
        f.write(f"  Resonance Frequency: {struct_props[4]:.2f} Hz\n")
        f.write(f"  Max Deflection: {struct_props[5]:.6f} m\n")

def generate_3d_model(model, floor_plan_paths, exterior_paths, output_name, device='cpu'):
    """Generate 3D model from floor plans and exterior images"""

    print(f"\nGenerating 3D model: {output_name}")
    print("="*60)

    # Load and prepare images
    print(f"Loading {len(floor_plan_paths)} floor plan(s), {len(exterior_paths)} exterior(s)...")
    floor_plans, exteriors = load_images(floor_plan_paths, exterior_paths)

    floor_plans = [fp.unsqueeze(0).to(device) for fp in floor_plans]
    exteriors = [ext.unsqueeze(0).to(device) for ext in exteriors]

    # Generate
    print("Generating 3D geometry...")
    with torch.no_grad():
        output = model(floor_plans, exteriors)

    # Extract results
    vertices = output['vertices'][0].cpu().numpy()
    materials = output['materials'][0].cpu().numpy()
    struct_props = output['structural_properties'][0].cpu().numpy()
    is_sound = output['is_structurally_sound'][0].item()

    print(f"✓ Generated {len(vertices)} vertices ({len(vertices)//3} triangles)")

    # Save outputs
    os.makedirs('output', exist_ok=True)

    save_obj(vertices, f'output/{output_name}.obj')
    print(f"✓ Saved: output/{output_name}.obj")

    save_stl(vertices, f'output/{output_name}.stl')
    print(f"✓ Saved: output/{output_name}.stl")

    save_materials(materials, f'output/{output_name}_materials.txt')
    print(f"✓ Saved: output/{output_name}_materials.txt")

    save_analysis(struct_props, is_sound, f'output/{output_name}_analysis.txt')
    print(f"✓ Saved: output/{output_name}_analysis.txt")

    print(f"\n{'='*60}")
    print(f"✓ SUCCESS! Open output/{output_name}.obj in Blender/CAD software")
    print(f"{'='*60}")

    return vertices, materials, struct_props, is_sound

# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*60)
    print("FLOOR PLAN TO 3D CAD - INFERENCE")
    print("="*60)

    CHECKPOINT_PATH = 'checkpoints/final_model.pth'
    NUM_VERTICES = 2048  # MUST match training config!
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {DEVICE}")

    # Load model
    model = load_model(CHECKPOINT_PATH, num_vertices=NUM_VERTICES, device=DEVICE)

    # EDIT THIS: Your house ID
    HOUSE_ID = '133006hlt'

    floor_plan_paths = [
        f'floor_plans2/{HOUSE_ID}_floorplan_1.jpg',
    ]

    exterior_paths = [
        f'house_images2/{HOUSE_ID}_house_1.jpg',
        f'house_images2/{HOUSE_ID}_house_2.jpg',
        f'house_images2/{HOUSE_ID}_house_3.jpg',
    ]

    # Generate!
    generate_3d_model(model, floor_plan_paths, exterior_paths, HOUSE_ID, device=DEVICE)
