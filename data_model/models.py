# models.py - Neural Network Architectures
import torch
import torch.nn as nn
import torch.nn.functional as F

class FloorPlanSegmentationModel(nn.Module):
    """U-Net for segmenting walls, doors, windows, rooms"""
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
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
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
    """ResNet-based feature extractor for exterior images"""
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
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class MultiFloorFusionTransformer(nn.Module):
    """Transformer-based fusion for multi-floor 3D reconstruction"""
    def __init__(self, d_model=512, nhead=8, num_layers=6, max_vertices=2048):
        super().__init__()
        self.d_model = d_model
        self.max_vertices = max_vertices
        self.floorplan_encoder = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(256 * 16, d_model)
        )
        self.floor_embedding = nn.Embedding(10, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2048,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, 1024), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(1024, 2048), nn.ReLU(),
            nn.Linear(2048, max_vertices * 3)
        )

    def forward(self, floor_plans, exterior_features):
        batch_size = floor_plans[0].shape[0]
        device = floor_plans[0].device
        floor_tokens = []
        for floor_idx, fp in enumerate(floor_plans):
            fp_encoded = self.floorplan_encoder(fp)
            floor_pos = self.floor_embedding(torch.tensor([floor_idx] * batch_size, device=device))
            floor_tokens.append(fp_encoded + floor_pos)
        floor_tokens = torch.stack(floor_tokens, dim=1)
        all_tokens = torch.cat([floor_tokens, exterior_features], dim=1)
        fused = self.transformer(all_tokens)
        global_feature = fused.mean(dim=1)
        vertices = self.reconstruction_head(global_feature)
        vertices = vertices.view(batch_size, self.max_vertices, 3)
        return torch.tanh(vertices) * 10.0

class MeshGenerator(nn.Module):
    """Generates mesh materials from vertices"""
    def __init__(self, num_vertices=2048):
        super().__init__()
        self.num_vertices = num_vertices
        self.material_predictor = nn.Sequential(
            nn.Linear(num_vertices * 3, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, num_vertices * 5)
        )

    def forward(self, vertices):
        batch_size = vertices.shape[0]
        vertices_flat = vertices.view(batch_size, -1)
        materials = self.material_predictor(vertices_flat)
        materials = materials.view(batch_size, self.num_vertices, 5)
        materials[:, :, 0] = torch.sigmoid(materials[:, :, 0]) * 2500 + 500
        materials[:, :, 1] = torch.sigmoid(materials[:, :, 1]) * 199e9 + 1e9
        materials[:, :, 2] = torch.sigmoid(materials[:, :, 2]) * 0.35 + 0.1
        materials[:, :, 3] = torch.sigmoid(materials[:, :, 3]) * 0.4 + 0.1
        return materials

class FEASurrogateModel(nn.Module):
    """Predicts structural soundness"""
    def __init__(self, num_vertices=2048):
        super().__init__()
        self.vertex_processor = nn.Sequential(
            nn.Linear(8, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.attention = nn.MultiheadAttention(128, num_heads=8, batch_first=True)
        self.structural_analyzer = nn.Sequential(
            nn.Linear(num_vertices * 128, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 6)
        )
        self.soundness_classifier = nn.Sequential(
            nn.Linear(6, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, vertices, materials):
        batch_size = vertices.shape[0]
        vertex_features = torch.cat([vertices, materials], dim=-1)
        processed = self.vertex_processor(vertex_features)
        attended, _ = self.attention(processed, processed, processed)
        attended = attended + processed
        processed_flat = attended.view(batch_size, -1)
        structural_props = self.structural_analyzer(processed_flat)
        is_sound = self.soundness_classifier(structural_props)
        return structural_props, is_sound

class FloorPlanTo3DPipeline(nn.Module):
    """Complete end-to-end pipeline"""
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
        return {
            'vertices': vertices,
            'materials': materials,
            'structural_properties': structural_props,
            'is_structurally_sound': is_sound
        }
