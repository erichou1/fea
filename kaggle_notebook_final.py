# Floor Plan to 3D CAD - Kaggle Training Notebook (FINAL FIX)
# Handles files in input directory or needs extraction

# ═══════════════════════════════════════════════════════════════════════
# CELL 1: Setup and Installation
# ═══════════════════════════════════════════════════════════════════════

print("="*80)
print("FLOOR PLAN TO 3D CAD - KAGGLE TRAINING")
print("="*80)

import os
import sys
import shutil

# Check GPU
import torch
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Install additional packages
print("\nInstalling required packages...")
!pip install -q trimesh tensorboard

print("\n✓ Setup complete!")

# ═══════════════════════════════════════════════════════════════════════
# CELL 2: Setup Data Directories (FIXED)
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("SETTING UP DATA DIRECTORIES")
print("="*80)

from collections import defaultdict
import zipfile

# Your dataset path - update this to match your dataset name
DATASET_NAME = "floor-plans-and-house-images"  # Change if different
DATASET_PATH = f"/kaggle/input/{DATASET_NAME}"

print(f"\nDataset path: {DATASET_PATH}")

# Check what's in the dataset input folder
print("\nContents of dataset:")
if os.path.exists(DATASET_PATH):
    for item in os.listdir(DATASET_PATH):
        item_path = os.path.join(DATASET_PATH, item)
        if os.path.isfile(item_path):
            size = os.path.getsize(item_path) / (1024*1024)
            print(f"  - {item} ({size:.2f} MB)")
        elif os.path.isdir(item_path):
            num_files = len([f for f in os.listdir(item_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
            print(f"  - {item}/ ({num_files} images)")

# Check if folders already exist in input (unzipped upload)
floor_plans_input = os.path.join(DATASET_PATH, "floor_plans2")
house_images_input = os.path.join(DATASET_PATH, "house_images2")

floor_plans_dir = 'floor_plans2'
house_images_dir = 'house_images2'

if os.path.exists(floor_plans_input) and os.path.isdir(floor_plans_input):
    print(f"\n✓ Found floor_plans2 folder in input (already unzipped)")
    floor_plans_dir = floor_plans_input
else:
    # Extract from ZIP
    floor_plans_zip = os.path.join(DATASET_PATH, "floor_plans2.zip")
    if os.path.exists(floor_plans_zip):
        print(f"\nExtracting floor_plans2.zip...")
        with zipfile.ZipFile(floor_plans_zip, 'r') as zip_ref:
            zip_ref.extractall('.')
        # Check if extracted to nested folder
        if os.path.exists('floor_plans2/floor_plans2'):
            for item in os.listdir('floor_plans2/floor_plans2'):
                shutil.move(f'floor_plans2/floor_plans2/{item}', f'floor_plans2/{item}')
            os.rmdir('floor_plans2/floor_plans2')
        print(f"✓ Extracted")
        floor_plans_dir = 'floor_plans2'

if os.path.exists(house_images_input) and os.path.isdir(house_images_input):
    print(f"✓ Found house_images2 folder in input (already unzipped)")
    house_images_dir = house_images_input
else:
    # Extract from ZIP
    house_images_zip = os.path.join(DATASET_PATH, "house_images2.zip")
    if os.path.exists(house_images_zip):
        print(f"\nExtracting house_images2.zip...")
        with zipfile.ZipFile(house_images_zip, 'r') as zip_ref:
            zip_ref.extractall('.')
        # Check if extracted to nested folder
        if os.path.exists('house_images2/house_images2'):
            for item in os.listdir('house_images2/house_images2'):
                shutil.move(f'house_images2/house_images2/{item}', f'house_images2/{item}')
            os.rmdir('house_images2/house_images2')
        print(f"✓ Extracted")
        house_images_dir = 'house_images2'

print(f"\n" + "="*80)
print("DATA DIRECTORIES CONFIGURED")
print("="*80)
print(f"Floor plans: {floor_plans_dir}")
print(f"House images: {house_images_dir}")

# Debug: Show what's in the directories
def show_directory_contents(directory, max_files=10):
    if not os.path.exists(directory):
        print(f"\n⚠ {directory} does not exist!")
        return []

    files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"\n{directory}:")
    print(f"  Total images: {len(files)}")

    if files:
        print(f"  First {min(len(files), max_files)} examples:")
        for f in files[:max_files]:
            print(f"    - {f}")
    else:
        print("  ⚠ NO IMAGES FOUND!")

    return files

floor_files = show_directory_contents(floor_plans_dir)
house_files = show_directory_contents(house_images_dir)

# Parse IDs
def extract_id_from_filename(filename):
    """Extract house ID from filename"""
    # Remove extension
    name = filename.rsplit('.', 1)[0]
    # Get first part before underscore
    return name.split('_')[0]

floor_ids = defaultdict(list)
for f in floor_files:
    house_id = extract_id_from_filename(f)
    floor_ids[house_id].append(f)

house_ids_dict = defaultdict(list)
for f in house_files:
    house_id = extract_id_from_filename(f)
    house_ids_dict[house_id].append(f)

print(f"\n" + "="*80)
print("HOUSE ID DETECTION")
print("="*80)
print(f"\nFloor plan IDs: {len(floor_ids)}")
if floor_ids:
    for house_id in list(floor_ids.keys())[:5]:
        print(f"  {house_id}: {floor_ids[house_id]}")

print(f"\nHouse image IDs: {len(house_ids_dict)}")
if house_ids_dict:
    for house_id in list(house_ids_dict.keys())[:5]:
        print(f"  {house_id}: {house_ids_dict[house_id]}")

# Find matching
matching_ids = set(floor_ids.keys()).intersection(set(house_ids_dict.keys()))

print(f"\n" + "="*80)
print(f"MATCHING HOUSES: {len(matching_ids)}")
print("="*80)

if matching_ids:
    print(f"\n✓ SUCCESS! Found {len(matching_ids)} houses with both floor plans and exteriors")
    print("\nFirst 5 examples:")
    for house_id in list(matching_ids)[:5]:
        print(f"  {house_id}:")
        print(f"    Floor plans ({len(floor_ids[house_id])}): {floor_ids[house_id]}")
        print(f"    Exteriors ({len(house_ids_dict[house_id])}): {house_ids_dict[house_id]}")
else:
    print("\n✗ ERROR: No matching house IDs found!")
    print("\nFloor plan IDs:", list(floor_ids.keys())[:10])
    print("House image IDs:", list(house_ids_dict.keys())[:10])
    raise ValueError("No matching house IDs")

# Store for later use
FLOOR_PLANS_DIR = floor_plans_dir
HOUSE_IMAGES_DIR = house_images_dir

# ═══════════════════════════════════════════════════════════════════════
# CELL 3: Model Architecture
# ═══════════════════════════════════════════════════════════════════════

import torch.nn as nn
import torch.nn.functional as F

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
        self.bn1, self.relu, self.maxpool = nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool, self.fc = nn.AdaptiveAvgPool2d((1, 1)), nn.Linear(512, feature_dim)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        for _ in range(1, blocks):
            layers.extend([nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)])
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.fc(torch.flatten(self.avgpool(self.layer4(self.layer3(self.layer2(self.layer1(x))))), 1))

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
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, 1024), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, max_vertices * 3)
        )

    def forward(self, floor_plans, exterior_features):
        batch_size, device = floor_plans[0].shape[0], floor_plans[0].device
        floor_tokens = [self.floorplan_encoder(fp) + self.floor_embedding(torch.tensor([i] * batch_size, device=device)) 
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
        materials = self.material_predictor(vertices.view(vertices.shape[0], -1)).view(vertices.shape[0], self.num_vertices, 5)
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
        structural_props = self.structural_analyzer((attended + processed).view(vertices.shape[0], -1))
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
        return {'vertices': vertices, 'materials': materials, 'structural_properties': structural_props, 'is_structurally_sound': is_sound}

print("✓ Model architecture loaded")

# ═══════════════════════════════════════════════════════════════════════
# CELL 4: Dataset
# ═══════════════════════════════════════════════════════════════════════

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class HouseDataset(Dataset):
    def __init__(self, floor_plans_dir, house_images_dir, img_size=512):
        self.floor_plans_dir = floor_plans_dir
        self.house_images_dir = house_images_dir
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Group files by ID
        self.floor_plans = defaultdict(list)
        self.house_images = defaultdict(list)

        for f in os.listdir(floor_plans_dir):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                house_id = extract_id_from_filename(f)
                self.floor_plans[house_id].append(f)

        for f in os.listdir(house_images_dir):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                house_id = extract_id_from_filename(f)
                self.house_images[house_id].append(f)

        # Get matching IDs
        self.house_ids = sorted([h for h in self.floor_plans.keys() 
                                if h in self.house_images])

        print(f"\nDataset: {len(self.house_ids)} houses")

    def __len__(self):
        return len(self.house_ids)

    def __getitem__(self, idx):
        house_id = self.house_ids[idx]
        floor_plans = [self.transform(Image.open(os.path.join(self.floor_plans_dir, f)).convert('RGB')) 
                      for f in sorted(self.floor_plans[house_id])]
        exteriors = [self.transform(Image.open(os.path.join(self.house_images_dir, f)).convert('RGB'))
                    for f in sorted(self.house_images[house_id])]
        return {'floor_plans': floor_plans, 'exteriors': exteriors, 'house_id': house_id}

def collate_fn(batch):
    max_floors = max(len(item['floor_plans']) for item in batch)
    max_exteriors = max(len(item['exteriors']) for item in batch)
    floor_plans_padded = [torch.stack([item['floor_plans'][i] if i < len(item['floor_plans']) else torch.zeros_like(item['floor_plans'][0]) 
                                       for item in batch]) for i in range(max_floors)]
    exteriors_padded = [torch.stack([item['exteriors'][i] if i < len(item['exteriors']) else torch.zeros_like(item['exteriors'][0]) 
                                     for item in batch]) for i in range(max_exteriors)]
    return {'floor_plans': floor_plans_padded, 'exteriors': exteriors_padded, 'house_ids': [item['house_id'] for item in batch]}

print("✓ Dataset loaded")

# ═══════════════════════════════════════════════════════════════════════
# CELL 5: Training Setup
# ═══════════════════════════════════════════════════════════════════════

from tqdm import tqdm

config = {
    'num_vertices': 1024,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'batch_size': 1,
    'num_epochs': 50,
    'save_every': 5
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FloorPlanTo3DPipeline(num_vertices=config['num_vertices']).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'], eta_min=1e-6)

dataset = HouseDataset(FLOOR_PLANS_DIR, HOUSE_IMAGES_DIR)
train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2, collate_fn=collate_fn, pin_memory=True)

print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Device: {device}")
print(f"Batches: {len(train_loader)}")

os.makedirs('/kaggle/working/checkpoints', exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# CELL 6: Training
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("TRAINING")
print("="*80)

for epoch in range(config['num_epochs']):
    model.train()
    total_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
    for batch in pbar:
        floor_plans = [fp.to(device) for fp in batch['floor_plans']]
        exteriors = [ext.to(device) for ext in batch['exteriors']]

        optimizer.zero_grad()
        outputs = model(floor_plans, exteriors)

        vertex_loss = torch.abs(torch.std(outputs['vertices'], dim=1).mean() - 3.0)
        material_loss = torch.var(outputs['materials'], dim=1).mean() * 0.1
        soundness_loss = nn.MSELoss()(outputs['is_structurally_sound'], torch.ones_like(outputs['is_structurally_sound']) * 0.8)
        struct_props = outputs['structural_properties']
        stress_reg = torch.abs(struct_props[:, 0] - 50.0).mean()
        safety_reg = torch.abs(struct_props[:, 2] - 2.5).mean()

        loss = 0.3 * vertex_loss + 0.2 * material_loss + 1.0 * soundness_loss + 0.1 * stress_reg + 0.1 * safety_reg

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    scheduler.step()

    if (epoch + 1) % config['save_every'] == 0:
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, 
                   f'/kaggle/working/checkpoints/checkpoint_epoch_{epoch}.pth')
        print(f"✓ Saved checkpoint")

torch.save({'model_state_dict': model.state_dict()}, '/kaggle/working/checkpoints/final_model.pth')
print("\n✓ TRAINING COMPLETE! Download final_model.pth from Output")
