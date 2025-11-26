# custom_dataset.py - Direct loader for floor_plans2 and house_images2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import os
from collections import defaultdict

class DirectHouseDataset(Dataset):
    """
    Load data directly from floor_plans2 and house_images2 folders
    without reorganizing files.
    """

    def __init__(self, floor_plans_dir='floor_plans2', 
                 house_images_dir='house_images2',
                 transform=None, img_size=512):

        self.floor_plans_dir = floor_plans_dir
        self.house_images_dir = house_images_dir
        self.img_size = img_size

        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

        # Parse and group files by house ID
        self.floor_plans = defaultdict(list)
        self.house_images = defaultdict(list)

        # Parse floor plans: ID_floorplan_NUMBER.jpg
        if os.path.exists(floor_plans_dir):
            for filename in sorted(os.listdir(floor_plans_dir)):
                if filename.endswith(('.jpg', '.png', '.jpeg')):
                    house_id = filename.split('_')[0]
                    self.floor_plans[house_id].append(filename)

        # Parse house images: ID_house_NUMBER.jpg
        if os.path.exists(house_images_dir):
            for filename in sorted(os.listdir(house_images_dir)):
                if filename.endswith(('.jpg', '.png', '.jpeg')):
                    house_id = filename.split('_')[0]
                    self.house_images[house_id].append(filename)

        # Get house IDs that have both floor plans and exteriors
        self.house_ids = sorted([
            hid for hid in self.floor_plans.keys() 
            if hid in self.house_images and 
               len(self.floor_plans[hid]) > 0 and 
               len(self.house_images[hid]) > 0
        ])

        print(f"Loaded {len(self.house_ids)} houses with complete data")
        if self.house_ids:
            print(f"Example house ID: {self.house_ids[0]}")
            print(f"  - Floor plans: {len(self.floor_plans[self.house_ids[0]])}")
            print(f"  - Exteriors: {len(self.house_images[self.house_ids[0]])}")

    def __len__(self):
        return len(self.house_ids)

    def __getitem__(self, idx):
        house_id = self.house_ids[idx]

        # Load floor plans
        floor_plans = []
        for filename in sorted(self.floor_plans[house_id]):
            img_path = os.path.join(self.floor_plans_dir, filename)
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            floor_plans.append(img)

        # Load exterior images
        exteriors = []
        for filename in sorted(self.house_images[house_id]):
            img_path = os.path.join(self.house_images_dir, filename)
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            exteriors.append(img)

        return {
            'floor_plans': floor_plans,
            'exteriors': exteriors,
            'house_id': house_id,
            'num_floors': len(floor_plans),
            'num_exteriors': len(exteriors)
        }

def direct_collate_fn(batch):
    """Custom collate function for variable numbers of images"""
    max_floors = max(item['num_floors'] for item in batch)
    max_exteriors = max(item['num_exteriors'] for item in batch)

    # Pad floor plans
    floor_plans_padded = []
    for i in range(max_floors):
        floor_batch = []
        for item in batch:
            if i < len(item['floor_plans']):
                floor_batch.append(item['floor_plans'][i])
            else:
                # Zero padding
                floor_batch.append(torch.zeros_like(item['floor_plans'][0]))
        floor_plans_padded.append(torch.stack(floor_batch))

    # Pad exteriors
    exteriors_padded = []
    for i in range(max_exteriors):
        ext_batch = []
        for item in batch:
            if i < len(item['exteriors']):
                ext_batch.append(item['exteriors'][i])
            else:
                ext_batch.append(torch.zeros_like(item['exteriors'][0]))
        exteriors_padded.append(torch.stack(ext_batch))

    return {
        'floor_plans': floor_plans_padded,
        'exteriors': exteriors_padded,
        'house_ids': [item['house_id'] for item in batch],
        'num_floors': [item['num_floors'] for item in batch],
        'num_exteriors': [item['num_exteriors'] for item in batch]
    }

def create_direct_dataloader(floor_plans_dir='floor_plans2',
                             house_images_dir='house_images2',
                             batch_size=4, 
                             num_workers=4, 
                             shuffle=True):
    """Create DataLoader directly from your folders"""
    dataset = DirectHouseDataset(floor_plans_dir, house_images_dir)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=direct_collate_fn,
        pin_memory=True
    )

# Example usage
if __name__ == '__main__':
    print("Testing direct data loading...")

    dataloader = create_direct_dataloader(
        floor_plans_dir='floor_plans2',
        house_images_dir='house_images2',
        batch_size=2,
        shuffle=False
    )

    if len(dataloader) > 0:
        batch = next(iter(dataloader))
        print(f"\nBatch loaded successfully!")
        print(f"Number of floor plan levels: {len(batch['floor_plans'])}")
        print(f"Number of exterior image sets: {len(batch['exteriors'])}")
        print(f"Batch size: {batch['floor_plans'][0].shape[0]}")
        print(f"House IDs in batch: {batch['house_ids']}")
    else:
        print("\n⚠ No data found. Check your directory paths.")
