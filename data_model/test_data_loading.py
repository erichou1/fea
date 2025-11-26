# test_data_loading.py - Test if your data loads correctly
import os
from custom_dataset import DirectHouseDataset, create_direct_dataloader

print("=" * 80)
print("TESTING DATA LOADING FROM YOUR FOLDERS")
print("=" * 80)

# Check if directories exist
floor_plans_dir = './floor_plans2'
house_images_dir = './house_images2'

print(f"\nChecking directories...")
print(f"Floor plans: {floor_plans_dir} - {'✓ EXISTS' if os.path.exists(floor_plans_dir) else '✗ NOT FOUND'}")
print(f"House images: {house_images_dir} - {'✓ EXISTS' if os.path.exists(house_images_dir) else '✗ NOT FOUND'}")

if not os.path.exists(floor_plans_dir) or not os.path.exists(house_images_dir):
    print("\n⚠ Please ensure both directories exist in the current folder")
    exit(1)

# Load dataset
print(f"\nLoading dataset...")
dataset = DirectHouseDataset(
    floor_plans_dir=floor_plans_dir,
    house_images_dir=house_images_dir
)

if len(dataset) == 0:
    print("\n⚠ No houses found with matching floor plans and exterior images")
    print("\nExpected filename format:")
    print("  Floor plans: {ID}_floorplan_{NUMBER}.jpg  (e.g., 14915rk_floorplan_2.jpg)")
    print("  Exteriors: {ID}_house_{NUMBER}.jpg  (e.g., 14915rk_house_1.jpg)")
    exit(1)

print(f"\n✓ Successfully loaded {len(dataset)} houses")

# Show first few examples
print(f"\nFirst 5 houses:")
for i in range(min(5, len(dataset))):
    sample = dataset[i]
    print(f"  {i+1}. {sample['house_id']}: {sample['num_floors']} floors, {sample['num_exteriors']} exteriors")

# Test dataloader
print(f"\nTesting DataLoader...")
dataloader = create_direct_dataloader(
    floor_plans_dir=floor_plans_dir,
    house_images_dir=house_images_dir,
    batch_size=2,
    num_workers=0,  # Use 0 for testing
    shuffle=False
)

batch = next(iter(dataloader))
print(f"\n✓ DataLoader working!")
print(f"  Batch size: {len(batch['house_ids'])}")
print(f"  Floor plan levels: {len(batch['floor_plans'])}")
print(f"  Exterior image sets: {len(batch['exteriors'])}")
print(f"  First floor plan shape: {batch['floor_plans'][0].shape}")
print(f"  Houses in batch: {batch['house_ids']}")

print(f"\n" + "=" * 80)
print("✓ DATA LOADING TEST PASSED!")
print("=" * 80)
print(f"\nYou're ready to train!")
print(f"Run: python train_direct.py")
