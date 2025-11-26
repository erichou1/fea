# preprocess_data.py - Organize your existing data structure
import os
import shutil
from collections import defaultdict
import json
from PIL import Image

def organize_data(floor_plans_dir='floor_plans2', 
                  house_images_dir='house_images2',
                  output_dir='data/organized'):
    """
    Organizes floor plans and house images by ID.

    Input structure:
        floor_plans2/
            14915rk_floorplan_1.jpg
            14915rk_floorplan_2.jpg
            23456ab_floorplan_1.jpg
        house_images2/
            14915rk_house_1.jpg
            14915rk_house_2.jpg
            23456ab_house_1.jpg

    Output structure:
        data/organized/
            14915rk/
                floor_plans/
                    floor_0.jpg
                    floor_1.jpg
                exteriors/
                    exterior_0.jpg
                    exterior_1.jpg
                metadata.json
            23456ab/
                ...
    """

    print("=" * 80)
    print("ORGANIZING DATA FOR TRAINING")
    print("=" * 80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Parse floor plans
    floor_plans = defaultdict(list)
    if os.path.exists(floor_plans_dir):
        for filename in os.listdir(floor_plans_dir):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                # Parse filename: ID_floorplan_NUMBER.jpg
                parts = filename.split('_')
                if len(parts) >= 3 and 'floorplan' in filename.lower():
                    house_id = parts[0]
                    floor_plans[house_id].append(filename)

    print(f"\nFound {len(floor_plans)} unique houses in floor_plans2/")

    # Parse house exterior images
    house_images = defaultdict(list)
    if os.path.exists(house_images_dir):
        for filename in os.listdir(house_images_dir):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                # Parse filename: ID_house_NUMBER.jpg
                parts = filename.split('_')
                if len(parts) >= 3 and 'house' in filename.lower():
                    house_id = parts[0]
                    house_images[house_id].append(filename)

    print(f"Found {len(house_images)} unique houses in house_images2/")

    # Get all unique house IDs
    all_house_ids = set(list(floor_plans.keys()) + list(house_images.keys()))
    print(f"\nTotal unique houses: {len(all_house_ids)}")

    # Organize each house
    organized_count = 0
    skipped_count = 0

    for house_id in sorted(all_house_ids):
        # Skip if no floor plans or no house images
        if house_id not in floor_plans or house_id not in house_images:
            print(f"⚠ Skipping {house_id}: Missing floor plans or house images")
            skipped_count += 1
            continue

        # Create house directory
        house_dir = os.path.join(output_dir, house_id)
        floor_plans_output = os.path.join(house_dir, 'floor_plans')
        exteriors_output = os.path.join(house_dir, 'exteriors')

        os.makedirs(floor_plans_output, exist_ok=True)
        os.makedirs(exteriors_output, exist_ok=True)

        # Copy and rename floor plans
        sorted_floor_plans = sorted(floor_plans[house_id])
        for idx, filename in enumerate(sorted_floor_plans):
            src = os.path.join(floor_plans_dir, filename)
            dst = os.path.join(floor_plans_output, f'floor_{idx}.jpg')
            shutil.copy2(src, dst)

        # Copy and rename exterior images
        sorted_exteriors = sorted(house_images[house_id])
        for idx, filename in enumerate(sorted_exteriors):
            src = os.path.join(house_images_dir, filename)
            dst = os.path.join(exteriors_output, f'exterior_{idx}.jpg')
            shutil.copy2(src, dst)

        # Get image dimensions for metadata
        sample_floor_plan = Image.open(os.path.join(floor_plans_output, 'floor_0.jpg'))
        width, height = sample_floor_plan.size

        # Create metadata.json
        metadata = {
            "house_id": house_id,
            "num_floors": len(sorted_floor_plans),
            "num_exterior_images": len(sorted_exteriors),
            "floor_plan_resolution": [width, height],
            "building_height": len(sorted_floor_plans) * 3.0,  # Estimate: 3m per floor
            "floor_height": 3.0,
            "building_type": "unknown",
            "original_floor_plans": sorted_floor_plans,
            "original_exteriors": sorted_exteriors
        }

        with open(os.path.join(house_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, indent=2, fp=f)

        print(f"✓ {house_id}: {len(sorted_floor_plans)} floors, {len(sorted_exteriors)} exteriors")
        organized_count += 1

    print(f"\n" + "=" * 80)
    print(f"SUMMARY")
    print("=" * 80)
    print(f"Successfully organized: {organized_count} houses")
    print(f"Skipped: {skipped_count} houses")
    print(f"Output directory: {output_dir}")

    # Split into train/val
    print(f"\nCreating train/val split (80/20)...")
    split_data(output_dir)

    return organized_count

def split_data(organized_dir='data/organized', 
               train_ratio=0.8):
    """Split organized data into train and validation sets"""

    train_dir = 'data/train'
    val_dir = 'data/val'

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Get all house directories
    houses = [d for d in os.listdir(organized_dir) 
              if os.path.isdir(os.path.join(organized_dir, d))]
    houses.sort()

    # Calculate split
    split_idx = int(len(houses) * train_ratio)
    train_houses = houses[:split_idx]
    val_houses = houses[split_idx:]

    # Move to train
    for house_id in train_houses:
        src = os.path.join(organized_dir, house_id)
        dst = os.path.join(train_dir, house_id)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    # Move to val
    for house_id in val_houses:
        src = os.path.join(organized_dir, house_id)
        dst = os.path.join(val_dir, house_id)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    print(f"✓ Train set: {len(train_houses)} houses -> data/train/")
    print(f"✓ Val set: {len(val_houses)} houses -> data/val/")

def inspect_data(floor_plans_dir='floor_plans2', 
                 house_images_dir='house_images2'):
    """Inspect your current data structure"""

    print("\n" + "=" * 80)
    print("DATA INSPECTION")
    print("=" * 80)

    # Check floor plans
    if os.path.exists(floor_plans_dir):
        floor_files = [f for f in os.listdir(floor_plans_dir) 
                       if f.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"\nFloor Plans Directory: {floor_plans_dir}")
        print(f"Total files: {len(floor_files)}")
        if floor_files:
            print(f"Example files:")
            for f in floor_files[:5]:
                print(f"  - {f}")

        # Count by ID
        ids = set()
        for f in floor_files:
            house_id = f.split('_')[0]
            ids.add(house_id)
        print(f"Unique house IDs: {len(ids)}")
    else:
        print(f"\n⚠ Directory not found: {floor_plans_dir}")

    # Check house images
    if os.path.exists(house_images_dir):
        house_files = [f for f in os.listdir(house_images_dir) 
                       if f.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"\nHouse Images Directory: {house_images_dir}")
        print(f"Total files: {len(house_files)}")
        if house_files:
            print(f"Example files:")
            for f in house_files[:5]:
                print(f"  - {f}")

        # Count by ID
        ids = set()
        for f in house_files:
            house_id = f.split('_')[0]
            ids.add(house_id)
        print(f"Unique house IDs: {len(ids)}")
    else:
        print(f"\n⚠ Directory not found: {house_images_dir}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess floor plans and house images')
    parser.add_argument('--floor_plans_dir', default='floor_plans2', 
                       help='Directory with floor plan images')
    parser.add_argument('--house_images_dir', default='house_images2',
                       help='Directory with house exterior images')
    parser.add_argument('--output_dir', default='data/organized',
                       help='Output directory for organized data')
    parser.add_argument('--inspect', action='store_true',
                       help='Only inspect data without organizing')

    args = parser.parse_args()

    if args.inspect:
        inspect_data(args.floor_plans_dir, args.house_images_dir)
    else:
        inspect_data(args.floor_plans_dir, args.house_images_dir)
        print("\n" + "=" * 80)
        print("STARTING ORGANIZATION...")
        print("=" * 80)
        organize_data(args.floor_plans_dir, args.house_images_dir, args.output_dir)
        print("\n✓ Data preprocessing complete!")
        print("\nNext steps:")
        print("  1. Review organized data in data/train/ and data/val/")
        print("  2. Run: python train.py")
