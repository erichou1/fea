"""
Download 3DWire dataset and extract wireframe data
"""
import os
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm

def download_3dwire_dataset():
    """Download 3DWire dataset from GitHub"""
    print("=" * 60)
    print("Downloading 3DWire Dataset")
    print("=" * 60)
    
    # Create directories
    os.makedirs('data/3dwire_raw', exist_ok=True)
    os.makedirs('data/3dwire_processed', exist_ok=True)
    
    # Dataset info
    dataset_url = "https://github.com/3d-house-wireframe/3dwire"
    
    print("\nManual Download Required:")
    print(f"1. Visit: {dataset_url}")
    print("2. Follow instructions to download NPZ files")
    print("3. Place NPZ files in: data/3dwire_raw/")
    print("\nPress Enter after downloading files...")
    input()
    
    # Verify files exist
    npz_files = list(Path('data/3dwire_raw').glob('*.npz'))
    
    if len(npz_files) == 0:
        print("\n❌ ERROR: No NPZ files found in data/3dwire_raw/")
        print("Please download the dataset first.")
        return False
    
    print(f"\n✅ Found {len(npz_files)} wireframe files")
    return True

def load_wireframe(npz_path):
    """Load a single wireframe NPZ file"""
    data = np.load(npz_path, allow_pickle=True)
    
    return {
        'id': Path(npz_path).stem,
        'vertices': data.get('vertices', data.get('coords', None)),
        'edges': data.get('edges', data.get('lines', None)),
        'semantics': data.get('semantics', data.get('labels', None))
    }

def inspect_dataset():
    """Inspect downloaded dataset"""
    print("\n" + "=" * 60)
    print("Inspecting Dataset")
    print("=" * 60)
    
    npz_files = list(Path('data/3dwire_raw').glob('*.npz'))
    
    if len(npz_files) == 0:
        print("❌ No files to inspect")
        return
    
    # Load first file as example
    sample = load_wireframe(npz_files[0])
    
    print(f"\nTotal files: {len(npz_files)}")
    print(f"\nSample wireframe: {sample['id']}")
    print(f"  Vertices: {sample['vertices'].shape if sample['vertices'] is not None else 'None'}")
    print(f"  Edges: {sample['edges'].shape if sample['edges'] is not None else 'None'}")
    print(f"  Semantics: {sample['semantics'].shape if sample['semantics'] is not None else 'None'}")
    
    if sample['vertices'] is not None:
        print(f"\nVertex range:")
        print(f"  X: [{sample['vertices'][:, 0].min():.2f}, {sample['vertices'][:, 0].max():.2f}]")
        print(f"  Y: [{sample['vertices'][:, 1].min():.2f}, {sample['vertices'][:, 1].max():.2f}]")
        print(f"  Z: [{sample['vertices'][:, 2].min():.2f}, {sample['vertices'][:, 2].max():.2f}]")
    
    if sample['semantics'] is not None:
        unique_semantics = np.unique(sample['semantics'])
        print(f"\nUnique semantic labels: {unique_semantics}")

if __name__ == '__main__':
    if download_3dwire_dataset():
        inspect_dataset()
        print("\n✅ Data download complete!")
        print("\nNext step: Run process_wireframes.py")
