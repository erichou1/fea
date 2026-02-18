# complete_pipeline.py - Full Pipeline: Floor Plans → 3D Models with Roofs

import os
import sys
import subprocess
from pathlib import Path

def run_plan2scene(floor_plan_path, house_id):
    """
    Run Plan2Scene on a floor plan

    Args:
        floor_plan_path: Path to floor plan image
        house_id: Unique house identifier
    """
    print(f"\nRunning Plan2Scene for house: {house_id}")
    print("="*60)

    # 1. Convert floor plan to Plan2Scene format
    # (You may need to use their raster-to-vector tool first)

    # 2. Run Plan2Scene inference
    cmd = [
        "python", "plan2scene/code/scripts/plan2scene/generate_scenes.py",
        "--floor-plan", floor_plan_path,
        "--output-dir", f"outputs/{house_id}",
        "--checkpoint", "pretrained_models/gnn_prop/checkpoint.pth"
    ]

    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    print(f"✓ Plan2Scene complete for {house_id}")

    return f"outputs/{house_id}/scene.obj"

def add_roof(building_obj_path, roof_type='auto'):
    """Add procedural roof to building"""
    from roof_generator import add_roof_to_plan2scene_output

    output_path = building_obj_path.replace('.obj', '_with_roof.obj')
    add_roof_to_plan2scene_output(building_obj_path, output_path, roof_type)

    return output_path

def process_house(floor_plan_path, exterior_photos, house_id):
    """
    Complete pipeline for one house

    Args:
        floor_plan_path: Path to floor plan image
        exterior_photos: List of paths to exterior photos
        house_id: House identifier
    """
    print("\n" + "="*60)
    print(f"PROCESSING HOUSE: {house_id}")
    print("="*60)

    # Step 1: Run Plan2Scene
    building_obj = run_plan2scene(floor_plan_path, house_id)

    # Step 2: Add roof
    final_model = add_roof(building_obj, roof_type='auto')

    # Step 3: Optionally texture with exterior photos
    # (Advanced: project exterior photos onto model)

    print(f"\n✓ COMPLETE! Final model: {final_model}")

    return final_model

# Main execution
if __name__ == "__main__":
    # Example: Process all houses in your dataset
    houses = {
        '133006hlt': {
            'floor_plan': 'floor_plans2/133006hlt_floorplan_1.jpg',
            'exteriors': [
                'house_images2/133006hlt_house_1.jpg',
                'house_images2/133006hlt_house_2.jpg',
                'house_images2/133006hlt_house_3.jpg',
            ]
        },
        '64557sc': {
            'floor_plan': 'floor_plans2/64557sc_floorplan_1.jpg',
            'exteriors': [
                'house_images2/64557sc_house_1.jpg',
                # ... more photos
            ]
        }
        # Add more houses...
    }

    for house_id, data in houses.items():
        try:
            final_model = process_house(
                data['floor_plan'],
                data['exteriors'],
                house_id
            )
            print(f"✓ {house_id}: {final_model}")
        except Exception as e:
            print(f"✗ {house_id} failed: {e}")
