"""
Generate part label JSONs for STEP files based on STL part naming.
Maps STL parts to STEP volume indices for label preservation during meshing.
"""
import argparse
import json
from pathlib import Path
import gmsh
from typing import Dict, List


def extract_sample_id(filename: str) -> str:
    """Extract sample ID from filename like '00000_exterior_walls.stl'"""
    return filename.split('_')[0]


def get_part_type(filename: str) -> str:
    """Get part type from filename"""
    stem = Path(filename).stem
    if 'exterior_wall' in stem:
        return 'exterior_walls'
    elif 'interior_room' in stem or 'interior_wall' in stem:
        return 'interior_rooms'
    elif 'roof' in stem:
        return 'roof'
    elif 'attic_floor' in stem:
        return 'attic_floor'
    elif 'floor' in stem and 'attic' not in stem:
        return 'floor'
    return None


def analyze_step_volumes(step_path: Path) -> Dict[str, List[int]]:
    """
    Load STEP file and try to infer part labels from volume positions/sizes.
    This is a heuristic approach since STEP files don't embed our part names.
    
    Returns: {part_name: [volume_tags]}
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    
    try:
        gmsh.model.add(step_path.stem)
        gmsh.model.occ.importShapes(str(step_path))
        gmsh.model.occ.synchronize()
        
        volumes = gmsh.model.getEntities(3)
        
        # Get bounding boxes and volumes for heuristic classification
        volume_info = []
        for dim, tag in volumes:
            bbox = gmsh.model.occ.getBoundingBox(dim, tag)
            mass = gmsh.model.occ.getMass(dim, tag)
            center_z = (bbox[2] + bbox[5]) / 2  # Z-center
            height = bbox[5] - bbox[2]
            
            volume_info.append({
                'tag': tag,
                'bbox': bbox,
                'mass': mass,
                'center_z': center_z,
                'height': height
            })
        
        # Heuristic classification (simple but effective)
        # Sort by Z position
        volume_info.sort(key=lambda x: x['center_z'])
        
        total_volumes = len(volume_info)
        
        # Typical house structure from bottom to top:
        # - Floor (lowest ~10-20%)
        # - Walls + Interior (middle ~60-70%)  
        # - Roof (highest ~10-20%)
        
        part_labels = {
            'floor': [],
            'exterior_walls': [],
            'interior_rooms': [],
            'roof': [],
            'attic_floor': []
        }
        
        if total_volumes <= 3:
            # Simple case: floor, walls, roof
            if total_volumes >= 1:
                part_labels['floor'] = [volume_info[0]['tag']]
            if total_volumes >= 2:
                part_labels['exterior_walls'] = [volume_info[1]['tag']]
            if total_volumes >= 3:
                part_labels['roof'] = [volume_info[2]['tag']]
        else:
            # More complex structure
            # Bottom 15% -> floor
            floor_cutoff = int(total_volumes * 0.15)
            if floor_cutoff == 0:
                floor_cutoff = 1
            
            # Top 15% -> roof/attic
            roof_cutoff = total_volumes - max(1, int(total_volumes * 0.15))
            
            part_labels['floor'] = [v['tag'] for v in volume_info[:floor_cutoff]]
            part_labels['roof'] = [v['tag'] for v in volume_info[roof_cutoff:]]
            
            # Middle volumes: classify by mass (heavier = exterior walls)
            middle_volumes = volume_info[floor_cutoff:roof_cutoff]
            if middle_volumes:
                middle_volumes.sort(key=lambda x: x['mass'], reverse=True)
                
                # Largest 40% by mass -> exterior walls
                # Remaining -> interior rooms
                exterior_cutoff = max(1, int(len(middle_volumes) * 0.4))
                
                part_labels['exterior_walls'] = [v['tag'] for v in middle_volumes[:exterior_cutoff]]
                part_labels['interior_rooms'] = [v['tag'] for v in middle_volumes[exterior_cutoff:]]
        
        # Remove empty labels
        part_labels = {k: v for k, v in part_labels.items() if v}
        
        return part_labels
        
    finally:
        gmsh.finalize()


def load_stl_part_structure(parts_dir: Path, sample_id: str) -> Dict[str, int]:
    """
    Check which STL parts exist for this sample.
    Returns: {part_name: count}
    """
    parts = {}
    for stl_file in parts_dir.glob(f"{sample_id}_*.stl"):
        part_type = get_part_type(stl_file.name)
        if part_type:
            parts[part_type] = parts.get(part_type, 0) + 1
    return parts


def main():
    parser = argparse.ArgumentParser(
        description="Generate part label JSONs for STEP files based on heuristics"
    )
    parser.add_argument("--step-dir", type=str, required=True, help="Directory with STEP files")
    parser.add_argument("--stl-parts-dir", type=str, default=None, 
                       help="Optional: STL parts dir for validation")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for label JSONs")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N files")
    
    args = parser.parse_args()
    
    step_dir = Path(args.step_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stl_parts_dir = Path(args.stl_parts_dir) if args.stl_parts_dir else None
    
    step_files = sorted(step_dir.glob("*.step"))
    if args.limit:
        step_files = step_files[:args.limit]
    
    print(f"Generating part labels for {len(step_files)} STEP files...")
    
    success = 0
    for step_file in step_files:
        try:
            sample_id = step_file.stem.replace('_parts', '')
            
            # Generate labels from STEP geometry
            part_labels = analyze_step_volumes(step_file)
            
            if not part_labels:
                print(f"WARNING: No volumes found in {step_file.name}")
                continue
            
            # Optional: validate against STL structure
            if stl_parts_dir:
                stl_parts = load_stl_part_structure(stl_parts_dir, sample_id)
                if stl_parts and set(stl_parts.keys()) != set(part_labels.keys()):
                    print(f"  Note: {step_file.name} - STL parts: {list(stl_parts.keys())} "
                          f"vs inferred: {list(part_labels.keys())}")
            
            # Save JSON
            output_json = output_dir / f"{step_file.stem}_labels.json"
            with open(output_json, 'w') as f:
                json.dump(part_labels, f, indent=2)
            
            success += 1
            
            if success % 100 == 0:
                print(f"  Processed {success}/{len(step_files)}...")
                
        except Exception as e:
            print(f"ERROR processing {step_file.name}: {e}")
            continue
    
    print(f"\nCompleted: {success}/{len(step_files)} label files generated")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
