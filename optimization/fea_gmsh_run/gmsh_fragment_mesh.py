"""
gmsh_fragment_mesh.py - Fragment STEP geometry and generate tetrahedral mesh with Physical Groups.

Supports per-part labeling via fragment mapping to preserve part identity through boolean operations.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gmsh


def safe_remove_all_duplicates():
    """Remove duplicate entities, handling API variations."""
    try:
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()
    except Exception as e:
        print(f"[warn] removeAllDuplicates skipped: {e}")


def load_part_labels(path: Path) -> Dict[str, List[int]]:
    """
    Load part labels JSON: {part_name: [entity_tags_before_fragment]}.
    Example: {"exterior_wall": [1, 2], "roof": [3]}
    """
    with open(path, "r") as f:
        return json.load(f)


def map_parts_through_fragment(
    part_to_tags: Dict[str, List[int]],
    fragment_map: List[Tuple[Tuple[int, int], List[Tuple[int, int]]]],
) -> Dict[str, List[int]]:
    """
    Use Gmsh fragment mapping to track part identity through boolean operations.
    
    fragment_map: list of ((dim, parent_tag), [(dim, child_tag), ...])
    Returns: {part_name: [child_volume_tags]}
    """
    # Build parent -> children lookup (dim=3 volumes only)
    parent_to_children: Dict[int, List[int]] = {}
    for (dim, parent), children in fragment_map:
        if dim == 3:
            parent_to_children[parent] = [t for d, t in children if d == 3]
    
    # Map each part's original tags to their children after fragment
    result: Dict[str, List[int]] = {}
    for part_name, original_tags in part_to_tags.items():
        child_tags = []
        for tag in original_tags:
            if tag in parent_to_children:
                child_tags.extend(parent_to_children[tag])
            else:
                # Tag survived unchanged (or was removed - check if still exists)
                try:
                    if gmsh.model.occ.getMaxTag(3) >= tag:
                        # Check if entity exists
                        masses = gmsh.model.occ.getMass(3, tag)
                        if masses > 0:
                            child_tags.append(tag)
                except Exception:
                    pass
        result[part_name] = list(set(child_tags))  # dedupe
    
    return result


def export_physical_groups_json(
    path: Path,
    part_to_physical: Dict[str, Dict],
    physical_to_part: Dict[int, str],
):
    """
    Export bidirectional physical groups mapping.
    Format: {
        "by_id": {physical_id: part_name, ...},
        "by_name": {part_name: {dim: 3, physical_id: N}, ...}
    }
    """
    data = {
        "by_id": {str(k): v for k, v in physical_to_part.items()},
        "by_name": part_to_physical,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote physical groups mapping: {path}")


def main():
    ap = argparse.ArgumentParser(
        description="Fragment STEP geometry and generate mesh with Physical Groups"
    )
    ap.add_argument("step", type=str, help="Input STEP file")
    ap.add_argument("msh", type=str, help="Output mesh file")
    ap.add_argument("--h", type=float, default=0.10, help="Mesh element size")
    ap.add_argument("--msh-version", type=float, default=2.2, help="MSH file version")
    ap.add_argument("--algo3d", type=int, default=10,
                    help="3D meshing algorithm: 10=HXT, 1=Delaunay, 4=Frontal")
    ap.add_argument("--debug-brep", type=int, default=1,
                    help="Write debug .brep files")
    ap.add_argument("--part-labels-json", type=str, default=None,
                    help="JSON file with {part_name: [volume_tags]} before fragment")

    args = ap.parse_args()

    step_path = Path(args.step)
    msh_path = Path(args.msh)
    part_labels_path = Path(args.part_labels_json) if args.part_labels_json else None

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add(step_path.stem)

    # Import STEP geometry
    gmsh.model.occ.importShapes(str(step_path))
    gmsh.model.occ.synchronize()

    safe_remove_all_duplicates()

    vols = gmsh.model.getEntities(3)
    print(f"Imported volumes (dim=3): {len(vols)}")

    if not vols:
        if args.debug_brep:
            dbg = msh_path.with_suffix(".import_only.brep")
            gmsh.write(str(dbg))
            print("Wrote debug geometry:", dbg)
        gmsh.finalize()
        raise RuntimeError("No volumes in STEP (dim=3) after import (before fragment).")

    # Load part labels if provided
    part_to_tags: Optional[Dict[str, List[int]]] = None
    if part_labels_path and part_labels_path.exists():
        part_to_tags = load_part_labels(part_labels_path)
        print(f"Loaded part labels: {list(part_to_tags.keys())}")

    # Fragment all volumes against each other
    fragment_map = []
    if len(vols) > 1:
        try:
            # Fragment returns mapping of parent entities to children
            out_dimtags, out_map = gmsh.model.occ.fragment(
                vols, [], removeObject=True, removeTool=True
            )
            fragment_map = list(zip(vols, out_map))
        except TypeError:
            # Older API without return values
            gmsh.model.occ.fragment(vols, [])
        
        gmsh.model.occ.synchronize()
        safe_remove_all_duplicates()
        vols = gmsh.model.getEntities(3)
        print(f"Volumes after fragment: {len(vols)}")

    if args.debug_brep:
        dbg = msh_path.with_suffix(".brep")
        gmsh.write(str(dbg))
        print("Wrote debug geometry:", dbg)

    # Create Physical Groups
    all_vol_tags = [t for (d, t) in vols]
    
    # Track physical group assignments
    part_to_physical: Dict[str, Dict] = {}
    physical_to_part: Dict[int, str] = {}
    next_phys_id = 1

    if part_to_tags and fragment_map:
        # Map parts through fragment
        part_to_children = map_parts_through_fragment(part_to_tags, fragment_map)
        
        for part_name, child_tags in part_to_children.items():
            if child_tags:
                phys_id = next_phys_id
                next_phys_id += 1
                gmsh.model.addPhysicalGroup(3, child_tags, phys_id)
                gmsh.model.setPhysicalName(3, phys_id, part_name)
                part_to_physical[part_name] = {"dim": 3, "physical_id": phys_id}
                physical_to_part[phys_id] = part_name
                print(f"Physical Group {phys_id}: {part_name} -> {len(child_tags)} volumes")
    elif part_to_tags:
        # No fragment needed, use original tags directly
        for part_name, tags in part_to_tags.items():
            valid_tags = [t for t in tags if t in all_vol_tags]
            if valid_tags:
                phys_id = next_phys_id
                next_phys_id += 1
                gmsh.model.addPhysicalGroup(3, valid_tags, phys_id)
                gmsh.model.setPhysicalName(3, phys_id, part_name)
                part_to_physical[part_name] = {"dim": 3, "physical_id": phys_id}
                physical_to_part[phys_id] = part_name
                print(f"Physical Group {phys_id}: {part_name} -> {len(valid_tags)} volumes")

    # Always create "House" group containing all volumes (backward compatibility)
    house_phys_id = next_phys_id
    gmsh.model.addPhysicalGroup(3, all_vol_tags, house_phys_id)
    gmsh.model.setPhysicalName(3, house_phys_id, "House")
    part_to_physical["House"] = {"dim": 3, "physical_id": house_phys_id}
    physical_to_part[house_phys_id] = "House"
    print(f"Physical Group {house_phys_id}: House -> {len(all_vol_tags)} volumes (all)")

    # Export physical groups mapping
    phys_json_path = msh_path.with_name(msh_path.stem + "_physical_groups.json")
    export_physical_groups_json(phys_json_path, part_to_physical, physical_to_part)

    # Mesh settings
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", float(args.h))
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", float(args.h))
    gmsh.option.setNumber("Mesh.MshFileVersion", float(args.msh_version))
    gmsh.option.setNumber("Mesh.Algorithm3D", int(args.algo3d))

    # Generate 3D mesh
    gmsh.model.mesh.generate(3)

    # Save mesh - CRITICAL: Only save volumes (dim=3) to avoid SfePy confusion
    msh_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Option 1: SaveAll=1 saves physical groups (needed)
    # But we must ensure only 3D elements are included
    gmsh.option.setNumber("Mesh.SaveAll", 1)
    
    # Option 2: Explicitly save only dimension 3
    # Set to 0 to suppress saving of elements from lower dimensions
    gmsh.option.setNumber("Mesh.SaveElementTagType", 1)  # Save by physical group
    
    # CRITICAL FIX: Only save 3D elements
    # Remove lower-dimensional elements from the mesh before saving
    # This prevents SfePy from getting confused about topology
    try:
        # Get all entities
        edges = gmsh.model.getEntities(1)
        faces = gmsh.model.getEntities(2)
        
        # Remove mesh on lower-dimensional entities (keep geometry, remove discretization)
        for dim, tag in edges + faces:
            gmsh.model.mesh.clear([(dim, tag)])
    except Exception as e:
        print(f"Warning: Could not clear lower-dimensional meshes: {e}")
    
    gmsh.write(str(msh_path))
    gmsh.finalize()
    print("Wrote mesh:", msh_path)


if __name__ == "__main__":
    main()
