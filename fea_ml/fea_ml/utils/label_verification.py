"""
Label verification utilities for checking Physical Group preservation in meshes.

Verifies that part labels survive the STEP -> fragment -> mesh pipeline and
provides fallback strategies when labels are missing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class PhysicalGroupInfo:
    """Information about a physical group in a mesh."""
    dim: int
    tag: int
    name: str
    element_count: int


@dataclass 
class LabelVerificationResult:
    """Result of label verification check."""
    success: bool
    physical_groups: List[PhysicalGroupInfo]
    missing_parts: List[str]
    message: str


def load_physical_groups_json(path: Path) -> Dict:
    """
    Load bidirectional physical groups mapping.
    
    Expected format:
    {
        "by_id": {"1": "exterior_wall", "2": "roof", ...},
        "by_name": {"exterior_wall": {"dim": 3, "physical_id": 1}, ...}
    }
    """
    with open(path, "r") as f:
        return json.load(f)


def verify_mesh_labels(
    msh_path: Path,
    expected_parts: Optional[List[str]] = None,
) -> LabelVerificationResult:
    """
    Verify that a .msh file contains expected physical group tags.
    
    Args:
        msh_path: Path to the .msh file
        expected_parts: List of expected part names (optional)
        
    Returns:
        LabelVerificationResult with success status and details
    """
    try:
        import meshio
    except ImportError:
        return LabelVerificationResult(
            success=False,
            physical_groups=[],
            missing_parts=expected_parts or [],
            message="meshio not installed - cannot verify mesh labels"
        )
    
    try:
        mesh = meshio.read(msh_path)
    except Exception as e:
        return LabelVerificationResult(
            success=False,
            physical_groups=[],
            missing_parts=expected_parts or [],
            message=f"Failed to read mesh: {e}"
        )
    
    # Extract physical groups from mesh
    found_groups: List[PhysicalGroupInfo] = []
    found_names: Set[str] = set()
    
    # Check for gmsh:physical in cell_data
    if hasattr(mesh, "cell_data") and mesh.cell_data:
        for cell_type, data_dict in mesh.cell_data.items() if isinstance(mesh.cell_data, dict) else []:
            if "gmsh:physical" in data_dict:
                physical_tags = data_dict["gmsh:physical"]
                unique_tags = np.unique(physical_tags)
                for tag in unique_tags:
                    count = np.sum(physical_tags == tag)
                    found_groups.append(PhysicalGroupInfo(
                        dim=3 if "tetra" in cell_type.lower() else 2,
                        tag=int(tag),
                        name=f"group_{tag}",  # Name not stored in msh directly
                        element_count=int(count)
                    ))
    
    # Try reading from field_data (contains name -> (tag, dim) mapping)
    if hasattr(mesh, "field_data") and mesh.field_data:
        for name, (tag, dim) in mesh.field_data.items():
            found_names.add(name)
            # Update existing group with name if tag matches
            for group in found_groups:
                if group.tag == tag:
                    group.name = name
                    break
            else:
                # Group not found in cell_data, add placeholder
                found_groups.append(PhysicalGroupInfo(
                    dim=int(dim),
                    tag=int(tag),
                    name=name,
                    element_count=0
                ))
    
    # Check for missing expected parts
    missing_parts = []
    if expected_parts:
        for part in expected_parts:
            if part not in found_names and part != "House":
                missing_parts.append(part)
    
    success = len(found_groups) > 0 and len(missing_parts) == 0
    
    if not found_groups:
        message = "ERROR: No physical groups found in mesh. Labels were lost in pipeline."
    elif missing_parts:
        message = f"WARNING: Missing parts: {missing_parts}. Found: {list(found_names)}"
    else:
        message = f"OK: Found {len(found_groups)} physical groups: {list(found_names)}"
    
    return LabelVerificationResult(
        success=success,
        physical_groups=found_groups,
        missing_parts=missing_parts,
        message=message
    )


def verify_physical_groups_json(
    json_path: Path,
    expected_parts: Optional[List[str]] = None,
) -> LabelVerificationResult:
    """
    Verify that physical_groups.json contains expected parts.
    
    Args:
        json_path: Path to physical_groups.json
        expected_parts: List of expected part names
        
    Returns:
        LabelVerificationResult
    """
    if not json_path.exists():
        return LabelVerificationResult(
            success=False,
            physical_groups=[],
            missing_parts=expected_parts or [],
            message=f"physical_groups.json not found at {json_path}"
        )
    
    try:
        data = load_physical_groups_json(json_path)
    except Exception as e:
        return LabelVerificationResult(
            success=False,
            physical_groups=[],
            missing_parts=expected_parts or [],
            message=f"Failed to parse physical_groups.json: {e}"
        )
    
    found_groups = []
    by_name = data.get("by_name", {})
    
    for name, info in by_name.items():
        found_groups.append(PhysicalGroupInfo(
            dim=info.get("dim", 3),
            tag=info.get("physical_id", 0),
            name=name,
            element_count=0  # Not stored in JSON
        ))
    
    found_names = set(by_name.keys())
    missing_parts = []
    
    if expected_parts:
        for part in expected_parts:
            if part not in found_names:
                missing_parts.append(part)
    
    success = len(found_groups) > 0 and len(missing_parts) == 0
    
    if not found_groups:
        message = "ERROR: No physical groups in JSON"
    elif missing_parts:
        message = f"WARNING: Missing parts in JSON: {missing_parts}"
    else:
        message = f"OK: Found parts in JSON: {list(found_names)}"
    
    return LabelVerificationResult(
        success=success,
        physical_groups=found_groups,
        missing_parts=missing_parts,
        message=message
    )


def get_part_elements_from_mesh(
    msh_path: Path,
    physical_groups_json: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract element indices for each part from a mesh file.
    
    Args:
        msh_path: Path to .msh file
        physical_groups_json: Optional path to physical_groups.json for name mapping
        
    Returns:
        Dict mapping part names to arrays of element indices
    """
    import meshio
    
    mesh = meshio.read(msh_path)
    
    # Load name mapping if available
    tag_to_name: Dict[int, str] = {}
    if physical_groups_json and physical_groups_json.exists():
        data = load_physical_groups_json(physical_groups_json)
        tag_to_name = {int(k): v for k, v in data.get("by_id", {}).items()}
    
    # Also check mesh field_data
    if hasattr(mesh, "field_data") and mesh.field_data:
        for name, (tag, dim) in mesh.field_data.items():
            tag_to_name[int(tag)] = name
    
    result: Dict[str, np.ndarray] = {}
    
    # Process each cell block
    offset = 0
    for i, cells in enumerate(mesh.cells):
        cell_type = cells.type
        cell_data = cells.data
        n_cells = len(cell_data)
        
        # Get physical tags for this block
        if "gmsh:physical" in mesh.cell_data:
            physical_tags = mesh.cell_data["gmsh:physical"][i]
            unique_tags = np.unique(physical_tags)
            
            for tag in unique_tags:
                name = tag_to_name.get(int(tag), f"group_{tag}")
                mask = physical_tags == tag
                indices = np.where(mask)[0] + offset
                
                if name in result:
                    result[name] = np.concatenate([result[name], indices])
                else:
                    result[name] = indices
        
        offset += n_cells
    
    return result
