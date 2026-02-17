"""Tests for label preservation utilities."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from fea_ml.utils.label_verification import (
    verify_physical_groups_json,
    load_physical_groups_json,
    LabelVerificationResult,
)


@pytest.fixture
def sample_physical_groups_json():
    """Create a sample physical_groups.json file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "physical_groups.json"
        
        data = {
            "by_id": {
                "1": "exterior_wall",
                "2": "interior_wall",
                "3": "roof",
                "4": "floor",
                "5": "House",
            },
            "by_name": {
                "exterior_wall": {"dim": 3, "physical_id": 1},
                "interior_wall": {"dim": 3, "physical_id": 2},
                "roof": {"dim": 3, "physical_id": 3},
                "floor": {"dim": 3, "physical_id": 4},
                "House": {"dim": 3, "physical_id": 5},
            },
        }
        
        with open(path, "w") as f:
            json.dump(data, f)
        
        yield path


def test_load_physical_groups(sample_physical_groups_json):
    """Test loading physical groups JSON."""
    data = load_physical_groups_json(sample_physical_groups_json)
    
    assert "by_id" in data
    assert "by_name" in data
    assert data["by_id"]["1"] == "exterior_wall"
    assert data["by_name"]["roof"]["physical_id"] == 3


def test_verify_physical_groups_success(sample_physical_groups_json):
    """Test verification with all expected parts present."""
    result = verify_physical_groups_json(
        sample_physical_groups_json,
        expected_parts=["exterior_wall", "interior_wall", "roof", "floor"],
    )
    
    assert result.success
    assert len(result.missing_parts) == 0
    assert len(result.physical_groups) == 5


def test_verify_physical_groups_missing(sample_physical_groups_json):
    """Test verification with missing parts."""
    result = verify_physical_groups_json(
        sample_physical_groups_json,
        expected_parts=["exterior_wall", "attic", "basement"],
    )
    
    assert not result.success
    assert "attic" in result.missing_parts
    assert "basement" in result.missing_parts


def test_verify_nonexistent_file():
    """Test verification of non-existent file."""
    result = verify_physical_groups_json(
        Path("/nonexistent/path.json"),
        expected_parts=["exterior_wall"],
    )
    
    assert not result.success
    assert "not found" in result.message.lower()


def test_bidirectional_mapping(sample_physical_groups_json):
    """Test that bidirectional mapping is consistent."""
    data = load_physical_groups_json(sample_physical_groups_json)
    
    # Check consistency: by_id and by_name should match
    for id_str, name in data["by_id"].items():
        if name in data["by_name"]:
            assert data["by_name"][name]["physical_id"] == int(id_str)
