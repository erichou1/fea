"""Deterministic family-level partition manifests with leakage rejection."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from typing import Dict, Iterable, Mapping, Sequence


PARTITIONS = ("fit", "development", "calibration", "confirmation")
DEFAULT_FRACTIONS = {
    "fit": 0.60,
    "development": 0.20,
    "calibration": 0.10,
    "confirmation": 0.10,
}


class FamilyLeakageError(ValueError):
    """A base family occurs in more than one data role or is unassigned."""


def _family_samples(samples: Iterable[Mapping[str, object]]) -> Dict[str, list]:
    by_family = defaultdict(list)
    seen_samples = set()
    for sample in samples:
        family_id = sample.get("family_id")
        sample_id = sample.get("sample_id")
        if not isinstance(family_id, str) or not family_id:
            raise FamilyLeakageError("each sample requires a non-empty explicit family_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise FamilyLeakageError("each sample requires a non-empty sample_id")
        if sample_id in seen_samples:
            raise FamilyLeakageError("duplicate sample_id: {}".format(sample_id))
        seen_samples.add(sample_id)
        by_family[family_id].append(sample_id)
    if not by_family:
        raise FamilyLeakageError("split input has no families")
    return dict(by_family)


def _family_partition_counts(total: int, fractions: Mapping[str, float]) -> Dict[str, int]:
    if set(fractions) != set(PARTITIONS) or abs(sum(fractions.values()) - 1.0) > 1e-12:
        raise ValueError("fractions must name every partition and sum to 1")
    positive_roles = [name for name in PARTITIONS if fractions[name] > 0]
    if total < len(positive_roles):
        raise FamilyLeakageError(
            "split requires at least {} families for positive functional roles".format(len(positive_roles))
        )
    raw = {name: total * fractions[name] for name in PARTITIONS}
    counts = {name: int(raw[name]) for name in PARTITIONS}
    remainder = total - sum(counts.values())
    for name in sorted(PARTITIONS, key=lambda item: (raw[item] - counts[item], item), reverse=True)[:remainder]:
        counts[name] += 1
    return counts


def build_family_split(
    samples: Iterable[Mapping[str, object]],
    *,
    seed: int,
    fractions: Mapping[str, float] = DEFAULT_FRACTIONS,
) -> Dict[str, list]:
    """Create a stable split of sample IDs while keeping every family intact."""
    by_family = _family_samples(samples)
    family_ids = sorted(by_family)
    random.Random(seed).shuffle(family_ids)
    counts = _family_partition_counts(len(family_ids), fractions)
    manifest = {name: [] for name in PARTITIONS}
    cursor = 0
    for partition in PARTITIONS:
        assigned = family_ids[cursor : cursor + counts[partition]]
        cursor += counts[partition]
        for family_id in assigned:
            manifest[partition].extend(sorted(by_family[family_id]))
    validate_family_split(
        [
            {"sample_id": sample_id, "family_id": family_id}
            for family_id, sample_ids in by_family.items()
            for sample_id in sample_ids
        ],
        manifest,
    )
    return manifest


def build_family_split_manifest(
    samples: Iterable[Mapping[str, object]],
    *,
    seed: int,
    fractions: Mapping[str, float] = DEFAULT_FRACTIONS,
) -> Dict[str, object]:
    """Return a self-describing deterministic manifest with explicit family IDs."""
    sample_list = list(samples)
    split = build_family_split(sample_list, seed=seed, fractions=fractions)
    family_by_sample = {
        str(sample["sample_id"]): str(sample["family_id"])
        for sample in sample_list
    }
    partitions = {
        partition: {
            "family_ids": sorted({family_by_sample[sample_id] for sample_id in split[partition]}),
            "sample_ids": list(split[partition]),
        }
        for partition in PARTITIONS
    }
    return {
        "schema_version": "1.0.0",
        "algorithm": "family-id-v1",
        "seed": seed,
        "fractions": {name: fractions[name] for name in PARTITIONS},
        "partitions": partitions,
    }


def validate_family_split(
    samples: Iterable[Mapping[str, object]], split: Mapping[str, Sequence[str]]
) -> None:
    """Fail closed unless every source sample is assigned once and families never cross roles."""
    if set(split) != set(PARTITIONS):
        raise FamilyLeakageError("split must contain exactly: {}".format(", ".join(PARTITIONS)))
    by_family = _family_samples(samples)
    family_by_sample = {
        sample_id: family_id for family_id, sample_ids in by_family.items() for sample_id in sample_ids
    }
    assigned = {}
    for partition in PARTITIONS:
        for sample_id in split[partition]:
            if sample_id not in family_by_sample:
                raise FamilyLeakageError("split references unknown sample_id: {}".format(sample_id))
            if sample_id in assigned:
                raise FamilyLeakageError("sample_id appears in multiple partitions: {}".format(sample_id))
            assigned[sample_id] = partition
    empty_roles = [partition for partition in PARTITIONS if not split[partition]]
    if empty_roles:
        raise FamilyLeakageError(
            "{} must contain at least one family".format(", ".join(empty_roles))
        )
    missing = sorted(set(family_by_sample) - set(assigned))
    if missing:
        raise FamilyLeakageError("split leaves source samples unassigned: {}".format(", ".join(missing)))
    family_partitions = defaultdict(set)
    for sample_id, partition in assigned.items():
        family_partitions[family_by_sample[sample_id]].add(partition)
    leaked = sorted(family_id for family_id, roles in family_partitions.items() if len(roles) != 1)
    if leaked:
        raise FamilyLeakageError("family appears in multiple partitions: {}".format(", ".join(leaked)))


def split_sha256(split: Mapping[str, object]) -> str:
    """SHA-256 of a canonical JSON split payload, including manifest metadata when present."""
    canonical = json.dumps(split, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
