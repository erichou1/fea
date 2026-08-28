"""G2 dense-surrogate data and training contracts.

Confirmation is intentionally not representable by a dataset handle.  Every role
is admitted before archive access; inputs are separately SHA-256 anchored and
case targets are accepted only after their G1b case digest verifies.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np

from .manifest import ManifestVerificationError, open_new_artifact_root, read_regular_path_snapshot, sha256_file, write_new_regular_path
from .splits import PARTITIONS, validate_family_split_manifest

TARGET_NAMES = ("compliance", "max_von_mises", "max_displacement")
_ARCHIVE_PREFIX = "fea_ml/data/runs_real/"


class SurrogateError(ValueError):
    """An anchored G2 data, training, or artifact contract was violated."""


class SurrogateRoleError(SurrogateError):
    """A requested role would violate G2's sealed-partition contract."""


@dataclass(frozen=True)
class SurrogateExample:
    sample_id: str
    channels: np.ndarray
    targets: dict[str, float]
    packed_occupancy_nbytes: int


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _lower_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise SurrogateError("{} must be an exact lowercase SHA-256 digest".format(label))
    return value


def _snapshot_json(path: Path, expected: str, label: str) -> tuple[dict[str, object], str]:
    _lower_digest(expected, label + " anchor")
    try:
        snapshot = read_regular_path_snapshot(path, label)
        value = json.loads(snapshot.bytes.decode("utf-8"))
    except (ManifestVerificationError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SurrogateError("{} is unavailable or malformed".format(label)) from error
    if snapshot.sha256 != expected:
        raise SurrogateError("{} sha256 mismatch".format(label))
    if not isinstance(value, dict):
        raise SurrogateError("{} must be an object".format(label))
    return value, snapshot.sha256



def _load_verified_case(root: Path, sample_id: str) -> dict[str, object]:
    """Read one G1b case only after its embedded canonical digest validates."""
    if not isinstance(sample_id, str) or not sample_id or any(character in sample_id for character in "/\\\x00"):
        raise SurrogateError("G1b case sample ID is unsafe")
    try:
        snapshot = read_regular_path_snapshot(root / "cases" / (sample_id + ".json"), "G1b case")
        case = json.loads(snapshot.bytes.decode("utf-8"))
    except (ManifestVerificationError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SurrogateError("G1b case is unavailable or malformed") from error
    if not isinstance(case, dict) or case.get("sample_id") != sample_id:
        raise SurrogateError("G1b case identity is invalid")
    if case.get("case_digest") != _digest({key: value for key, value in case.items() if key != "case_digest"}):
        raise SurrogateError("G1b case digest mismatch")
    return case


def _role_ids(split: Mapping[str, object], role: str) -> list[str]:
    try:
        validate_family_split_manifest(split)
        ids = split["partitions"][role]["sample_ids"]  # type: ignore[index]
    except (KeyError, TypeError, ValueError) as error:
        raise SurrogateError("frozen split is invalid") from error
    if not isinstance(ids, list) or not ids or any(not isinstance(item, str) or not item for item in ids):
        raise SurrogateError("{} role membership is invalid".format(role))
    return sorted(ids)


def _cohort_roles(cohort: Mapping[str, object], clusters: Mapping[str, object], split: Mapping[str, object]) -> dict[str, str]:
    records = cohort.get("records")
    eligible = cohort.get("eligible_ids")
    cluster_rows = clusters.get("clusters")
    if not isinstance(records, list) or not isinstance(eligible, list) or not isinstance(cluster_rows, list):
        raise SurrogateError("G1b cohort or cluster role manifest is malformed")
    eligible_ids = set(eligible)
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in eligible_ids):
        raise SurrogateError("G1b eligible IDs are invalid")
    cohort_by_id: dict[str, Mapping[str, object]] = {}
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("sample_id"), str):
            raise SurrogateError("G1b cohort record is malformed")
        sample_id = record["sample_id"]
        if sample_id in cohort_by_id:
            raise SurrogateError("G1b cohort has duplicate sample IDs")
        cohort_by_id[sample_id] = record
    cluster_role: dict[str, str] = {}
    for row in cluster_rows:
        if not isinstance(row, Mapping) or row.get("role") not in PARTITIONS or not isinstance(row.get("members"), list):
            raise SurrogateError("G1b cluster role manifest is malformed")
        for sample_id in row["members"]:
            if not isinstance(sample_id, str) or sample_id in cluster_role:
                raise SurrogateError("G1b cluster membership is invalid")
            cluster_role[sample_id] = str(row["role"])
    split_roles = {sample_id: role for role in PARTITIONS for sample_id in _role_ids(split, role)}
    if set(split_roles) != set(cohort_by_id) or set(split_roles) != set(cluster_role):
        raise SurrogateError("G1b cohort, cluster roles, and frozen split do not cover the same IDs")
    admitted: dict[str, str] = {}
    for sample_id in sorted(eligible_ids):
        record = cohort_by_id.get(sample_id)
        if record is None or record.get("exclusion_reasons") != []:
            raise SurrogateError("eligible G1b record is inconsistent")
        role = split_roles[sample_id]
        if record.get("role") != role or cluster_role[sample_id] != role:
            raise SurrogateError("G1b role disagreement")
        admitted[sample_id] = role
    return admitted


def _read_npz_member(archive: zipfile.ZipFile, sample_id: str, leaves: tuple[str, ...], label: str) -> np.ndarray:
    last_error: Exception | None = None
    for leaf in leaves:
        try:
            with archive.open(_ARCHIVE_PREFIX + sample_id + "/" + leaf, "r") as opened:
                payload = opened.read()
            with np.load(io.BytesIO(payload), allow_pickle=False) as loaded:
                if loaded.files != ["data"]:
                    raise SurrogateError("{} NPZ must contain exactly data".format(label))
                return loaded["data"]
        except KeyError as error:
            last_error = error
    raise SurrogateError("{} payload is missing".format(label)) from last_error


def _targets_from_case(case: Mapping[str, object]) -> dict[str, float]:
    solver = case.get("solver")
    if not isinstance(solver, Mapping) or solver.get("status") != "success":
        raise SurrogateError("G1b case is not a successful canonical baseline")
    raw = {
        "compliance": solver.get("compliance_j"),
        "max_von_mises": solver.get("max_gauss_von_mises_pa", solver.get("max_von_mises_pa")),
        "max_displacement": solver.get("max_displacement_m"),
    }
    try:
        values = {name: float(value) for name, value in raw.items()}
    except (TypeError, ValueError) as error:
        raise SurrogateError("G1b canonical target schema is invalid") from error
    if not all(math.isfinite(value) and value > 0.0 for value in values.values()):
        raise SurrogateError("G1b canonical targets must be finite and positive")
    return values


class RoleDataset:
    """Lazy role-scoped archive reader with packed occupancy cache."""

    def __init__(self, *, role: str, sample_ids: list[str], archive: Path, g1b_root: Path, provenance: Mapping[str, str]) -> None:
        self.role = role
        self.sample_ids = tuple(sample_ids)
        self._archive = archive
        self._g1b_root = g1b_root
        self.provenance = dict(provenance)
        self._packed: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _payload(self, sample_id: str) -> tuple[np.ndarray, np.ndarray]:
        cached = self._packed.get(sample_id)
        if cached is not None:
            return cached
        try:
            with zipfile.ZipFile(self._archive, "r") as archive:
                occupancy = _read_npz_member(archive, sample_id, ("occ.npz",), "occupancy")
                parts = _read_npz_member(archive, sample_id, ("part.npz", "parts.npz"), "part labels")
        except (OSError, zipfile.BadZipFile) as error:
            raise SurrogateError("anchored archive cannot be opened") from error
        if occupancy.shape != (64, 64, 64) or occupancy.dtype not in (np.dtype(np.uint8), np.dtype(np.bool_)):
            raise SurrogateError("occupancy has an invalid 64-cubed binary schema")
        if parts.shape != (64, 64, 64) or parts.dtype not in (np.dtype(np.uint8), np.dtype(np.int8)):
            raise SurrogateError("part labels have an invalid 64-cubed integer schema")
        if not np.all((occupancy == 0) | (occupancy == 1)) or not np.all((parts >= 0) & (parts <= 5)):
            raise SurrogateError("occupancy or part-label value is outside the frozen schema")
        packed = np.packbits(occupancy.reshape(-1), bitorder="little")
        compact_parts = parts.astype(np.uint8, copy=True)
        self._packed[sample_id] = (packed, compact_parts)
        return self._packed[sample_id]

    def __iter__(self) -> Iterator[SurrogateExample]:
        for sample_id in self.sample_ids:
            packed, parts = self._payload(sample_id)
            try:
                case = _load_verified_case(self._g1b_root, sample_id)
            except SurrogateError as error:
                raise SurrogateError("G1b case digest verification failed") from error
            if case.get("role") != self.role or case.get("exclusion_reasons") != []:
                raise SurrogateError("G1b case is outside this eligible role")
            occupancy = np.unpackbits(packed, bitorder="little", count=64 ** 3).reshape(64, 64, 64)
            channels = np.stack((occupancy.astype(np.float32), parts.astype(np.float32)), axis=0)
            yield SurrogateExample(sample_id=sample_id, channels=channels, targets=_targets_from_case(case), packed_occupancy_nbytes=int(packed.nbytes))



def compute_fit_normalization(dataset: RoleDataset) -> dict[str, object]:
    """Compute the sole permitted target normalization record from fit examples."""
    if not isinstance(dataset, RoleDataset) or dataset.role != "fit":
        raise SurrogateRoleError("normalization statistics may be computed from fit only")
    examples = list(dataset)
    if not examples:
        raise SurrogateError("fit role has no examples for normalization")
    transformed = {name: np.array([math.log(example.targets[name]) for example in examples], dtype=np.float64) for name in TARGET_NAMES}
    record: dict[str, object] = {
        "schema_version": "1.0.0", "role": "fit", "source_sample_ids": [example.sample_id for example in examples],
        "source_sample_count": len(examples), "split_manifest_sha256": dataset.provenance["split_manifest_sha256"],
        "archive_sha256": dataset.provenance["archive_sha256"], "cohort_manifest_sha256": dataset.provenance["cohort_manifest_sha256"],
        "cluster_role_manifest_sha256": dataset.provenance["cluster_role_manifest_sha256"], "target_names": list(TARGET_NAMES),
        "transform": {"name": "natural_log", "domain": "strictly_positive", "clipping": "none"},
        "means": {name: float(values.mean()) for name, values in transformed.items()},
        "scales": {name: max(float(values.std(ddof=0)), 1e-6) for name, values in transformed.items()},
    }
    record["stats_digest"] = _digest(record)
    return record



def _require_torch():
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as error:
        raise SurrogateError("PyTorch is required; install the locked ml dependency group") from error
    return torch, nn


def DenseSurrogateCNN(*, target_names: tuple[str, ...] = TARGET_NAMES, base_channels: int = 16):
    """Create a compact dense 3-D CNN with heteroscedastic named-target head.

    The factory keeps importing :mod:`sasto.surrogate` possible in the solver-only
    environment; PyTorch is loaded only when model construction is requested.
    """
    torch, nn = _require_torch()
    if not target_names or len(set(target_names)) != len(target_names) or any(not isinstance(name, str) or not name for name in target_names):
        raise SurrogateError("model target names must be unique non-empty strings")
    if not isinstance(base_channels, int) or isinstance(base_channels, bool) or base_channels < 2:
        raise SurrogateError("base_channels must be an integer of at least two")

    class _DenseSurrogateCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            c = base_channels
            def block(input_channels: int, output_channels: int, stride: int) -> nn.Sequential:
                return nn.Sequential(nn.Conv3d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                                     nn.GroupNorm(1, output_channels), nn.SiLU())
            self.features = nn.Sequential(block(2, c, 2), block(c, 2 * c, 2), block(2 * c, 4 * c, 2), block(4 * c, 4 * c, 2), nn.AdaptiveAvgPool3d(1))
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(4 * c, 4 * c), nn.SiLU(), nn.Linear(4 * c, 2 * len(target_names)))
            self.target_names = tuple(target_names)

        @property
        def parameter_count(self) -> int:
            return sum(parameter.numel() for parameter in self.parameters())

        def forward(self, channels):
            if channels.ndim != 5 or channels.shape[1] != 2:
                raise SurrogateError("model input must be [batch, 2, 64, 64, 64]")
            raw = self.head(self.features(channels))
            mean, raw_scale = raw.chunk(2, dim=1)
            return {"mean": mean, "dispersion": torch.nn.functional.softplus(raw_scale) + 1e-6}

    return _DenseSurrogateCNN()


def deterministic_seed(namespace: str, campaign_seed: int, member_index: int) -> int:
    """Derive a stable non-Python-hash seed for one independently trained member."""
    if not isinstance(namespace, str) or not namespace or not isinstance(campaign_seed, int) or not isinstance(member_index, int) or member_index < 0:
        raise SurrogateError("seed namespace, campaign seed, and member index are invalid")
    payload = "{}\0{}\0{}".format(namespace, campaign_seed, member_index).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2 ** 31 - 1)



def development_early_stopping_epoch(development_losses, *, patience: int) -> int:
    """Select the 1-indexed epoch using DEVELOPMENT metrics only.

    Callers pass one scalar per completed epoch; calibration and confirmation
    objects are deliberately absent from this API to make them unusable for a
    stopping decision.
    """
    if not isinstance(patience, int) or isinstance(patience, bool) or patience < 1:
        raise SurrogateError("development early-stopping patience must be positive")
    values = list(development_losses)
    if not values:
        raise SurrogateError("development early stopping requires at least one loss")
    best_loss = float("inf"); best_epoch = 0; stale = 0
    for epoch, loss in enumerate(values, start=1):
        loss = float(loss)
        if not math.isfinite(loss):
            raise SurrogateError("development loss must be finite")
        if loss < best_loss:
            best_loss = loss; best_epoch = epoch; stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    return best_epoch


def _write_new_json(path: Path, value: Mapping[str, object], role: str) -> str:
    try:
        return write_new_regular_path(path, _canonical_bytes(dict(value)) + b"\n", role)
    except ManifestVerificationError as error:
        raise SurrogateError("cannot append {}".format(role)) from error


def train_smoke_ensemble(
    *, output_root: Path, examples, target_names: tuple[str, ...], normalization_stats_digest: str, source_bundle_sha256: str,
    split_sha256: str, archive_sha256: str, cohort_manifest_sha256: str, member_count: int = 5, epochs: int = 1,
    base_channels: int = 4, device: str = "cpu", campaign_seed: int = 20260828,
) -> dict[str, object]:
    """Train a bounded synthetic smoke ensemble; it is explicitly nonpromotable.

    This path intentionally accepts in-memory examples only.  It cannot be pointed
    at the real cohort and therefore cannot bypass the pending G1b PASS gate.
    """
    torch, _ = _require_torch()
    if not 1 <= len(examples) <= 32 or not 1 <= member_count <= 5 or not 1 <= epochs <= 3:
        raise SurrogateError("SMOKE_ONLY_NONPROMOTABLE requires 1..32 examples, 1..5 members, and 1..3 epochs")
    for digest, label in ((normalization_stats_digest, "normalization stats"), (source_bundle_sha256, "source bundle"),
                          (split_sha256, "split"), (archive_sha256, "archive"), (cohort_manifest_sha256, "cohort manifest")):
        _lower_digest(digest, label + " digest")
    if device not in {"cpu", "mps"} or (device == "mps" and not torch.backends.mps.is_available()):
        raise SurrogateError("requested smoke device is unavailable")
    converted = []
    for sample_id, channels, targets in examples:
        if not isinstance(sample_id, str) or not sample_id or tuple(channels.shape) != (2, 64, 64, 64) or tuple(targets.shape) != (len(target_names),):
            raise SurrogateError("synthetic smoke example schema is invalid")
        converted.append((sample_id, channels.detach().to(dtype=torch.float32), targets.detach().to(dtype=torch.float32)))
    root = Path(output_root)
    try:
        with open_new_artifact_root(root) as root_fd:
            os.mkdir("members", dir_fd=root_fd)
    except (ManifestVerificationError, OSError) as error:
        raise SurrogateError("smoke artifact root must be new and append-only") from error
    members: list[dict[str, object]] = []
    for member_index in range(member_count):
        seed = deterministic_seed("sasto-v-g2-dense-ensemble-v1", campaign_seed, member_index)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if device == "mps":
            torch.mps.empty_cache()
        model = DenseSurrogateCNN(target_names=target_names, base_channels=base_channels).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        started = time.perf_counter(); steps = 0; total_loss = 0.0
        model.train()
        for _epoch in range(epochs):
            for _sample_id, channels, targets in converted:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(channels.unsqueeze(0).to(device))
                expected = targets.unsqueeze(0).to(device)
                dispersion = prediction["dispersion"]
                loss = (0.5 * ((expected - prediction["mean"]) / dispersion).square() + torch.log(dispersion)).mean()
                loss.backward(); optimizer.step()
                total_loss += float(loss.detach().cpu()); steps += 1
        wall_seconds = time.perf_counter() - started
        peak_memory = int(torch.mps.current_allocated_memory()) if device == "mps" else 0
        checkpoint_path = root / "members" / "member-{:02d}.pt".format(member_index)
        if checkpoint_path.exists():
            raise SurrogateError("checkpoint path must be append-only")
        torch.save({"state_dict": model.state_dict(), "target_names": target_names, "base_channels": base_channels, "seed": seed}, checkpoint_path)
        checkpoint_sha = sha256_file(checkpoint_path)
        ledger = {"wall_seconds": wall_seconds, "epochs": epochs, "steps": steps, "peak_memory_bytes": peak_memory,
                  "device": device, "samples_per_second": (len(converted) * epochs) / wall_seconds if wall_seconds else 0.0}
        manifest: dict[str, object] = {"schema_version": "1.0.0", "label": "SMOKE_ONLY_NONPROMOTABLE", "member_index": member_index,
            "seed_namespace": "sasto-v-g2-dense-ensemble-v1", "seed": seed, "source_bundle_sha256": source_bundle_sha256,
            "cohort_manifest_sha256": cohort_manifest_sha256, "split_sha256": split_sha256, "archive_sha256": archive_sha256,
            "normalization_stats_digest": normalization_stats_digest, "epoch_count": epochs, "target_names": list(target_names),
            "parameter_count": model.parameter_count, "checkpoint": {"path": checkpoint_path.name, "sha256": checkpoint_sha},
            "final_metrics": {"train_heteroscedastic_nll": total_loss / steps}, "compute_ledger": ledger}
        manifest["manifest_digest"] = _digest(manifest)
        manifest_sha = _write_new_json(root / "members" / "member-{:02d}.json".format(member_index), manifest, "member manifest")
        members.append({"member_index": member_index, "seed": seed, "manifest_sha256": manifest_sha, "checkpoint_sha256": checkpoint_sha,
                        "parameter_count": model.parameter_count, "samples_per_second": ledger["samples_per_second"]})
    summary: dict[str, object] = {"schema_version": "1.0.0", "label": "SMOKE_ONLY_NONPROMOTABLE", "member_count": member_count,
        "sample_count": len(converted), "members": members, "source_bundle_sha256": source_bundle_sha256,
        "split_sha256": split_sha256, "archive_sha256": archive_sha256, "cohort_manifest_sha256": cohort_manifest_sha256,
        "normalization_stats_digest": normalization_stats_digest}
    summary["summary_digest"] = _digest(summary)
    summary_sha = _write_new_json(root / "smoke-summary.json", summary, "smoke summary")
    return {"label": "SMOKE_ONLY_NONPROMOTABLE", "members": members, "summary_sha256": summary_sha, "output_root": str(root)}


def open_role_dataset(
    *,
    role: str,
    split_manifest: Path,
    expected_split_sha256: str,
    archive: Path,
    expected_archive_sha256: str,
    g1b_root: Path,
    expected_cohort_manifest_sha256: str,
    expected_cluster_role_manifest_sha256: str,
    calibration_pass: bool = False,
) -> RoleDataset:
    """Open one explicitly allowed G2 role with all source anchors verified."""
    if role == "confirmation":
        raise SurrogateRoleError("confirmation is sealed and cannot be opened by G2")
    if role not in {"fit", "development", "calibration"}:
        raise SurrogateRoleError("G2 role must be fit, development, or calibration")
    if role == "calibration" and not calibration_pass:
        raise SurrogateRoleError("calibration may be opened only for the calibration pass")
    if role != "calibration" and calibration_pass:
        raise SurrogateRoleError("calibration pass flag is valid only for calibration")
    split, split_sha = _snapshot_json(split_manifest, expected_split_sha256, "split manifest")
    cohort, cohort_sha = _snapshot_json(Path(g1b_root) / "cohort-manifest.json", expected_cohort_manifest_sha256, "cohort manifest")
    clusters, cluster_sha = _snapshot_json(Path(g1b_root) / "cluster-role-manifest.json", expected_cluster_role_manifest_sha256, "cluster role manifest")
    expected_archive_sha256 = _lower_digest(expected_archive_sha256, "archive anchor")
    try:
        archive_sha = sha256_file(archive)
    except ManifestVerificationError as error:
        raise SurrogateError("archive is unavailable") from error
    if archive_sha != expected_archive_sha256:
        raise SurrogateError("archive sha256 mismatch")
    admitted = _cohort_roles(cohort, clusters, split)
    sample_ids = [sample_id for sample_id in _role_ids(split, role) if sample_id in admitted]
    if not sample_ids:
        raise SurrogateError("role has no G1b-eligible samples")
    return RoleDataset(role=role, sample_ids=sample_ids, archive=Path(archive), g1b_root=Path(g1b_root), provenance={
        "split_manifest_sha256": split_sha, "archive_sha256": archive_sha,
        "cohort_manifest_sha256": cohort_sha, "cluster_role_manifest_sha256": cluster_sha,
    })


def surrogate_source_bundle(root: Path | None = None) -> tuple[dict[str, str], str]:
    """Hash every load-bearing G2 source/config byte through no-follow reads."""
    source_root = Path(root) if root is not None else Path(__file__).parents[2]
    paths = (".python-version", "pyproject.toml", "uv.lock", "src/sasto/surrogate.py", "src/sasto/manifest.py", "src/sasto/splits.py")
    files: dict[str, str] = {}
    try:
        for relative in paths:
            files[relative] = sha256_file(source_root / relative)
    except ManifestVerificationError as error:
        raise SurrogateError("cannot hash G2 source bundle") from error
    return files, _digest([{"path": path, "sha256": files[path]} for path in sorted(files)])


def _synthetic_smoke_examples():
    torch, _ = _require_torch()
    examples = []
    for index in range(4):
        channels = torch.zeros((2, 64, 64, 64), dtype=torch.float32)
        channels[0, index + 1, 1, 1] = 1.0; channels[1, index + 1, 1, 1] = float((index % 5) + 1)
        examples.append(("synthetic-smoke-{:02d}".format(index), channels, torch.tensor([0.1 * index, 0.2 * index, 0.3 * index], dtype=torch.float32)))
    return examples


def _read_smoke_summary(path: Path) -> dict[str, object]:
    try:
        snapshot = read_regular_path_snapshot(path, "smoke summary")
        summary = json.loads(snapshot.bytes.decode("utf-8"))
    except (ManifestVerificationError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SurrogateError("smoke summary is unavailable or malformed") from error
    if not isinstance(summary, dict) or summary.get("label") != "SMOKE_ONLY_NONPROMOTABLE":
        raise SurrogateError("smoke summary is not explicitly nonpromotable")
    if summary.get("summary_digest") != _digest({key: value for key, value in summary.items() if key != "summary_digest"}):
        raise SurrogateError("smoke summary digest mismatch")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="G2 dense surrogate ensemble (confirmation sealed)")
    parser.add_argument("--mode", choices=("prepare-stats", "train", "predict", "report"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-only-nonpromotable", action="store_true")
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--expected-split-manifest-sha256")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--expected-archive-sha256")
    parser.add_argument("--g1b-root", type=Path)
    parser.add_argument("--expected-cohort-manifest-sha256")
    parser.add_argument("--expected-cluster-role-manifest-sha256")
    args = parser.parse_args()
    try:
        if args.mode == "prepare-stats":
            if not all((args.split_manifest, args.expected_split_manifest_sha256, args.archive, args.expected_archive_sha256, args.g1b_root,
                        args.expected_cohort_manifest_sha256, args.expected_cluster_role_manifest_sha256)):
                raise SurrogateError("prepare-stats requires every anchored split, archive, and G1b manifest input")
            fit = open_role_dataset(role="fit", split_manifest=args.split_manifest, expected_split_sha256=args.expected_split_manifest_sha256,
                archive=args.archive, expected_archive_sha256=args.expected_archive_sha256, g1b_root=args.g1b_root,
                expected_cohort_manifest_sha256=args.expected_cohort_manifest_sha256,
                expected_cluster_role_manifest_sha256=args.expected_cluster_role_manifest_sha256)
            result = compute_fit_normalization(fit)
            _write_new_json(args.output, result, "normalization statistics")
        elif args.mode == "train":
            if not args.smoke_only_nonpromotable:
                raise SurrogateError("G1b independent certification is pending: train requires --smoke-only-nonpromotable (SMOKE_ONLY_NONPROMOTABLE); full fitting is blocked")
            files, bundle_sha = surrogate_source_bundle()
            del files
            synthetic_stats = _digest({"label": "SMOKE_ONLY_NONPROMOTABLE", "transform": "natural_log"})
            result = train_smoke_ensemble(output_root=args.output, examples=_synthetic_smoke_examples(), target_names=TARGET_NAMES,
                normalization_stats_digest=synthetic_stats, source_bundle_sha256=bundle_sha, split_sha256="0" * 64, archive_sha256="0" * 64,
                cohort_manifest_sha256="0" * 64, member_count=args.members, epochs=args.epochs, base_channels=2, device=args.device)
        else:
            summary = _read_smoke_summary(args.output / "smoke-summary.json")
            result = {"mode": args.mode, "label": summary["label"], "member_count": summary["member_count"],
                      "sample_count": summary["sample_count"], "summary_digest": summary["summary_digest"]}
    except (SurrogateError, OSError) as error:
        print("REJECTED: {}".format(error), file=__import__("sys").stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
