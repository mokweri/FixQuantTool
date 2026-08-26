"""Filesystem-backed registry for validated FixQuant model releases."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import yaml

SCHEMA_VERSION = 1
DEFAULT_ZOO_ROOT = Path(__file__).resolve().parents[2] / "model_zoo"
DEFAULT_RELEASE_REPOSITORY = "mokweri/FixQuantTool"
DEFAULT_POLICY = {
    "minimum_validation_samples": 50000,
    "maximum_tilecnn_top1_drop": 1.0,
    "maximum_tilecnn_top5_drop": 0.5,
    "required_artifacts": [
        "best_checkpoint",
        "calibration_report",
        "threshold_log",
        "qconfig",
    ],
}


class ZooError(RuntimeError):
    """Raised when registry validation or an immutable operation fails."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def get_zoo_root(root: Optional[os.PathLike | str] = None) -> Path:
    value = root or os.environ.get("FIXQUANT_ZOO_ROOT") or DEFAULT_ZOO_ROOT
    return Path(value).expanduser().resolve()


def get_cache_root(
    root: Optional[os.PathLike | str] = None,
    cache_root: Optional[os.PathLike | str] = None,
) -> Path:
    """Return the ignored checkpoint cache associated with a model zoo."""
    value = cache_root or os.environ.get("FIXQUANT_ZOO_CACHE")
    if value:
        return Path(value).expanduser().resolve()
    return get_zoo_root(root) / ".artifacts"


def select_dataset_path(
    explicit_path: Optional[str] = None,
    release_path: Optional[str] = None,
    *,
    environ: Optional[Dict[str, str]] = None,
    fallback: Optional[str] = None,
) -> Optional[str]:
    """Choose a dataset path without binding a release to its training host.

    Precedence is explicit CLI value, local environment, recorded release path,
    then the caller's fallback.
    """
    environment = os.environ if environ is None else environ
    return (
        explicit_path
        or environment.get("FIXQUANT_DATA_DIR")
        or release_path
        or fallback
    )


def sha256_file(path: os.PathLike | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: os.PathLike | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ZooError(f"Expected a mapping in {path}")
    return value


def load_json(path: os.PathLike | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ZooError(f"Expected an object in {path}")
    return value


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def write_yaml(path: os.PathLike | str, value: Dict[str, Any]) -> None:
    text = yaml.safe_dump(value, sort_keys=False, default_flow_style=False)
    _atomic_write(Path(path), text)


def write_json(path: os.PathLike | str, value: Dict[str, Any]) -> None:
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    _atomic_write(Path(path), text)


def git_commit(repo_root: os.PathLike | str) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _safe_name(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ZooError(f"Invalid {label}: {value!r}")
    return value


def _version(value: str) -> str:
    value = value if value.startswith("v") else f"v{value}"
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", value):
        raise ZooError(f"Version must use semantic form vMAJOR.MINOR.PATCH: {value}")
    return value


def _repository(value: str) -> str:
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*",
        value,
    ):
        raise ZooError(f"GitHub repository must use owner/name form: {value!r}")
    return value


def _relative_artifact_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ZooError(f"Unsafe release artifact path: {value!r}")
    return path


def candidate_directory(root: os.PathLike | str, candidate_id: str) -> Path:
    return get_zoo_root(root) / ".candidates" / _safe_name(candidate_id, "candidate ID")


def load_candidate(root: os.PathLike | str, candidate_id: str) -> Dict[str, Any]:
    path = candidate_directory(root, candidate_id) / "candidate.yaml"
    if not path.is_file():
        raise ZooError(f"Candidate not found: {candidate_id}")
    return load_yaml(path)


def save_candidate(root: os.PathLike | str, candidate: Dict[str, Any]) -> Path:
    path = candidate_directory(root, candidate["candidate_id"]) / "candidate.yaml"
    write_yaml(path, candidate)
    return path


def _first_match(paths: Iterable[Path]) -> Optional[Path]:
    return next((path for path in paths if path.is_file()), None)


def register_candidate(
    run_dir: os.PathLike | str,
    *,
    root: Optional[os.PathLike | str] = None,
    candidate_id: Optional[str] = None,
    model: Optional[str] = None,
    dataset_name: str = "imagenet1k",
    dataset_path: Optional[str] = None,
    profile: Optional[str] = None,
    cle: Optional[bool] = None,
) -> Tuple[Dict[str, Any], Path]:
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise ZooError(f"Run directory not found: {run_dir}")

    run_manifest_path = run_dir / "run_manifest.yaml"
    run_manifest = load_yaml(run_manifest_path) if run_manifest_path.is_file() else {}
    model_info = run_manifest.get("model", {})
    if isinstance(model_info, str):
        manifest_model = model_info
    else:
        manifest_model = model_info.get("name")
    model = model or manifest_model or run_dir.name
    model = _safe_name(model, "model")

    quantization = run_manifest.get("quantization", {})
    if cle is None:
        cle = bool(quantization.get("cle", False))
    profile = profile or ("int8-tqt-cle" if cle else "int8-tqt")
    profile = _safe_name(profile, "quantization profile")
    dataset_name = _safe_name(dataset_name, "dataset")

    checkpoint = run_dir / "checkpoint" / "model_best.pth.tar"
    if not checkpoint.is_file():
        raise ZooError(f"Genuine best checkpoint not found: {checkpoint}")

    job_id = str(
        run_manifest.get("provenance", {}).get("slurm_job_id")
        or run_dir.parent.name
    )
    candidate_id = candidate_id or f"{model}-run-{job_id}"
    candidate_id = _safe_name(candidate_id, "candidate ID")
    target_dir = candidate_directory(root, candidate_id)
    if target_dir.exists():
        raise ZooError(f"Candidate already exists: {candidate_id}")

    calibration_report = _first_match(run_dir.glob("*_calib_report.csv"))
    threshold_log = run_dir / "logs" / "quant_thresholds.csv"
    latest = run_dir / "checkpoint" / "latest.pth.tar"
    artifacts = {
        "best_checkpoint": str(checkpoint),
        "latest_checkpoint": str(latest) if latest.is_file() else None,
        "calibration_report": (
            str(calibration_report) if calibration_report else None
        ),
        "threshold_log": str(threshold_log) if threshold_log.is_file() else None,
        "run_manifest": str(run_manifest_path) if run_manifest_path.is_file() else None,
    }

    candidate = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "status": "candidate",
        "created_at": utc_now(),
        "model": model,
        "dataset": {
            "name": dataset_name,
            "path": dataset_path or run_manifest.get("dataset", {}).get("path"),
        },
        "quantization": {
            "profile": profile,
            "method": "tqt",
            "weight_bits": 8,
            "activation_bits": 8,
            "cle": bool(cle),
        },
        "source": {
            "run_dir": str(run_dir),
            "git_commit": run_manifest.get("provenance", {}).get("git_commit"),
            "slurm_job_id": job_id,
        },
        "artifacts": artifacts,
        "checkpoint_sha256": sha256_file(checkpoint),
        "evaluations": {},
    }
    target_dir.mkdir(parents=True, exist_ok=False)
    path = save_candidate(root, candidate)
    return candidate, path


def _metric_value(document: Dict[str, Any], name: str) -> float:
    metrics = document.get("metrics", {})
    if name not in metrics:
        raise ZooError(f"Metric {name!r} is missing")
    return float(metrics[name])


def validate_candidate(
    root: os.PathLike | str,
    candidate_id: str,
    *,
    qat_metrics_path: Optional[os.PathLike | str] = None,
    deploy_metrics_path: Optional[os.PathLike | str] = None,
    policy_path: Optional[os.PathLike | str] = None,
) -> Dict[str, Any]:
    candidate = load_candidate(root, candidate_id)
    directory = candidate_directory(root, candidate_id)
    evaluation_dir = directory / "evaluation"
    qat_path = Path(qat_metrics_path or evaluation_dir / "qat_metrics.json")
    deploy_path = Path(deploy_metrics_path or evaluation_dir / "tilecnn_metrics.json")
    if not qat_path.is_file() or not deploy_path.is_file():
        raise ZooError("Both QAT and TileCNN metrics are required")

    qat = load_json(qat_path)
    deploy = load_json(deploy_path)
    policy = dict(DEFAULT_POLICY)
    if policy_path:
        policy.update(load_yaml(policy_path))

    expected_hash = candidate["checkpoint_sha256"]
    qat_hash = qat.get("checkpoint_sha256")
    deploy_hash = deploy.get("checkpoint_sha256")
    validation_samples = min(
        int(qat.get("validation_samples", 0)),
        int(deploy.get("validation_samples", 0)),
    )
    qat_top1 = _metric_value(qat, "top1")
    qat_top5 = _metric_value(qat, "top5")
    deploy_top1 = _metric_value(deploy, "top1")
    deploy_top5 = _metric_value(deploy, "top5")
    top1_drop = qat_top1 - deploy_top1
    top5_drop = qat_top5 - deploy_top5
    qconfig_path = evaluation_dir / "qconfig.json"

    checks = [
        {
            "name": "qat_checkpoint_hash",
            "passed": qat_hash == expected_hash,
            "actual": qat_hash,
            "expected": expected_hash,
        },
        {
            "name": "tilecnn_checkpoint_hash",
            "passed": deploy_hash == expected_hash,
            "actual": deploy_hash,
            "expected": expected_hash,
        },
        {
            "name": "validation_samples",
            "passed": validation_samples >= int(policy["minimum_validation_samples"]),
            "actual": validation_samples,
            "minimum": int(policy["minimum_validation_samples"]),
        },
        {
            "name": "tilecnn_top1_drop",
            "passed": top1_drop <= float(policy["maximum_tilecnn_top1_drop"]),
            "actual": top1_drop,
            "maximum": float(policy["maximum_tilecnn_top1_drop"]),
        },
        {
            "name": "tilecnn_top5_drop",
            "passed": top5_drop <= float(policy["maximum_tilecnn_top5_drop"]),
            "actual": top5_drop,
            "maximum": float(policy["maximum_tilecnn_top5_drop"]),
        },
    ]
    artifact_paths = dict(candidate.get("artifacts", {}))
    artifact_paths["qconfig"] = str(qconfig_path) if qconfig_path.is_file() else None
    for artifact_name in policy.get("required_artifacts", []):
        artifact_path = artifact_paths.get(artifact_name)
        checks.append({
            "name": f"artifact_{artifact_name}",
            "passed": bool(artifact_path and Path(artifact_path).is_file()),
            "actual": artifact_path,
            "expected": "existing file",
        })
    passed = all(check["passed"] for check in checks)
    report = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "validated_at": utc_now(),
        "passed": passed,
        "policy": policy,
        "checks": checks,
        "metrics": {
            "qat": qat["metrics"],
            "tilecnn": deploy["metrics"],
            "tilecnn_delta": {
                "top1": deploy_top1 - qat_top1,
                "top5": deploy_top5 - qat_top5,
            },
        },
        "validation_samples": validation_samples,
        "checkpoint_sha256": expected_hash,
    }
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    stored_qat = evaluation_dir / "qat_metrics.json"
    stored_deploy = evaluation_dir / "tilecnn_metrics.json"
    if qat_path.resolve() != stored_qat.resolve():
        shutil.copy2(qat_path, stored_qat)
    if deploy_path.resolve() != stored_deploy.resolve():
        shutil.copy2(deploy_path, stored_deploy)
    write_json(evaluation_dir / "validation_report.json", report)

    candidate["status"] = "validated" if passed else "rejected"
    candidate["artifacts"] = artifact_paths
    candidate["evaluations"] = {
        "qat": str(stored_qat),
        "tilecnn": str(stored_deploy),
        "report": str(evaluation_dir / "validation_report.json"),
    }
    candidate["updated_at"] = utc_now()
    save_candidate(root, candidate)
    return report


def parse_release_id(release_id: str) -> Tuple[str, str, str, str]:
    try:
        coordinates, version = release_id.rsplit("@", 1)
        model, dataset, profile = coordinates.split("/")
    except ValueError as exc:
        raise ZooError(
            "Release ID must be model/dataset/profile@version"
        ) from exc
    return (
        _safe_name(model, "model"),
        _safe_name(dataset, "dataset"),
        _safe_name(profile, "profile"),
        _version(version),
    )


def release_directory(root: os.PathLike | str, release_id: str) -> Path:
    model, dataset, profile, version = parse_release_id(release_id)
    return get_zoo_root(root) / "releases" / model / dataset / profile / version


def github_release_tag(release_id: str) -> str:
    model, dataset, profile, version = parse_release_id(release_id)
    return f"model-zoo-{model}-{dataset}-{profile}-{version}"


def _github_download_record(
    release_id: str,
    repository: str,
    asset: str,
    size_bytes: int,
) -> Dict[str, Any]:
    repository = _repository(repository)
    asset = _safe_name(asset, "release asset")
    tag = github_release_tag(release_id)
    url = f"https://github.com/{repository}/releases/download/{tag}/{asset}"
    return {
        "provider": "github-release",
        "repository": repository,
        "tag": tag,
        "asset": asset,
        "urls": [url],
        "size_bytes": int(size_bytes),
    }


def _checkpoint_details(
    root: os.PathLike | str,
    release_id: str,
    manifest: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, str, str]:
    manifest = manifest or load_release(root, release_id)
    relative = manifest.get("artifacts", {}).get("best_checkpoint")
    if not isinstance(relative, str):
        raise ZooError(f"Release has no best checkpoint: {release_id}")
    relative_path = _relative_artifact_path(relative)
    expected = manifest.get("checksums", {}).get(relative)
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ZooError(f"Release has no valid checkpoint checksum: {release_id}")
    return release_directory(root, release_id) / relative_path, relative, expected


def _verified_file(path: Path, expected_sha256: str) -> bool:
    return path.is_file() and sha256_file(path) == expected_sha256


def checkpoint_cache_path(
    root: os.PathLike | str,
    release_id: str,
    relative: str,
    cache_root: Optional[os.PathLike | str] = None,
) -> Path:
    model, dataset, profile, version = parse_release_id(release_id)
    return (
        get_cache_root(root, cache_root)
        / model
        / dataset
        / profile
        / version
        / _relative_artifact_path(relative)
    )


def _copy_artifact(source: Optional[str], destination: Path) -> Optional[str]:
    if not source:
        return None
    source_path = Path(source)
    if not source_path.is_file():
        raise ZooError(f"Artifact disappeared before promotion: {source_path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    return destination.as_posix()


def _model_card(manifest: Dict[str, Any]) -> str:
    metrics = manifest["metrics"]
    return (
        f"# {manifest['model']} {manifest['version']}\n\n"
        f"- Dataset: `{manifest['dataset']['name']}`\n"
        f"- Quantization profile: `{manifest['quantization']['profile']}`\n"
        f"- QAT top-1/top-5: {metrics['qat']['top1']:.4f}% / "
        f"{metrics['qat']['top5']:.4f}%\n"
        f"- TileCNN top-1/top-5: {metrics['tilecnn']['top1']:.4f}% / "
        f"{metrics['tilecnn']['top5']:.4f}%\n"
        f"- Source commit: `{manifest['provenance'].get('git_commit')}`\n"
        f"- Source Slurm job: `{manifest['provenance'].get('slurm_job_id')}`\n"
    )


def promote_candidate(
    root: os.PathLike | str,
    candidate_id: str,
    version: str,
    *,
    repository: str = DEFAULT_RELEASE_REPOSITORY,
) -> Tuple[Dict[str, Any], Path]:
    root_path = get_zoo_root(root)
    candidate = load_candidate(root_path, candidate_id)
    version = _version(version)
    model = candidate["model"]
    dataset = candidate["dataset"]["name"]
    profile = candidate["quantization"]["profile"]
    release_id = f"{model}/{dataset}/{profile}@{version}"
    destination = release_directory(root_path, release_id)
    if destination.exists():
        raise ZooError(f"Immutable release already exists: {release_id}")
    if candidate.get("status") != "validated":
        raise ZooError(
            f"Candidate must be validated before promotion; status={candidate.get('status')}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{version}.", suffix=".staging", dir=destination.parent)
    )
    try:
        copied = {}
        copied["best_checkpoint"] = _copy_artifact(
            candidate["artifacts"]["best_checkpoint"],
            staging / "qat" / "model_best.pth.tar",
        )
        optional_destinations = {
            "calibration_report": staging / "qat" / "calibration_report.csv",
            "threshold_log": staging / "qat" / "quant_thresholds.csv",
            "run_manifest": staging / "run_manifest.yaml",
            "qconfig": staging / "deployment" / "qconfig.json",
        }
        for name, target in optional_destinations.items():
            copied[name] = _copy_artifact(candidate["artifacts"].get(name), target)

        evaluation_dir = candidate_directory(root_path, candidate_id) / "evaluation"
        for name in ("qat_metrics.json", "tilecnn_metrics.json", "validation_report.json"):
            _copy_artifact(str(evaluation_dir / name), staging / "evaluation" / name)

        report = load_json(evaluation_dir / "validation_report.json")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "release_id": release_id,
            "status": "released",
            "version": version,
            "released_at": utc_now(),
            "model": model,
            "dataset": candidate["dataset"],
            "quantization": candidate["quantization"],
            "provenance": candidate["source"],
            "metrics": report["metrics"],
            "validation_samples": report["validation_samples"],
            "artifacts": {
                key: (str(Path(path).relative_to(staging)) if path else None)
                for key, path in copied.items()
            },
        }
        write_json(staging / "metrics.json", report["metrics"])
        _atomic_write(staging / "model_card.md", _model_card(manifest))

        checksum_entries = []
        for path in sorted(staging.rglob("*")):
            if path.is_file() and path.name not in {"manifest.yaml", "checksums.sha256"}:
                checksum_entries.append(
                    (str(path.relative_to(staging)), sha256_file(path))
                )
        manifest["checksums"] = dict(checksum_entries)
        checkpoint_relative = manifest["artifacts"]["best_checkpoint"]
        checkpoint_path = staging / checkpoint_relative
        manifest["downloads"] = {
            checkpoint_relative: _github_download_record(
                release_id,
                repository,
                checkpoint_path.name,
                checkpoint_path.stat().st_size,
            )
        }
        write_yaml(staging / "manifest.yaml", manifest)
        checksum_text = "".join(
            f"{digest}  {name}\n" for name, digest in checksum_entries
        )
        _atomic_write(staging / "checksums.sha256", checksum_text)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    candidate["status"] = "released"
    candidate["release_id"] = release_id
    candidate["released_at"] = utc_now()
    save_candidate(root_path, candidate)
    return manifest, destination


def load_release(root: os.PathLike | str, release_id: str) -> Dict[str, Any]:
    path = release_directory(root, release_id) / "manifest.yaml"
    if not path.is_file():
        raise ZooError(f"Release not found: {release_id}")
    return load_yaml(path)


def list_releases(root: Optional[os.PathLike | str] = None) -> list[Dict[str, Any]]:
    root_path = get_zoo_root(root)
    if not root_path.exists():
        return []
    releases = []
    for path in root_path.glob("releases/*/*/*/v*/manifest.yaml"):
        try:
            releases.append(load_yaml(path))
        except ZooError:
            continue
    return sorted(releases, key=lambda item: item["release_id"])


def list_candidates(root: Optional[os.PathLike | str] = None) -> list[Dict[str, Any]]:
    directory = get_zoo_root(root) / ".candidates"
    if not directory.exists():
        return []
    candidates = []
    for path in directory.glob("*/candidate.yaml"):
        try:
            candidates.append(load_yaml(path))
        except ZooError:
            continue
    return sorted(candidates, key=lambda item: item["candidate_id"])


def configure_release_download(
    root: os.PathLike | str,
    release_id: str,
    *,
    repository: str = DEFAULT_RELEASE_REPOSITORY,
) -> Dict[str, Any]:
    """Add deterministic GitHub Release metadata to an existing release."""
    manifest = load_release(root, release_id)
    checkpoint, relative, expected = _checkpoint_details(root, release_id, manifest)
    if not checkpoint.is_file():
        raise ZooError(f"Released checkpoint is not present locally: {checkpoint}")
    actual = sha256_file(checkpoint)
    if actual != expected:
        raise ZooError(
            "Released checkpoint checksum mismatch: "
            f"expected {expected}, got {actual}"
        )
    record = _github_download_record(
        release_id,
        repository,
        checkpoint.name,
        checkpoint.stat().st_size,
    )
    manifest["downloads"] = {relative: record}
    write_yaml(release_directory(root, release_id) / "manifest.yaml", manifest)
    return record


def _download_record(manifest: Dict[str, Any], relative: str) -> Dict[str, Any]:
    record = manifest.get("downloads", {}).get(relative)
    if not isinstance(record, dict):
        raise ZooError(
            "Release has no checkpoint download metadata; publish or configure "
            f"the release first: {manifest.get('release_id')}"
        )
    urls = record.get("urls")
    size_bytes = record.get("size_bytes")
    if (
        not isinstance(urls, list)
        or not urls
        or not all(isinstance(url, str) for url in urls)
        or not isinstance(size_bytes, int)
        or size_bytes < 0
    ):
        raise ZooError("Checkpoint download metadata is malformed")
    for url in urls:
        if urllib.parse.urlparse(url).scheme != "https":
            raise ZooError(f"Checkpoint download URL must use HTTPS: {url}")
    return record


def fetch_release(
    root: os.PathLike | str,
    release_id: str,
    *,
    cache_root: Optional[os.PathLike | str] = None,
    force: bool = False,
    opener=None,
) -> Dict[str, Any]:
    """Fetch and verify a release checkpoint into the ignored local cache."""
    manifest = load_release(root, release_id)
    local_checkpoint, relative, expected = _checkpoint_details(
        root, release_id, manifest
    )
    cached = checkpoint_cache_path(root, release_id, relative, cache_root)

    if not force and _verified_file(local_checkpoint, expected):
        return {
            "release_id": release_id,
            "checkpoint": str(local_checkpoint),
            "checkpoint_sha256": expected,
            "downloaded": False,
            "source": "release-directory",
        }
    if not force and _verified_file(cached, expected):
        return {
            "release_id": release_id,
            "checkpoint": str(cached),
            "checkpoint_sha256": expected,
            "downloaded": False,
            "source": "cache",
        }

    record = _download_record(manifest, relative)
    cached.parent.mkdir(parents=True, exist_ok=True)
    open_url = opener or urllib.request.urlopen
    failures = []
    for url in record["urls"]:
        fd, temporary_name = tempfile.mkstemp(
            dir=cached.parent, prefix=f".{cached.name}.", suffix=".part"
        )
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            size = 0
            with open_url(url, timeout=60) as response, temporary.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    size += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            if size != record["size_bytes"]:
                raise ZooError(
                    f"download size mismatch: expected {record['size_bytes']}, got {size}"
                )
            actual = sha256_file(temporary)
            if actual != expected:
                raise ZooError(
                    f"download checksum mismatch: expected {expected}, got {actual}"
                )
            os.replace(temporary, cached)
            return {
                "release_id": release_id,
                "checkpoint": str(cached),
                "checkpoint_sha256": expected,
                "downloaded": True,
                "source": url,
            }
        except (OSError, urllib.error.URLError, ZooError) as exc:
            failures.append(f"{url}: {exc}")
        finally:
            if temporary.exists():
                temporary.unlink()
    raise ZooError("Checkpoint download failed:\n  " + "\n  ".join(failures))


def publish_release(
    root: os.PathLike | str,
    release_id: str,
    *,
    target_commit: str,
    execute: bool = True,
) -> Dict[str, Any]:
    """Create a draft GitHub Release for one locally verified checkpoint."""
    if not re.fullmatch(r"[0-9a-fA-F]{40}", target_commit):
        raise ZooError("Publication target must be a full 40-character Git commit")
    manifest = load_release(root, release_id)
    checkpoint, relative, expected = _checkpoint_details(root, release_id, manifest)
    if not checkpoint.is_file() or sha256_file(checkpoint) != expected:
        raise ZooError("The local release checkpoint is missing or fails verification")
    record = _download_record(manifest, relative)
    required_github_fields = ("repository", "tag", "asset")
    if record.get("provider") != "github-release" or not all(
        isinstance(record.get(field), str) and record[field]
        for field in required_github_fields
    ):
        raise ZooError("GitHub Release publication metadata is malformed")
    if checkpoint.stat().st_size != record["size_bytes"]:
        raise ZooError("The local checkpoint size does not match download metadata")
    expected_url = _github_download_record(
        release_id,
        record["repository"],
        record["asset"],
        record["size_bytes"],
    )
    if record != expected_url:
        raise ZooError("GitHub Release download metadata is not canonical")

    zoo_root = get_zoo_root(root)
    repository_root = zoo_root.parent
    if (repository_root / ".git").exists():
        manifest_path = release_directory(root, release_id) / "manifest.yaml"
        manifest_relative = manifest_path.relative_to(repository_root).as_posix()
        try:
            committed_manifest = subprocess.check_output(
                ["git", "show", f"{target_commit}:{manifest_relative}"],
                cwd=repository_root,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ZooError(
                "Publication target does not contain the release manifest; "
                "commit and push the metadata first"
            ) from exc
        if committed_manifest != manifest_path.read_bytes():
            raise ZooError(
                "Publication target contains different release metadata; "
                "commit the current manifest first"
            )

    title = (
        f"{manifest['model']} {manifest['quantization']['profile']} "
        f"{manifest['version']}"
    )
    notes = (
        f"Validated FixQuant model-zoo release `{release_id}`.\n\n"
        f"Checkpoint SHA-256: `{expected}`"
    )
    command = [
        "gh",
        "release",
        "create",
        record["tag"],
        str(checkpoint),
        "--repo",
        record["repository"],
        "--target",
        target_commit.lower(),
        "--title",
        title,
        "--notes",
        notes,
        "--draft",
    ]
    if execute:
        try:
            subprocess.run(command, check=True)
        except OSError as exc:
            raise ZooError(f"Could not run GitHub CLI: {exc}") from exc
        except subprocess.CalledProcessError as exc:
            raise ZooError(
                f"GitHub CLI failed while creating release {record['tag']}"
            ) from exc
    return {
        "release_id": release_id,
        "tag": record["tag"],
        "repository": record["repository"],
        "target_commit": target_commit.lower(),
        "draft": True,
        "executed": execute,
        "command": shlex.join(command),
    }


def build_catalog(root: Optional[os.PathLike | str] = None) -> Dict[str, Any]:
    releases = []
    for manifest in list_releases(root):
        checkpoint_relative = manifest.get("artifacts", {}).get("best_checkpoint")
        download = manifest.get("downloads", {}).get(checkpoint_relative, {})
        releases.append({
            "release_id": manifest["release_id"],
            "status": manifest["status"],
            "model": manifest["model"],
            "dataset": manifest["dataset"]["name"],
            "profile": manifest["quantization"]["profile"],
            "version": manifest["version"],
            "qat_top1": manifest["metrics"]["qat"]["top1"],
            "tilecnn_top1": manifest["metrics"]["tilecnn"]["top1"],
            "git_commit": manifest["provenance"].get("git_commit"),
            "downloadable": bool(download.get("urls")),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "zoo_root": ".",
        "releases": releases,
    }


def verify_release(root: os.PathLike | str, release_id: str) -> Dict[str, Any]:
    directory = release_directory(root, release_id)
    manifest = load_release(root, release_id)
    failures = []
    for relative, expected in manifest.get("checksums", {}).items():
        path = directory / relative
        if not path.is_file():
            failures.append({"path": relative, "error": "missing"})
            continue
        actual = sha256_file(path)
        if actual != expected:
            failures.append(
                {"path": relative, "error": "checksum", "expected": expected, "actual": actual}
            )
    return {
        "release_id": release_id,
        "verified": not failures,
        "checked_files": len(manifest.get("checksums", {})),
        "failures": failures,
    }


def resolve_release(
    root: os.PathLike | str,
    release_id: str,
    *,
    cache_root: Optional[os.PathLike | str] = None,
) -> Dict[str, Any]:
    directory = release_directory(root, release_id)
    manifest = load_release(root, release_id)
    checkpoint, checkpoint_relative, expected = _checkpoint_details(
        root, release_id, manifest
    )
    cached = checkpoint_cache_path(
        root, release_id, checkpoint_relative, cache_root
    )
    if not _verified_file(checkpoint, expected) and _verified_file(cached, expected):
        checkpoint = cached
    return {
        "release_id": release_id,
        "release_dir": str(directory),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": expected,
        "checkpoint_available": _verified_file(checkpoint, expected),
        "model": manifest["model"],
        "dataset": manifest["dataset"],
        "profile": manifest["quantization"]["profile"],
        "cle": bool(manifest["quantization"].get("cle", False)),
        "metrics": manifest["metrics"],
    }
