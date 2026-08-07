import json
from pathlib import Path

import pytest
import yaml

from fixquant.model_zoo import (
    ZooError,
    load_candidate,
    promote_candidate,
    register_candidate,
    resolve_release,
    sha256_file,
    validate_candidate,
    verify_release,
)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def _training_run(tmp_path):
    run_dir = tmp_path / "results" / "qat" / "resnet18" / "1234" / "resnet18"
    checkpoint = run_dir / "checkpoint" / "model_best.pth.tar"
    _write(checkpoint, "best checkpoint")
    _write(run_dir / "checkpoint" / "latest.pth.tar", "latest checkpoint")
    _write(run_dir / "resnet18_calib_report.csv", "quantizer,frac\nq,7\n")
    _write(run_dir / "logs" / "quant_thresholds.csv", "epoch,frac\n0,7\n")
    manifest = {
        "model": {"name": "resnet18"},
        "dataset": {"name": "imagenet1k", "path": "/dataset/imagenet"},
        "quantization": {"cle": False},
        "provenance": {"git_commit": "abc123", "slurm_job_id": "1234"},
    }
    _write(run_dir / "run_manifest.yaml", yaml.safe_dump(manifest))
    return run_dir, checkpoint


def _metrics(path, checkpoint_hash, representation, top1, top5):
    value = {
        "schema_version": 1,
        "representation": representation,
        "checkpoint_sha256": checkpoint_hash,
        "validation_samples": 50000,
        "metrics": {"loss": 1.0, "top1": top1, "top5": top5},
    }
    _write(path, json.dumps(value))


def test_candidate_validation_promotion_and_resolution(tmp_path):
    zoo_root = tmp_path / "zoo"
    run_dir, checkpoint = _training_run(tmp_path)
    candidate, candidate_path = register_candidate(run_dir, root=zoo_root)

    assert candidate["candidate_id"] == "resnet18-run-1234"
    assert candidate["status"] == "candidate"
    assert candidate["checkpoint_sha256"] == sha256_file(checkpoint)
    assert candidate_path.is_file()

    evaluation = candidate_path.parent / "evaluation"
    _metrics(
        evaluation / "qat_metrics.json",
        candidate["checkpoint_sha256"],
        "qat",
        70.0,
        89.0,
    )
    _metrics(
        evaluation / "tilecnn_metrics.json",
        candidate["checkpoint_sha256"],
        "tilecnn",
        69.4,
        88.7,
    )
    _write(evaluation / "qconfig.json", "{}")
    report = validate_candidate(zoo_root, candidate["candidate_id"])
    assert report["passed"]
    assert load_candidate(zoo_root, candidate["candidate_id"])["status"] == "validated"

    manifest, release_dir = promote_candidate(
        zoo_root, candidate["candidate_id"], "1.0.0"
    )
    release_id = "resnet18/imagenet1k/int8-tqt@v1.0.0"
    assert manifest["release_id"] == release_id
    assert release_dir == (
        zoo_root
        / "releases"
        / "resnet18"
        / "imagenet1k"
        / "int8-tqt"
        / "v1.0.0"
    )
    assert (release_dir / "qat" / "model_best.pth.tar").read_text() == "best checkpoint"
    assert (release_dir / "manifest.yaml").is_file()
    assert (release_dir / "model_card.md").is_file()
    assert (release_dir / "deployment" / "qconfig.json").is_file()
    assert verify_release(zoo_root, release_id)["verified"]

    resolved = resolve_release(zoo_root, release_id)
    assert resolved["model"] == "resnet18"
    assert resolved["cle"] is False
    assert resolved["checkpoint"].endswith("qat/model_best.pth.tar")

    with pytest.raises(ZooError, match="already exists"):
        promote_candidate(zoo_root, candidate["candidate_id"], "v1.0.0")


def test_candidate_rejected_when_checkpoint_hash_does_not_match(tmp_path):
    zoo_root = tmp_path / "zoo"
    run_dir, _ = _training_run(tmp_path)
    candidate, candidate_path = register_candidate(run_dir, root=zoo_root)
    evaluation = candidate_path.parent / "evaluation"
    _metrics(evaluation / "qat_metrics.json", "wrong", "qat", 70.0, 89.0)
    _metrics(
        evaluation / "tilecnn_metrics.json",
        candidate["checkpoint_sha256"],
        "tilecnn",
        69.5,
        88.7,
    )
    _write(evaluation / "qconfig.json", "{}")

    report = validate_candidate(zoo_root, candidate["candidate_id"])
    assert not report["passed"]
    assert load_candidate(zoo_root, candidate["candidate_id"])["status"] == "rejected"
    with pytest.raises(ZooError, match="must be validated"):
        promote_candidate(zoo_root, candidate["candidate_id"], "1.0.0")


def test_release_verification_detects_tampering(tmp_path):
    zoo_root = tmp_path / "zoo"
    run_dir, _ = _training_run(tmp_path)
    candidate, candidate_path = register_candidate(run_dir, root=zoo_root)
    evaluation = candidate_path.parent / "evaluation"
    _metrics(
        evaluation / "qat_metrics.json",
        candidate["checkpoint_sha256"],
        "qat",
        70.0,
        89.0,
    )
    _metrics(
        evaluation / "tilecnn_metrics.json",
        candidate["checkpoint_sha256"],
        "tilecnn",
        69.5,
        88.7,
    )
    _write(evaluation / "qconfig.json", "{}")
    validate_candidate(zoo_root, candidate["candidate_id"])
    _, release_dir = promote_candidate(zoo_root, candidate["candidate_id"], "1.0.0")
    _write(release_dir / "qat" / "model_best.pth.tar", "tampered")

    report = verify_release(
        zoo_root, "resnet18/imagenet1k/int8-tqt@v1.0.0"
    )
    assert not report["verified"]
    assert report["failures"][0]["error"] == "checksum"
