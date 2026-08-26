import json
from io import BytesIO
from pathlib import Path

import pytest
import yaml

from fixquant.model_zoo import (
    ZooError,
    fetch_release,
    load_candidate,
    promote_candidate,
    publish_release,
    register_candidate,
    resolve_release,
    select_dataset_path,
    sha256_file,
    validate_candidate,
    verify_release,
)


def test_dataset_path_selection_is_machine_portable():
    environment = {"FIXQUANT_DATA_DIR": "/local/imagenet"}

    assert select_dataset_path(
        "/explicit/imagenet",
        "/arrhenius/imagenet",
        environ=environment,
        fallback="/fallback/imagenet",
    ) == "/explicit/imagenet"
    assert select_dataset_path(
        release_path="/arrhenius/imagenet",
        environ=environment,
        fallback="/fallback/imagenet",
    ) == "/local/imagenet"
    assert select_dataset_path(
        release_path="/arrhenius/imagenet",
        environ={},
        fallback="/fallback/imagenet",
    ) == "/arrhenius/imagenet"
    assert select_dataset_path(
        environ={},
        fallback="/fallback/imagenet",
    ) == "/fallback/imagenet"


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
    checkpoint_relative = manifest["artifacts"]["best_checkpoint"]
    download = manifest["downloads"][checkpoint_relative]
    assert download["provider"] == "github-release"
    assert download["tag"] == (
        "model-zoo-resnet18-imagenet1k-int8-tqt-v1.0.0"
    )
    assert download["urls"] == [
        "https://github.com/mokweri/FixQuantTool/releases/download/"
        "model-zoo-resnet18-imagenet1k-int8-tqt-v1.0.0/"
        "model_best.pth.tar"
    ]
    assert download["size_bytes"] == len("best checkpoint")
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


def test_release_checkpoint_fetches_to_verified_cache(tmp_path):
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
    manifest, release_dir = promote_candidate(
        zoo_root, candidate["candidate_id"], "1.0.0"
    )
    release_id = manifest["release_id"]
    checkpoint = release_dir / manifest["artifacts"]["best_checkpoint"]
    checkpoint.unlink()

    payload = b"best checkpoint"

    def opener(_url, timeout):
        assert timeout == 60
        return BytesIO(payload)

    cache_root = tmp_path / "cache"
    fetched = fetch_release(
        zoo_root,
        release_id,
        cache_root=cache_root,
        opener=opener,
    )
    assert fetched["downloaded"]
    assert Path(fetched["checkpoint"]).read_bytes() == payload

    resolved = resolve_release(
        zoo_root,
        release_id,
        cache_root=cache_root,
    )
    assert resolved["checkpoint"] == fetched["checkpoint"]
    assert resolved["checkpoint_available"]

    def unexpected_download(_url, timeout):
        raise AssertionError(f"unexpected download with timeout {timeout}")

    reused = fetch_release(
        zoo_root,
        release_id,
        cache_root=cache_root,
        opener=unexpected_download,
    )
    assert not reused["downloaded"]
    assert reused["source"] == "cache"


def test_release_fetch_rejects_wrong_checkpoint(tmp_path):
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
    manifest, release_dir = promote_candidate(
        zoo_root, candidate["candidate_id"], "1.0.0"
    )
    release_id = manifest["release_id"]
    (release_dir / manifest["artifacts"]["best_checkpoint"]).unlink()

    with pytest.raises(ZooError, match="checksum mismatch"):
        fetch_release(
            zoo_root,
            release_id,
            cache_root=tmp_path / "cache",
            opener=lambda _url, timeout: BytesIO(b"bad checkpoint!"),
        )


def test_publish_dry_run_is_bound_to_release_and_commit(tmp_path):
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
    manifest, _ = promote_candidate(
        zoo_root, candidate["candidate_id"], "1.0.0"
    )

    report = publish_release(
        zoo_root,
        manifest["release_id"],
        target_commit="a" * 40,
        execute=False,
    )
    assert not report["executed"]
    assert report["draft"]
    assert report["target_commit"] == "a" * 40
    assert "gh release create" in report["command"]
    assert "--draft" in report["command"]
