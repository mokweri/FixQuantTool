#!/usr/bin/env python3
"""Evaluate a model-zoo candidate and apply the configured quality gates."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from fixquant.model_zoo import (
    ZooError,
    candidate_directory,
    get_zoo_root,
    load_candidate,
    validate_candidate,
)


def _run(command):
    print("Running:", " ".join(str(item) for item in command), flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_id")
    parser.add_argument("--root", default=None)
    parser.add_argument("--policy", default="configs/model_zoo_policy.yaml")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--qat-batch-size", type=int, default=128)
    parser.add_argument("--deploy-batch-size", type=int, default=64)
    args = parser.parse_args()

    root = get_zoo_root(args.root)
    candidate = load_candidate(root, args.candidate_id)
    checkpoint = candidate["artifacts"]["best_checkpoint"]
    dataroot = candidate["dataset"].get("path")
    if not dataroot:
        raise ZooError("Candidate dataset path is missing")
    dataset = "imagenet" if candidate["dataset"]["name"].startswith("imagenet") else candidate["dataset"]["name"]
    evaluation = candidate_directory(root, args.candidate_id) / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parent.parent

    common = [
        "--dataset", dataset,
        "--dataroot", dataroot,
        "--model", candidate["model"],
        "--checkpoint", checkpoint,
        "--n_worker", str(args.workers),
        "--gpus", "0",
    ]
    if candidate["quantization"].get("cle"):
        common.append("--cle")

    qat_metrics = evaluation / "qat_metrics.json"
    _run([
        sys.executable,
        str(repo_root / "tools" / "qat_test.py"),
        *common,
        "--test_batch_size", str(args.qat_batch_size),
        "--save_dir", str(evaluation / "qat-runtime"),
        "--metrics-output", str(qat_metrics),
    ])

    deploy_metrics = evaluation / "tilecnn_metrics.json"
    qconfig = evaluation / "qconfig.json"
    _run([
        sys.executable,
        str(repo_root / "tools" / "deploy_eval.py"),
        *common,
        "--model_type", "tilecnn",
        "--test_batch_size", str(args.deploy_batch_size),
        "--save_dir", str(evaluation / "tilecnn-runtime"),
        "--metrics-output", str(deploy_metrics),
        "--qconfig-output", str(qconfig),
    ])

    report = validate_candidate(
        root,
        args.candidate_id,
        qat_metrics_path=qat_metrics,
        deploy_metrics_path=deploy_metrics,
        policy_path=args.policy,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ZooError as exc:
        print(f"candidate validation error: {exc}", file=sys.stderr)
        raise SystemExit(2)
