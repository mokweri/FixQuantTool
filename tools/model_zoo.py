#!/usr/bin/env python3
"""Manage FixQuant model-zoo candidates and immutable releases."""

import argparse
import json
import sys

from fixquant.model_zoo import (
    ZooError,
    build_catalog,
    get_zoo_root,
    list_candidates,
    list_releases,
    load_candidate,
    load_release,
    promote_candidate,
    register_candidate,
    resolve_release,
    validate_candidate,
    verify_release,
    write_yaml,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="Model-zoo root (or FIXQUANT_ZOO_ROOT)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    register = subparsers.add_parser("register", help="Register a training run")
    register.add_argument("run_dir")
    register.add_argument("--candidate-id")
    register.add_argument("--model")
    register.add_argument("--dataset-name", default="imagenet1k")
    register.add_argument("--dataset-path")
    register.add_argument("--profile")
    register.add_argument("--cle", action="store_true", default=None)

    list_parser = subparsers.add_parser("list", help="List immutable releases")
    list_parser.add_argument("--json", action="store_true")
    list_parser.add_argument("--candidates", action="store_true")

    show = subparsers.add_parser("show", help="Show a candidate or release")
    show.add_argument("identifier")
    show.add_argument("--candidate", action="store_true")

    validate = subparsers.add_parser("validate", help="Apply candidate quality gates")
    validate.add_argument("candidate_id")
    validate.add_argument("--qat-metrics")
    validate.add_argument("--deploy-metrics")
    validate.add_argument("--policy")

    promote = subparsers.add_parser("promote", help="Create an immutable release")
    promote.add_argument("candidate_id")
    promote.add_argument("--version", required=True)

    verify = subparsers.add_parser("verify", help="Verify all release checksums")
    verify.add_argument("release_id")

    resolve = subparsers.add_parser("resolve", help="Resolve a release for consumers")
    resolve.add_argument("release_id")

    catalog = subparsers.add_parser("catalog", help="Generate a lightweight catalog")
    catalog.add_argument("--output", help="Write YAML suitable for Git tracking")
    return parser


def main():
    args = build_parser().parse_args()
    root = get_zoo_root(args.root)
    if args.command == "register":
        candidate, path = register_candidate(
            args.run_dir,
            root=root,
            candidate_id=args.candidate_id,
            model=args.model,
            dataset_name=args.dataset_name,
            dataset_path=args.dataset_path,
            profile=args.profile,
            cle=args.cle,
        )
        print(json.dumps({"candidate": candidate, "manifest": str(path)}, indent=2))
    elif args.command == "list":
        releases = list_candidates(root) if args.candidates else list_releases(root)
        if args.json:
            print(json.dumps(releases, indent=2))
        elif args.candidates:
            for candidate in releases:
                print(
                    f"{candidate['candidate_id']}  "
                    f"status={candidate['status']}  model={candidate['model']}"
                )
        else:
            for release in releases:
                metrics = release["metrics"]
                print(
                    f"{release['release_id']}  "
                    f"QAT={metrics['qat']['top1']:.4f}  "
                    f"TileCNN={metrics['tilecnn']['top1']:.4f}"
                )
    elif args.command == "show":
        value = (
            load_candidate(root, args.identifier)
            if args.candidate
            else load_release(root, args.identifier)
        )
        print(json.dumps(value, indent=2))
    elif args.command == "validate":
        report = validate_candidate(
            root,
            args.candidate_id,
            qat_metrics_path=args.qat_metrics,
            deploy_metrics_path=args.deploy_metrics,
            policy_path=args.policy,
        )
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            return 1
    elif args.command == "promote":
        manifest, path = promote_candidate(root, args.candidate_id, args.version)
        print(json.dumps({"release": manifest, "path": str(path)}, indent=2))
    elif args.command == "verify":
        report = verify_release(root, args.release_id)
        print(json.dumps(report, indent=2))
        if not report["verified"]:
            return 1
    elif args.command == "resolve":
        print(json.dumps(resolve_release(root, args.release_id), indent=2))
    elif args.command == "catalog":
        catalog = build_catalog(root)
        if args.output:
            write_yaml(args.output, catalog)
            print(f"Catalog written to {args.output}")
        else:
            print(json.dumps(catalog, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ZooError as exc:
        print(f"model-zoo error: {exc}", file=sys.stderr)
        raise SystemExit(2)
