#!/usr/bin/env python3
"""Check an exported DeepTile ModelPackage against the fabric's legality rules.

The DeepTile compiler lives in a separate repository. When it is not available,
this script is the acceptance gate: it reads ``graph.json`` and ``manifest.json``
from an exported package directory and verifies every constraint the accelerator
imposes on a graph it is asked to compile.

Pure standard library, so it runs on a login node.

Usage:
    python tools/check_deeptile_legality.py outputs/vgg16_tilecnn_int8_deeptile
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

# Weight-buffer budget: ceil(Cin / 16) * Kh * Kw entries per conv/linear.
CHANNEL_PACK = 16
WEIGHT_BUFFER_BUDGET = 512

# The accelerator implements exactly one pooling post-op.
LEGAL_MAXPOOL = {"kernel": [3, 3], "stride": [2, 2], "padding": [1, 1, 1, 1]}

# Square kernels the fabric runs; 5x5 is validated from kernel ABI version 2.
LEGAL_KERNELS = {1, 3, 5, 7}
LEGAL_STRIDES = {1, 2}


def weight_buffer_cost(cin: int, kh: int, kw: int) -> int:
    return math.ceil(cin / CHANNEL_PACK) * kh * kw


def check_graph(graph, problems):
    tensors = graph["tensors"]
    nodes = graph["nodes"]

    # Which node produces each tensor, so gap2d's producer can be identified.
    producer = {}
    for node in nodes:
        for tensor_id in node.get("outputs", {}).values():
            producer[tensor_id] = node

    for node in nodes:
        nid, op = node["id"], node["op"]
        attrs = node.get("attrs", {})

        if op in ("conv2d", "linear"):
            weight = tensors[node["inputs"]["weight"]]
            shape = weight["shape"]
            if op == "conv2d":
                # OIHW, where I is channels per group -- what one tile loads.
                _, cin, kh, kw = shape
                groups = int(attrs.get("groups", 1))
                out_channels = shape[0]
                stride = attrs.get("stride", [1, 1])
                padding = attrs.get("padding", [0, 0, 0, 0])
                dilation = attrs.get("dilation", [1, 1])

                if kh != kw:
                    problems.append(f"{nid}: kernel {kh}x{kw} is not square")
                elif kh not in LEGAL_KERNELS:
                    problems.append(
                        f"{nid}: kernel {kh} not in {sorted(LEGAL_KERNELS)}")
                if len(set(stride)) != 1:
                    problems.append(f"{nid}: strides {stride} are not equal")
                elif stride[0] not in LEGAL_STRIDES:
                    problems.append(
                        f"{nid}: stride {stride[0]} not in {sorted(LEGAL_STRIDES)}")
                if len(set(padding)) != 1:
                    problems.append(f"{nid}: padding {padding} is not symmetric")
                if any(d != 1 for d in dilation):
                    problems.append(f"{nid}: dilation {dilation} is not supported")
                if groups != 1 and groups != out_channels:
                    problems.append(
                        f"{nid}: groups={groups} is neither 1 nor depthwise "
                        f"(out_channels={out_channels})")
            else:
                # OI, lowered to a 1x1 convolution.
                cin, kh, kw = shape[1], 1, 1
                ifm_shape = list(tensors[node["inputs"]["ifm"]]["shape"])
                if len(ifm_shape) != 3 or ifm_shape[1:] != [1, 1]:
                    problems.append(
                        f"{nid}: linear input shape {ifm_shape} is not [C, 1, 1]")

            cost = weight_buffer_cost(cin, kh, kw)
            if cost > WEIGHT_BUFFER_BUDGET:
                problems.append(
                    f"{nid}: weight-buffer cost ceil({cin}/{CHANNEL_PACK})*{kh}*{kw} "
                    f"= {cost} exceeds the {WEIGHT_BUFFER_BUDGET} budget")

        elif op == "maxpool2d":
            for key, expected in LEGAL_MAXPOOL.items():
                actual = list(attrs.get(key, []))
                if actual != expected:
                    problems.append(
                        f"{nid}: maxpool {key} {actual} != {expected}")

        elif op == "gap2d":
            ifm_id = node["inputs"]["ifm"]
            source = producer.get(ifm_id)
            if source is None:
                problems.append(
                    f"{nid}: gap2d input '{ifm_id}' is a graph input, not a "
                    "convolution output (a standalone GAP is rejected)")
            elif source["op"] != "conv2d":
                problems.append(
                    f"{nid}: gap2d producer '{source['id']}' is a "
                    f"{source['op']}, must be a conv2d")
            ofm = tensors[node["outputs"]["ofm"]]
            if len(ofm["shape"]) != 3 or ofm["shape"][1:] != [1, 1]:
                problems.append(
                    f"{nid}: gap2d output shape {ofm['shape']} is not [C, 1, 1]")

        else:
            problems.append(f"{nid}: unsupported op '{op}'")


def check_manifest(package_dir, graph, manifest, problems):
    tensors = graph["tensors"]

    expected = {
        "parameters": sorted({s["file"] for s in tensors.values()
                              if s.get("kind") == "param" and s.get("file")}),
        "inputs": sorted({s["file"] for s in tensors.values()
                          if s.get("kind") == "input" and s.get("file")}),
        "references": sorted({s["file"] for s in tensors.values()
                              if s.get("kind") == "reference" and s.get("file")}),
    }
    artifacts = manifest.get("artifacts", {})

    for kind, files in expected.items():
        if not files:
            problems.append(f"manifest: graph declares no {kind}")
        listed = {record["path"]: record.get("sha256")
                  for record in artifacts.get(kind, [])}
        for path in files:
            if path not in listed:
                problems.append(f"manifest: {kind} '{path}' is not listed")
                continue
            recorded = listed[path]
            if not recorded:
                problems.append(f"manifest: {kind} '{path}' has no checksum")
                continue
            blob = package_dir / path
            if not blob.is_file():
                problems.append(f"manifest: {kind} '{path}' is missing on disk")
                continue
            actual = hashlib.sha256(blob.read_bytes()).hexdigest()
            if actual != recorded:
                problems.append(
                    f"manifest: {kind} '{path}' checksum mismatch "
                    f"(recorded {recorded[:12]}..., actual {actual[:12]}...)")

    graph_record = artifacts.get("graph", {})
    if graph_record.get("sha256"):
        actual = hashlib.sha256((package_dir / "graph.json").read_bytes()).hexdigest()
        if actual != graph_record["sha256"]:
            problems.append("manifest: graph.json checksum mismatch")
    else:
        problems.append("manifest: graph.json has no checksum")

    return expected


def summarize(graph):
    counts = {}
    for node in graph["nodes"]:
        counts[node["op"]] = counts.get(node["op"], 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("package_dir", help="Exported ModelPackage directory")
    parser.add_argument("--json", action="store_true",
                        help="Emit the result as JSON")
    args = parser.parse_args()

    package_dir = Path(args.package_dir)
    graph = json.loads((package_dir / "graph.json").read_text())
    manifest = json.loads((package_dir / "manifest.json").read_text())

    problems = []
    check_graph(graph, problems)
    artifacts = check_manifest(package_dir, graph, manifest, problems)

    result = {
        "package": str(package_dir),
        "model": graph["model"]["name"],
        "node_counts": summarize(graph),
        "artifact_counts": {k: len(v) for k, v in artifacts.items()},
        "problems": problems,
        "passed": not problems,
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Package: {result['package']}")
        print(f"Model:   {result['model']}")
        print("Nodes:   " + ", ".join(f"{op}={n}" for op, n in
                                      sorted(result["node_counts"].items())))
        print("Files:   " + ", ".join(f"{k}={n}" for k, n in
                                      sorted(result["artifact_counts"].items())))
        if problems:
            print(f"\nFAILED with {len(problems)} problem(s):")
            for problem in problems:
                print(f"  - {problem}")
        else:
            print("\nPASSED: graph and manifest satisfy every DeepTile legality rule.")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
