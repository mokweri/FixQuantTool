"""TileCNN exporter tests: full-graph export of a depthwise/ReLU6 model,
relu6/groups in graph.json, legality checks, and CLE numerics."""

import json

import pytest
import torch
import yaml
from pathlib import Path

from fixquant.graph.qat_processor import QatProcessor
from fixquant.graph.inference_processor import InferProcessor
from fixquant.emulation.model_introspector import StdModelInspector
from fixquant.export.tilecnn_exporter import TileCNNGraphExporter, _check_shift_legality
from tools.export_tilecnn_graph import (
    PREPROCESSING,
    build_parser,
    resolve_export_model,
    sha256_file,
    write_package_manifest,
)

from tests.models import TinyMobileBlockNet, synthetic_loader

CONFIG = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "configs/quant_config.yaml"))


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    torch.manual_seed(0)
    proc = QatProcessor(TinyMobileBlockNet(), CONFIG)
    qat = proc.quantize()
    proc.calibrate(synthetic_loader(), "cpu")
    proc.freeze()

    infer = InferProcessor(qat, CONFIG)
    hw = infer.convert_to_hardware_model()

    inspector = StdModelInspector(hw, default_input_frac=infer.input_frac or 5)
    nodes = inspector.topological_order()
    inspector.register_activation_hooks(nodes, capture_input=True, capture_output=True)
    with torch.no_grad():
        inspector.run_and_capture(torch.randn(1, 3, 32, 32))

    out_dir = tmp_path_factory.mktemp("export")
    exporter = TileCNNGraphExporter(inspector, model_name="tiny_mobile_block",
                                    default_input_frac=infer.input_frac or 5)
    exporter.export(str(out_dir))
    return out_dir, json.load(open(out_dir / "graph.json"))


def test_export_writes_graph_and_artifacts(exported):
    out_dir, graph = exported
    assert graph["schema"] == "tilecnn.graph.v1"
    assert (out_dir / "graph.json").exists()
    for ref_id in graph["graph"]["references"].values():
        f = out_dir / graph["tensors"][ref_id]["file"]
        assert f.exists() and f.stat().st_size > 0


def test_export_depthwise_conv_node(exported):
    _, graph = exported
    dw_nodes = [n for n in graph["nodes"] if n["op"] == "conv2d" and n["attrs"]["groups"] > 1]
    assert len(dw_nodes) == 1
    dw = dw_nodes[0]
    w_shape = graph["tensors"][dw["inputs"]["weight"]]["shape"]
    assert w_shape == [16, 1, 3, 3]


def test_export_relu6_post_op_preserved(exported):
    _, graph = exported
    relu6_nodes = [n for n in graph["nodes"] if n.get("post_ops", {}).get("relu6")]
    relu_nodes = [n for n in graph["nodes"] if n.get("post_ops", {}).get("relu")]
    assert len(relu6_nodes) == 2, "expand + depthwise convs must carry relu6, not relu"
    assert len(relu_nodes) >= 1     # stem conv carries plain relu


def test_export_all_params_int8_with_frac(exported):
    _, graph = exported
    for tid, spec in graph["tensors"].items():
        assert spec["dtype"] == "int8", tid
        assert isinstance(spec["frac"], int), tid


def test_legality_check_rejects_bad_shift():
    graph = {
        "tensors": {
            "in": {"frac": 40}, "w": {"frac": 7}, "b": {"frac": 7}, "out": {"frac": -10},
        },
        "nodes": [{
            "id": "conv", "op": "conv2d",
            "inputs": {"ifm": "in", "weight": "w", "bias": "b"},
            "outputs": {"ofm": "out"}, "post_ops": {},
        }],
    }
    with pytest.raises(ValueError, match="shift_out"):
        _check_shift_legality(graph)


def test_cle_preserves_function_and_equalizes():
    """CLE itself is mathematically exact through ReLU: compare the equalized
    model against a BN-folded + ReLU6→ReLU (but NOT equalized) reference so
    only the equalization step is under test."""
    import copy
    from torch.fx.experimental.optimization import fuse
    from fixquant.quantization.equalization import (
        cross_layer_equalize, replace_relu6_with_relu)

    torch.manual_seed(0)
    model = TinyMobileBlockNet().eval()
    # make the dw ranges pathologically unbalanced, like folded MobileNet
    with torch.no_grad():
        model.dw.weight[::2] *= 20.0

    ref_model = replace_relu6_with_relu(fuse(copy.deepcopy(model)))
    eq_model = replace_relu6_with_relu(fuse(copy.deepcopy(model)))

    def dw_ratio(gm):
        dw = next(m for m in gm.modules()
                  if isinstance(m, torch.nn.Conv2d) and m.groups > 1)
        r = dw.weight.detach().abs().amax(dim=(1, 2, 3))
        return (r.max() / r.min().clamp_min(1e-12)).item()

    ratio_before = dw_ratio(eq_model)
    cross_layer_equalize(eq_model, iterations=2)
    ratio_after = dw_ratio(eq_model)

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        ref = ref_model(x)
        got = eq_model(x)
    assert torch.allclose(ref, got, atol=1e-4), (ref - got).abs().max()
    assert ratio_after < ratio_before / 2, (ratio_before, ratio_after)


def test_export_resolves_model_zoo_release(tmp_path):
    release_id = "mobilenet_v2/imagenet1k/int8-tqt-cle@v1.0.0"
    release_dir = (
        tmp_path / "releases" / "mobilenet_v2" / "imagenet1k" /
        "int8-tqt-cle" / "v1.0.0"
    )
    checkpoint = release_dir / "qat" / "model_best.pth.tar"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_hash = sha256_file(checkpoint)
    manifest = {
        "release_id": release_id,
        "model": "mobilenet_v2",
        "dataset": {"name": "imagenet1k", "path": "/dataset/imagenet"},
        "quantization": {"profile": "int8-tqt-cle", "cle": True},
        "metrics": {},
        "artifacts": {"best_checkpoint": "qat/model_best.pth.tar"},
        "checksums": {"qat/model_best.pth.tar": checkpoint_hash},
    }
    (release_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest))

    args = build_parser().parse_args([
        "--zoo-model", release_id,
        "--zoo-root", str(tmp_path),
    ])
    resolved = resolve_export_model(args, Path("/unused/repo"))

    assert resolved == checkpoint
    assert args.model == "mobilenet_v2"
    assert args.cle is True
    assert args.zoo_release["checkpoint_sha256"] == checkpoint_hash


def test_package_manifest_records_release_and_artifact_provenance(tmp_path):
    out_dir = tmp_path / "package"
    (out_dir / "inputs").mkdir(parents=True)
    (out_dir / "params").mkdir()
    (out_dir / "refs").mkdir()
    (out_dir / "inputs" / "input.int8.bin").write_bytes(b"input")
    (out_dir / "params" / "weight.int8.bin").write_bytes(b"weight")
    (out_dir / "refs" / "output.int8.bin").write_bytes(b"output")
    graph = {
        "schema": "tilecnn.graph.v1",
        "tensors": {
            "input": {"kind": "input", "file": "inputs/input.int8.bin"},
            "weight": {
                "kind": "param", "file": "params/weight.int8.bin"
            },
            "output_ref": {
                "kind": "reference", "file": "refs/output.int8.bin"
            },
        },
    }
    (out_dir / "graph.json").write_text(json.dumps(graph))
    checkpoint = tmp_path / "model_best.pth.tar"
    quant_config = tmp_path / "quant_config.yaml"
    image = tmp_path / "image.JPEG"
    checkpoint.write_bytes(b"checkpoint")
    quant_config.write_text("quant: config\n")
    image.write_bytes(b"image")
    release = {
        "release_id": "resnet50/imagenet1k/int8-tqt@v1.0.0",
        "dataset": {"name": "imagenet1k"},
        "profile": "int8-tqt",
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    args = build_parser().parse_args([
        "--zoo-model", release["release_id"],
    ])

    manifest = write_package_manifest(
        out_dir, Path(__file__).resolve().parents[1], args, checkpoint,
        quant_config, image, release,
    )

    assert manifest["schema"] == "tilecnn.model-package.v1"
    assert manifest["model"]["release_id"] == release["release_id"]
    assert manifest["producer"]["git_revision"]
    assert manifest["sources"]["checkpoint"]["sha256"] == release[
        "checkpoint_sha256"
    ]
    assert manifest["preprocessing"] == PREPROCESSING
    assert manifest["integrity"] == {
        "algorithm": "sha256",
        "coverage": "all-graph-artifacts",
    }
    assert manifest["artifacts"]["graph"]["sha256"] == sha256_file(
        out_dir / "graph.json"
    )
    assert manifest["artifacts"]["references"][0]["sha256"] == sha256_file(
        out_dir / "refs" / "output.int8.bin"
    )
    assert manifest["artifacts"]["parameters"][0]["sha256"] == sha256_file(
        out_dir / "params" / "weight.int8.bin"
    )
    assert (out_dir / "manifest.json").is_file()
