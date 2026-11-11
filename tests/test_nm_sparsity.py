"""N:M masks: pattern arithmetic, channel classes, the physical reorder, and
that the masks survive quantization and training steps."""
import copy
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torchvision
import yaml

from fixquant.graph.qat_processor import QatProcessor
from fixquant.sparsity import nm

CONFIG = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "configs/quant_config.yaml"))
SIZE = 64


def resnet18():
    m = torchvision.models.resnet18(weights=None, num_classes=40)
    for mod in m.modules():                       # non-trivial BN statistics
        if isinstance(mod, nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.2, 0.2)
            mod.running_var.uniform_(0.5, 2.0)
            mod.weight.data.uniform_(0.5, 1.5)
            mod.bias.data.uniform_(-0.1, 0.1)
    return m.eval()


@pytest.mark.parametrize("base", ["filter24", "pair24", "oc_n16"])
@pytest.mark.parametrize("shape", [(32, 16, 3, 3), (24, 8, 1, 1), (1000, 64, 1, 1)])
def test_masks_keep_half_and_are_the_pattern(base, shape):
    absw = torch.rand(shape)
    m = nm.pattern_mask(absw, base)
    assert nm.check_pattern(m, base) is None
    if base != "oc_n16" or shape[0] % 16 == 0:
        assert m.float().mean().item() == pytest.approx(0.5)


def test_check_pattern_catches_violations():
    m = nm.pattern_mask(torch.rand(8, 8, 1, 1), "pair24")
    m[0, :, 0, 0] = True
    assert nm.check_pattern(m, "pair24") is not None
    m = nm.pattern_mask(torch.rand(8, 8, 1, 1), "pair24")
    m[1] = ~m[1]
    assert "differ" in nm.check_pattern(m, "pair24")
    m = nm.pattern_mask(torch.rand(32, 4, 1, 1), "oc_n16")
    m[:, 0] = True
    assert nm.check_pattern(m, "oc_n16") is not None


def test_parse_pattern():
    assert nm.parse_pattern("pair24+perm") == ("pair24", True)
    for bad in ("oc_n16+perm", "pair24+x", "group24"):
        with pytest.raises(ValueError):
            nm.parse_pattern(bad)


def test_residual_streams_share_a_class():
    classes, fixed, width = nm.channel_classes(resnet18(), SIZE)
    stream = classes["layer1.0.conv1"][0]
    for reader in ("layer1.1.conv1", "layer2.0.conv1", "layer2.0.downsample.0"):
        assert classes[reader][0] == stream
    assert classes["layer1.0.conv2"][1] == stream          # writers of the stream
    assert classes["layer1.1.conv2"][1] == stream
    assert classes["fc"][1] in fixed and width[stream] == 64


def test_permutation_is_a_real_reorder():
    """The +perm masks, after reordering the network's channels, are the
    plain pattern in storage order and the network computes the same."""
    model = resnet18()
    art = nm.build_nm(model, "pair24+perm", perm_iters=20, image_size=SIZE)
    assert art["perms"], "the search should find a better order somewhere"
    nm.apply_float(model, art)
    x = torch.randn(2, 3, SIZE, SIZE)
    ref = model(x)
    phys = copy.deepcopy(model)
    nm.permute_channels(phys, art["perms"], SIZE)
    assert torch.allclose(phys(x), ref, atol=1e-4, rtol=1e-4)
    mods = dict(phys.named_modules())
    for layer in art["layers"]:
        assert nm.check_pattern(nm.as4d(mods[layer].weight != 0), "pair24") is None, layer


def test_perm_retains_more_than_none():
    model = resnet18()
    plain = nm.build_nm(model, "pair24")
    perm = nm.build_nm(model, "pair24+perm", perm_iters=20, image_size=SIZE)
    ret = lambda a: sum(r["retained_l1"] for r in a["report"])
    assert ret(perm) > ret(plain)


def test_selected_layers_only():
    model = resnet18()
    art = nm.build_nm(model, "oc_n16", layers=["layer4.1.conv2", "fc"], image_size=SIZE)
    assert set(art["masks"]) == {"layer4.1.conv2", "fc"}
    with pytest.raises(KeyError):
        nm.build_nm(model, "oc_n16", layers=["conv1"], image_size=SIZE)   # the stem


def test_masks_survive_quantization_and_training_steps():
    model = resnet18()
    art = nm.build_nm(model, "pair24+perm", perm_iters=10, image_size=SIZE)
    nm.apply_float(model, art)
    qat = QatProcessor(model, CONFIG).quantize()
    masker = nm.Masker(qat, art)
    qat.train()
    opt = torch.optim.SGD(qat.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    masker.attach(opt)
    for _ in range(3):
        opt.zero_grad()
        qat(torch.randn(4, 3, SIZE, SIZE)).logsumexp(1).mean().backward()
        opt.step()
    check = nm.verify_state(qat.state_dict(), art)
    assert check["ok"], check["violations"]
    assert check["nonzero_fraction"] <= 0.5 + 1e-6


def test_masker_rejects_an_unpruned_model():
    model = resnet18()
    art = nm.build_nm(model, "filter24", image_size=SIZE)
    qat = QatProcessor(model, CONFIG).quantize()           # float model not pruned
    with pytest.raises(ValueError):
        nm.Masker(qat, art)


def test_verify_state_reports_nonzero_masked_weights(tmp_path):
    model = resnet18()
    art = nm.build_nm(model, "filter24", image_size=SIZE)
    nm.save(art, tmp_path / "m.pt")
    art = nm.load(tmp_path / "m.pt")
    nm.apply_float(model, art)
    state = model.state_dict()
    assert nm.verify_state(state, art)["ok"]
    state["fc.weight"] = state["fc.weight"] + 1.0
    res = nm.verify_state(state, art)
    assert not res["ok"] and any(v.startswith("fc:") for v in res["violations"])


def test_float_finetune_hands_its_masks_to_qat(tmp_path):
    """The two-stage flow: masks held through float SGD steps, the checkpoint
    reloaded, the same masks taken over (not chosen again) and held through
    quantization."""
    model = resnet18()
    art = nm.prepare(model, "pair24+perm", perm_iters=10, image_size=SIZE, save_dir=tmp_path)
    masker = nm.Masker(model, art)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True,
                          weight_decay=1e-4)
    masker.attach(opt)
    for _ in range(3):
        opt.zero_grad()
        model(torch.randn(4, 3, SIZE, SIZE)).logsumexp(1).mean().backward()
        opt.step()
    torch.save({"state_dict": model.state_dict()}, tmp_path / "best.pth.tar")
    assert nm.verify_checkpoint(tmp_path / "best.pth.tar", art)["ok"]

    second = resnet18()
    second.load_state_dict(torch.load(tmp_path / "best.pth.tar")["state_dict"])
    (tmp_path / "qat").mkdir()
    taken = nm.prepare(second, "pair24+perm", masks_path=tmp_path / "nm_masks.pt",
                       image_size=SIZE, save_dir=tmp_path / "qat")
    assert set(taken["masks"]) == set(art["masks"])
    assert all(torch.equal(taken["masks"][k], art["masks"][k]) for k in art["masks"])
    qat = QatProcessor(second, CONFIG).quantize()
    nm.Masker(qat, taken)                                  # masked weights still zero
    with pytest.raises(ValueError):
        nm.prepare(resnet18(), "filter24", masks_path=tmp_path / "nm_masks.pt")


def test_prepare_without_sparsity_is_a_no_op():
    model = resnet18()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    assert nm.prepare(model) is None
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
