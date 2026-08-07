from types import SimpleNamespace

import torch

from fixquant.training.run_manager import RunManager


def _manager(checkpoint_dir):
    manager = RunManager.__new__(RunManager)
    manager.__dict__["_save_path"] = str(checkpoint_dir)
    manager.run_config = SimpleNamespace(dataset="imagenet")
    return manager


def _checkpoint(epoch):
    return {
        "epoch": epoch,
        "best_acc": float(epoch),
        "state_dict": {"weight": torch.tensor([float(epoch)])},
    }


def test_latest_and_best_checkpoints_are_independent(tmp_path):
    manager = _manager(tmp_path)

    manager.save_checkpoint(_checkpoint(0), is_best=True)
    latest = torch.load(tmp_path / "latest.pth.tar", weights_only=False)
    best = torch.load(tmp_path / "model_best.pth.tar", weights_only=False)
    assert latest["epoch"] == 0
    assert best["epoch"] == 0
    assert latest["dataset"] == "imagenet"
    assert best["dataset"] == "imagenet"

    manager.save_checkpoint(_checkpoint(1), is_best=False)
    latest = torch.load(tmp_path / "latest.pth.tar", weights_only=False)
    best = torch.load(tmp_path / "model_best.pth.tar", weights_only=False)
    assert latest["epoch"] == 1
    assert best["epoch"] == 0

    manager.save_checkpoint(_checkpoint(2), is_best=True)
    latest = torch.load(tmp_path / "latest.pth.tar", weights_only=False)
    best = torch.load(tmp_path / "model_best.pth.tar", weights_only=False)
    assert latest["epoch"] == 2
    assert best["epoch"] == 2
    assert not [path for path in tmp_path.iterdir() if path.name.endswith(".tmp")]
