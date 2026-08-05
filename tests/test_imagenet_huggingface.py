import json

import pytest
from PIL import Image

from fixquant.data.imagenet import ImagenetDataProvider


def _write_split(root, name, labels):
    datasets = pytest.importorskip("datasets")
    images = [Image.new("RGB", (32, 32), (label, 0, 0)) for label in labels]
    split = datasets.Dataset.from_dict({"image": images, "label": labels})
    split.save_to_disk(root / name)


def test_huggingface_imagenet_dataset(tmp_path):
    _write_split(tmp_path, "train", [0, 1, 2])
    _write_split(tmp_path, "validation", [3, 4])
    (tmp_path / "dataset_dict.json").write_text(
        json.dumps({"splits": ["train", "validation"]})
    )

    provider = ImagenetDataProvider(
        save_path=tmp_path,
        train_batch_size=2,
        test_batch_size=2,
        n_worker=0,
        image_size=16,
    )

    train_images, train_labels = next(iter(provider.train_loader))
    val_images, val_labels = next(iter(provider.val_loader))

    assert provider.is_huggingface_dataset
    assert len(provider.train_loader.dataset) == 3
    assert len(provider.val_loader.dataset) == 2
    assert train_images.shape == (2, 3, 16, 16)
    assert val_images.shape == (2, 3, 16, 16)
    assert set(train_labels.tolist()).issubset({0, 1, 2})
    assert val_labels.tolist() == [3, 4]
