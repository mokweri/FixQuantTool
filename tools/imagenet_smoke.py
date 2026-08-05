"""Run one ImageNet training step to verify data, model, and GPU integration."""

import argparse

import torch
import torch.nn.functional as F

from fixquant.data import ImagenetDataProvider
from fixquant.models import get_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--model", default="mobilenet_v2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the Arrhenius smoke test")

    provider = ImagenetDataProvider(
        save_path=args.dataroot,
        train_batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        n_worker=args.workers,
        pin_memory=True,
    )
    images, labels = next(iter(provider.train_loader))
    images = images.cuda(non_blocking=True)
    labels = labels.cuda(non_blocking=True)

    model = get_model(args.model, pretrained=False).cuda().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    output = model(images)
    loss = F.cross_entropy(output, labels)
    loss.backward()
    optimizer.step()

    val_images, val_labels = next(iter(provider.val_loader))
    print(f"dataset={args.dataroot}")
    print(f"train_samples={len(provider.train_loader.dataset)}")
    print(f"validation_samples={len(provider.val_loader.dataset)}")
    print(f"train_batch={tuple(images.shape)}, labels={tuple(labels.shape)}")
    print(f"validation_batch={tuple(val_images.shape)}, labels={tuple(val_labels.shape)}")
    print(f"model={args.model}, loss={loss.item():.6f}")
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print("ImageNet smoke test: PASS")


if __name__ == "__main__":
    main()
