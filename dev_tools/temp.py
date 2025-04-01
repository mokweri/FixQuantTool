import os
import platform

if platform.system() == "Windows":
    dataset_path = "D:\\datasets\\imagenet"  # or any Windows-style path
elif platform.system() == "Linux":
    dataset_path = "/mnt/datasets/imagenet"  # for Ubuntu/Linux
else:
    raise RuntimeError("Unsupported OS")

print(f"Running on {platform.system()} → Dataset path set to: {dataset_path}")

