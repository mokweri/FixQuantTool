import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt

# Load a pre-trained ResNet-50 model from torchvision
model = models.resnet50(pretrained=True)

# Collect the weight ranges for each layer
layer_names = []
weight_ranges = []

for name, layer in model.named_modules():
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        weights = layer.weight.detach().cpu().numpy()
        layer_names.append(name)
        weight_ranges.append(weights.flatten())  # Flatten to collect weights as a single array for each layer

# Create a box plot for the weight ranges across layers
plt.figure(figsize=(15, 6))
plt.boxplot(weight_ranges, vert=True, patch_artist=True, showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "red", "markersize": 4},
            boxprops=dict(color="blue"), whiskerprops=dict(color="blue"),
            capprops=dict(color="blue"), medianprops=dict(color="green"))

# Customize plot appearance
plt.xticks(range(1, len(layer_names) + 1), layer_names, rotation=90, fontsize=8)
plt.yticks(fontsize=8)
plt.xlabel("Layer")
plt.ylabel("Effective Weight")
plt.ylim(-0.5, 0.5)  # Limiting y-axis range to make the plot more readable
plt.title("Effective Weight Range (FP ResNet-50)")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
