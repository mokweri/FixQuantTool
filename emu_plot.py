import matplotlib.pyplot as plt
import numpy as np

# Define layers and their rounding modes
layers = ['Layer 1', 'Layer 2']
rounding_modes = ['RND', 'RND_ZERO', 'RND_MIN_INF', 'RND_CONV', 'TRN']

# Synthetic quantization errors for each layer and mode
errors = {
    'Layer 1': [0.03, 0.04, 0.02, 0.05, 0.06],  # Simulated MSE for Layer 1
    'Layer 2': [0.05, 0.07, 0.06, 0.08, 0.09],  # Simulated MSE for Layer 2
}

# Define the x locations for the groups
x = np.arange(len(layers))

# Bar width and offsets
bar_width = 0.1
offsets = np.arange(len(rounding_modes)) * bar_width - (len(rounding_modes) - 1) * bar_width / 2

# Create the plot
plt.figure(figsize=(6, 5))
for i, mode in enumerate(rounding_modes):
    mode_errors = [errors[layer][i] for layer in layers]  # Collect errors for this mode
    plt.bar(x + offsets[i], mode_errors, width=bar_width, label=mode)

# Add labels, title, and legend
plt.xticks(x, layers)  # Set layer names on x-axis
plt.ylabel('Quantization Error (MSE)')
# plt.title('Quantization Errors Across Layers and Rounding Modes')
plt.legend(title='Rounding Modes', bbox_to_anchor=(0.05, 1), loc='upper left')

# Annotate bars with error values
for i, mode in enumerate(rounding_modes):
    for j, layer in enumerate(layers):
        plt.text(
            x[j] + offsets[i],
            errors[layer][i] + 0.001,
            f"{errors[layer][i]:.2f}",
            ha='center',
            va='bottom',
            fontsize=9,
        )

# Display the plot
plt.tight_layout()
plt.show()
