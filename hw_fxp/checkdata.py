import numpy as np

# Load weights
weights = np.fromfile("weights.data", dtype=np.int8)
print("Weights:", weights[:20])

# Load biases
biases = np.fromfile("biases.data", dtype=np.int8)
print("Biases:", biases[:20])

# Load input
input_data = np.fromfile("test_image.data", dtype=np.int8)
print("Input:", input_data[:20])

# Load reference output
ref_output = np.fromfile("ref_output2.data", dtype=np.int8)
print("Reference2 Output:", ref_output[110:130])

ref_output = np.fromfile("ref_output.data", dtype=np.int8)
print("Reference1 Output:", ref_output[110:130])