# Reference files
The `ref_output.data` and `test_image.data` files are raw binary files created from numpy arrays.
These files contain the input image and the numerical outputs from the model in a machine-readable binary format.

#### **File Contents**
- The file stores a **raw binary representation** of the numpy array's data.
- It does not include any metadata, such as: `Shape` or `dtype`


#### **Key Specifications**
1. **Binary Format**:
The data is stored in the memory layout of the numpy array (row-major order/**C-contiguous**).
2. **Element Size**:
Now the values is stored as `int8` hence the following file sizes:
    - `test_image.data` has ``IMAGE_FILE_SIZE = (224x224x3) `` bytes
    - `ref_output.data` has ``REF_FILE_SIZE = (1000x1) `` bytes

#### **How to Interpret the File**
To interpret the data correctly, you need the following additional information:
1. The `test_image.data` has the shape `[batch_size, height, width, channels]` -> [1,224,224,3]
2. The `ref_output.data` has the shape `[batch_size, classes]` -> [1,1000]

- The `test_image.data` is quantized with `frac_input=5`
- The `ref_output.data` is quantized with `frac_out=2`


#### **Reconstructing the Array**
The data in the binary files is in contigous format (row-major order), so you need to reshape for use as input to the model.
To read and recreate the array for processing, you can use python or C/C++:

``` python
import numpy as np

data = np.fromfile("ref_output.data", dtype=np.int8)
data = data.reshape(original_shape)  # Replace `original_shape` with the actual shape
```
I have also provided a `CPP` file to read the files

### Fixed point Arithmetic
The onnx file provided has quantized weights but in floating point format.
The conversion to int is based on the parameters 
1. `frac_W` - for weights
2. `frac_B` - for biases
3. `frac_out` - for activations

The conversion formulas between floating-point and integer values are as follows:

1. **Float to Integer Conversion**:

    ```
    int_value = round(float_value * (2 ** frac_))
    ```

    - `float_value` is the input floating-point number.
    - `frac_` is the fractional scaling factor which determines the scaling of the fixed-point representation.

2. **Integer to Float Conversion**:

    ```
    float_value = int_value / (2 ** frac_)
    ```

    - `int_value` is the quantized integer representation.
    - `frac_` is the same fractional scaling factor used during the conversion to integer.

### Reference Hardware Accelerator
To have a look at how this has been implemented  in a HLS-OpenCL accelerator targeting Xilinx FPGAs you may check out
this repository (https://github.com/doonny/PipeCNN)
