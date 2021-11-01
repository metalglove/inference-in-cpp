# Inference in C++ using ONNX

## Train a TensorFlow MNIST model to use for inference using the ONNX runtime
Create a python virtual environment and install the required packages to train a tensorflow model.
```
python -m venv train-env
.\train-env\Scripts\activate
python -m pip install -r requirements.txt
```
Run the `train` notebook to create a `minst_model.onnx` ONNX model to use for inference in the C++ project.


## Run the inference in C++ using ONNX
To run the C++ inference code, the ONNX runtime needs to be installed. <br>
The `CMakeLists.txt` file will fetch the ONNX runtime and build it. <br>
However, to build the code the Visual Studio 2019 compiler is needed. <br>

When running the code, the following will be printed:
```
Row:  0 0000000000000000000000000000
Row:  1 0000000000000000000000000000
Row:  2 0000000000000000000000000000
Row:  3 0000000000000000000000000000
Row:  4 0000000000011110000000000000
Row:  5 0000000000011110000000000000
Row:  6 0000000000011110000000000000
Row:  7 0000000000011110000000000000
Row:  8 0000000000011110000000000000
Row:  9 0000000000011110000000000000
Row: 10 0000000000011110000000000000
Row: 11 0000000000011110000000000000
Row: 12 0000000000011110000000000000
Row: 13 0000000000011110000000000000
Row: 14 0000000000011110000000000000
Row: 15 0000000000011110000000000000
Row: 16 0000000000011110000000000000
Row: 17 0000000000011110000000000000
Row: 18 0000000000011110000000000000
Row: 19 0000000000011110000000000000
Row: 20 0000000000011110000000000000
Row: 21 0000000000011110000000000000
Row: 22 0000000000011110000000000000
Row: 23 0000000000011110000000000000
Row: 24 0000000000000000000000000000
Row: 25 0000000000000000000000000000
Row: 26 0000000000000000000000000000
Row: 27 0000000000000000000000000000
0: 1.15297e-09
1: 0.999905
2: 1.30885e-07
3: 3.53474e-07
4: 9.88348e-08
5: 1.24045e-07
6: 5.27436e-06
7: 1.38264e-08
8: 8.94342e-05
9: 1.15949e-09
The result: 1
```

# ONNXruntime examples
This repository is a trimmed down version of the MNIST example. <br>
https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx
