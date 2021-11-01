# Train a TensorFlow model to use for inference using the ONNX runtime
Create a python virtual environment and install the required packages to train a tensorflow model.
```
python -m venv train-env
.\train-env\Scripts\activate
python -m pip install -r requirements.txt
```
Run the `train` notebook to create a `minst_model.onnx` ONNX model to use for inference in the C++ project.


Other ONNX examples:

https://github.com/microsoft/onnxruntime-extensions/blob/main/tutorials/tf2onnx_custom_ops_tutorial.ipynb

https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/keras-resnet50.ipynb
