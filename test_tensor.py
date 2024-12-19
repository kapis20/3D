import tensorflow as tf
import onnx
from onnx_tf.backend import prepare

print(f"TensorFlow version: {tf.__version__}")
print(f"ONNX version: {onnx.__version__}")
