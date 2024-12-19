from onnx_tf.backend import prepare
import onnx

# Load ONNX model
onnx_model = onnx.load("/home/kapis20/Projects/3D/3D/runs/detect/train2/weights/best.onnx")

# Convert to TensorFlow
tf_rep = prepare(onnx_model)

# Export as SavedModel
tf_rep.export_graph("yolov8_tf_model")
