# import numpy as np
# import tflite_runtime.interpreter as tflite
# import cv2

# # Load TFLite model and allocate tensors
# interpreter = tflite.Interpreter(model_path="yolov8.tflite")
# interpreter.allocate_tensors()

# # Get input and output tensors
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # Preprocess input image
# image_path = "your_image.jpg"  # Replace with your image path
# image = cv2.imread(image_path)
# image_resized = cv2.resize(image, (640, 640))  # Resize to model's input shape
# input_data = np.expand_dims(image_resized, axis=0).astype(np.float32)

# # Set input tensor
# interpreter.set_tensor(input_details[0]['index'], input_data)

# # Run inference
# interpreter.invoke()

# # Get the output
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print("Inference Results:", output_data)
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="yolov8.tflite")
interpreter.allocate_tensors()

print("Model loaded successfully!")
