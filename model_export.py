from ultralytics import YOLO

# Load your YOLO model
model = YOLO('/home/kapis20/Projects/3D/3D/runs/detect/train2/weights/best.pt')


# Export to ONNX format
model.export(format="onnx", dynamic=False)  # Disable dynamic for TFLite compatibility

