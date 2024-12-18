from ultralytics import YOLO


#model = YOLO("yolov8n.yaml")  # build a new model from scratch

# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
# #epochs based on the dataset size
# # Train the model
# results = model.train(
#     data='data.yaml',
#     epochs=50,       # Start with 50 epochs
#     imgsz=640,       # Image size
#     batch=8,         # Batch size (adjust based on GPU RAM)
#     patience=10,     # Early stopping after 10 epochs with no improvement
#     workers=4        # Number of CPU workers for loading data
# )

# Evaluate the trained model on the validation set
model = YOLO('/home/kapis20/Projects/3D/3D/runs/detect/train2/weights/best.pt')
metrics = model.val()

# Print evaluation results
print(metrics)
print(f"mAP50-95: {metrics.box.map:.3f}")      # mAP50-95
print(f"mAP50: {metrics.box.map50:.3f}")       # mAP50
print(f"Precision: {metrics.box.precision:.3f}")  # Precision
print(f"Recall: {metrics.box.recall:.3f}")        # Recall
