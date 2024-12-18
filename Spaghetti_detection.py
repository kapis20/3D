from ultralytics import YOLO


#model = YOLO("yolov8n.yaml")  # build a new model from scratch

model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
#epochs based on the dataset size
# Train the model
results = model.train(
    data='data.yaml',
    epochs=50,       # Start with 50 epochs
    imgsz=640,       # Image size
    batch=8,         # Batch size (adjust based on GPU RAM)
    patience=10,     # Early stopping after 10 epochs with no improvement
    workers=4        # Number of CPU workers for loading data
)

# Evaluate the trained model on the validation set
metrics = model.val()

# Print evaluation results
print(metrics)