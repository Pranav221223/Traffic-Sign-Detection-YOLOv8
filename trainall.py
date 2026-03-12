from ultralytics import YOLO
import os

# Define the path to your data configuration file
data_config_path = 'data/data.yaml' # <--- CHANGE THIS

# Define the model you want to use.
# 'yolov8n.pt' is the nano model (fastest, smallest, lower accuracy)
# 'yolov8s.pt' is the small model (good balance)
# 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt' are larger models (higher accuracy, slower)
model_to_use = 'yolov8n.pt'

# --- Training ---
print(f"Loading model: {model_to_use}")
model = YOLO(model_to_use) # Load a pre-trained YOLOv8 model

print("Starting training...")
# Train the model
# 'data': path to your data.yaml file
# 'epochs': number of training epochs (start with a small number, increase if needed)
# 'imgsz': image size for training (usually 640 is standard for YOLOv8)
# 'batch': batch size (adjust based on your GPU memory)
results = model.train(data=data_config_path, epochs=50, imgsz=640, batch=16) # You can adjust epochs and batch size

print("Training finished.")

# The trained model will be saved automatically by Ultralytics
# It's typically saved in runs/detect/trainX/weights/best.pt where X is an incrementing number
# The 'results' object contains information about the training run, including the save path.
# We can find the path to the best model from the results object or by looking in the runs folder.

# Example way to find the path (assuming the run was successful and 'best.pt' was saved)
# Note: This is an example path, you might need to inspect your 'runs' folder
# after training to find the exact path to the 'best.pt' file.
# The path will be something like: 'runs/detect/train/weights/best.pt' or 'runs/detect/train2/weights/best.pt' etc.
# Let's print a message indicating where to find the saved model.
print("\nTrained model saved successfully by Ultralytics.")
print("Look for the 'runs/detect/' directory created in the same location where you ran the training script.")
print("The best performing model checkpoint will be saved as 'weights/best.pt' inside the latest 'trainX' folder.")