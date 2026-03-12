from ultralytics import YOLO
import cv2
import os
from PIL import Image # <--- Add this line

# Define the path to your trained model's best.pt file
# You need to find this file in your 'runs/detect/' directory after training.
# Example path:
trained_model_path = 'runs/train3/weights/best.pt' # <--- CHANGE THIS TO YOUR ACTUAL .pt FILE PATH

# Define the path to the image(s) you want to predict on
# This can be a single image path, a list of image paths, or a directory containing images
image_to_predict_path = '123.jpg' # <--- CHANGE THIS

# --- Prediction ---
print(f"Loading trained model from: {trained_model_path}")
try:
    model = YOLO(trained_model_path) # Load your custom trained model
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Please verify the path to the trained model: {trained_model_path}")
    exit()

print(f"Performing prediction on: {image_to_predict_path}")
# Perform inference on the image(s)
# 'conf': confidence threshold (only show detections with confidence > this value)
# 'iou': IoU threshold for Non-Maximum Suppression (NMS)
results = model.predict(image_to_predict_path, conf=0.25, iou=0.7) # Adjust conf and iou as needed

# --- Process and Visualize Results ---
print("Processing results...")

# The results object contains details about the detections
# Each 'result' in 'results' corresponds to one input image (if you pass multiple images)
for r in results:
    # 'r.boxes' contains the bounding box coordinates, confidence, and class
    # 'r.names' is a dictionary mapping class IDs to class names

    # r.plot() returns a numpy array (OpenCV format - BGR)
    im_array = r.plot()

    # Convert BGR numpy array to RGB (needed for PIL) and then to PIL Image
    im = Image.fromarray(im_array[..., ::-1])
    im.show() # Display the image using PIL's show method

    # You can also access the raw data
    print("\nDetections found:")
    for box in r.boxes:
        class_id = int(box.cls)
        confidence = float(box.conf)
        bbox_xyxy = box.xyxy.tolist()[0] # [x1, y1, x2, y2]

        class_name = r.names[class_id]

        print(f"  - Class: {class_name} (ID: {class_id}), Confidence: {confidence:.2f}, BBox: {bbox_xyxy}")

print("\nPrediction finished. Visualizations displayed (may appear in separate windows).")