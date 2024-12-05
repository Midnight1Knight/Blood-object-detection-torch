import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import torch
from torchvision.transforms import functional as F
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from xml.etree import ElementTree as et


def rescale_predictions(predictions, original_width, original_height):
    """
    Rescale bounding box coordinates from 224x224 to the original image dimensions.

    Args:
        predictions (dict): Dictionary containing 'boxes', 'labels', and 'scores'.
        original_width (int): Original image width.
        original_height (int): Original image height.

    Returns:
        dict: Predictions with rescaled bounding box coordinates.
    """
    rescaled_boxes = []

    for box in predictions['boxes']:
        xmin, ymin, xmax, ymax = box
        # Rescale each coordinate
        xmin_rescaled = xmin * original_width / 224
        ymin_rescaled = ymin * original_height / 224
        xmax_rescaled = xmax * original_width / 224
        ymax_rescaled = ymax * original_height / 224
        rescaled_boxes.append([xmin_rescaled, ymin_rescaled, xmax_rescaled, ymax_rescaled])

    # Create a new predictions dictionary with rescaled boxes
    rescaled_predictions = {
        'boxes': rescaled_boxes,
        'labels': predictions['labels'],
        'scores': predictions['scores']
    }

    return rescaled_predictions


def run_inference(model, image_path, device, classes, confidence_threshold=0.5):
    """
    Run inference on a single image and return the predictions.

    Args:
        model (torch.nn.Module): The trained object detection model.
        image_path (str): Path to the input image.
        device (torch.device): Device to run the inference on ('cpu' or 'cuda').
        classes (list): List of class names.
        confidence_threshold (float): Minimum confidence for predictions to be considered.

    Returns:
        dict: A dictionary containing the boxes, labels, and scores of the predictions.
    """
    model.eval()

    # Load and preprocess the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image_resized = cv2.resize(image_rgb, (224, 224))  # Resize to match model's input size
    image_normalized = image_resized / 255.0  # Normalize pixel values
    image_tensor = F.to_tensor(image_normalized).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(image_tensor)[0]

    # Extract predictions above the confidence threshold
    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()

    filtered_boxes = []
    filtered_labels = []
    filtered_scores = []

    for box, label, score in zip(boxes, labels, scores):
        if score >= confidence_threshold:
            filtered_boxes.append(box)
            filtered_labels.append(classes[label])
            filtered_scores.append(score)

    return {
        'boxes': filtered_boxes,
        'labels': filtered_labels,
        'scores': filtered_scores
    }


def get_object_detection_model(num_classes):
    """
    Create a Faster R-CNN model with a ResNet-50 backbone for object detection.

    Args:
        num_classes (int): The number of classes (including background) in the dataset.

    Returns:
        torch.nn.Module: A Faster R-CNN model with a custom classification head.
    """
    # Load a pre-trained Faster R-CNN model with a ResNet-50 backbone
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the classifier head with a new one for our dataset
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_classes(path):
    classes = set()
    for filename in os.listdir(path):
        if filename.endswith('.xml'):
            with open(os.path.join(path, filename)) as f:
                tree = et.parse(f)
                root = tree.getroot()
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    classes.add(name)
    return list(classes)


def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if file_path:
        process_image(file_path)


def select_directory():
    directory_path = filedialog.askdirectory()
    if directory_path:
        for file in os.listdir(directory_path):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                process_image(os.path.join(directory_path, file))


def process_image(image_path):
    global img_label, result_label
    # Run inference
    predictions = run_inference(model, image_path, device, classes, confidence_threshold=0.5)
    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]
    rescaled_predictions = rescale_predictions(predictions, original_width, original_height)

    # Visualize results
    for box, label, score in zip(rescaled_predictions['boxes'], rescaled_predictions['labels'], rescaled_predictions['scores']):
        xmin, ymin, xmax, ymax = map(int, box)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, f"{label} ({score:.2f})", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Convert for Tkinter display
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    photo = ImageTk.PhotoImage(image)

    # Update GUI
    img_label.config(image=photo)
    img_label.image = photo
    num_classes = len(set(rescaled_predictions['labels']))
    result_label.config(text=f"Number of detected classes: {num_classes}")


def resize_image_and_boxes(image, boxes, target_width, target_height):
    """
    Resize the image to the target dimensions while adjusting bounding boxes.

    Args:
        image (np.ndarray): The original image as a NumPy array (H, W, C).
        boxes (list): List of bounding boxes in the format [xmin, ymin, xmax, ymax].
        target_width (int): The target width for resizing.
        target_height (int): The target height for resizing.

    Returns:
        resized_image (np.ndarray): The resized image.
        resized_boxes (list): The resized bounding boxes.
    """
    # Get the original dimensions of the image
    original_height, original_width = image.shape[:2]

    # Compute the resize ratios
    width_ratio = target_width / original_width
    height_ratio = target_height / original_height

    # Resize the image
    resized_image = cv2.resize(image, (target_width, target_height))

    # Adjust bounding boxes
    resized_boxes = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        resized_boxes.append([
            xmin * width_ratio,
            ymin * height_ratio,
            xmax * width_ratio,
            ymax * height_ratio
        ])

    return resized_image, resized_boxes


# Initialize main Tkinter window
root = tk.Tk()
root.title("Object Detection Inference")
root.geometry("800x600")

# Add GUI Components
btn_file = tk.Button(root, text="Load Single Image", command=select_file)
btn_file.pack(pady=10)

btn_dir = tk.Button(root, text="Load Directory", command=select_directory)
btn_dir.pack(pady=10)

img_label = tk.Label(root)
img_label.pack(pady=10)

result_label = tk.Label(root, text="Number of detected classes: 0")
result_label.pack(pady=10)

# Load model and classes
train_dir = 'data/karp/train'
test_dir = 'data/karp/test'

train_classes = get_classes(train_dir)
test_classes = get_classes(test_dir)
classes = list(set(train_classes + test_classes))

device = torch.device('cpu')  # or 'cuda'
model = get_object_detection_model(len(classes))
model.load_state_dict(torch.load('logs/pytorch_model-e3.pt', weights_only=False))

# Run Tkinter main loop
root.mainloop()
