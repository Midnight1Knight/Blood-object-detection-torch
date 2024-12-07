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
    rescaled_boxes = []
    for box in predictions['boxes']:
        xmin, ymin, xmax, ymax = box
        rescaled_boxes.append([
            xmin * original_width / 224,
            ymin * original_height / 224,
            xmax * original_width / 224,
            ymax * original_height / 224
        ])
    return {
        'boxes': rescaled_boxes,
        'labels': predictions['labels'],
        'scores': predictions['scores']
    }


def run_inference(model, image_path, device, classes, confidence_threshold=0.5):
    model.eval()
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image_resized = cv2.resize(image_rgb, (224, 224))
    image_normalized = image_resized / 255.0
    image_tensor = F.to_tensor(image_normalized).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(image_tensor)[0]

    filtered_boxes, filtered_labels, filtered_scores = [], [], []
    for box, label, score in zip(predictions['boxes'].cpu().numpy(), predictions['labels'].cpu().numpy(), predictions['scores'].cpu().numpy()):
        if score >= confidence_threshold:
            filtered_boxes.append(box)
            filtered_labels.append(classes[label])
            filtered_scores.append(score)

    return {'boxes': filtered_boxes, 'labels': filtered_labels, 'scores': filtered_scores}


def get_object_detection_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
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
                    classes.add(obj.find('name').text)
    return list(classes)


def process_image(image_path, model, device, classes, font_size, cell_thickness, confidence_threshold=0.5):
    predictions = run_inference(model, image_path, device, classes, confidence_threshold)
    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]
    rescaled_predictions = rescale_predictions(predictions, original_width, original_height)

    for box, label, score in zip(rescaled_predictions['boxes'], rescaled_predictions['labels'], rescaled_predictions['scores']):
        xmin, ymin, xmax, ymax = map(int, box)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), cell_thickness)
        cv2.putText(image, f"{label} ({score:.2f})", (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size / 24, (0, 255, 0), 1)

    max_width, max_height = 600, 400
    aspect_ratio = original_width / original_height
    if aspect_ratio > 1:
        display_width = max_width
        display_height = int(max_width / aspect_ratio)
    else:
        display_height = max_height
        display_width = int(max_height * aspect_ratio)

    image_resized = cv2.resize(image, (display_width, display_height))
    image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_resized), rescaled_predictions


class ObjectDetectionApp:
    def __init__(self, root, model, device, classes):
        self.root = root
        self.model = model
        self.device = device
        self.classes = classes
        self.font_size = 12
        self.cell_thickness = 2
        self.confidence_threshold = 0.5
        self.img_label = tk.Label(root)
        self.img_label.pack(pady=10)
        self.result_label = tk.Label(root, text="Number of detected classes: 0")
        self.result_label.pack(pady=10)
        self.init_ui()

    def init_ui(self):
        tk.Button(self.root, text="Load Single Image", command=self.select_file).pack(pady=10)
        font_slider = tk.Scale(self.root, from_=8, to=48, orient=tk.HORIZONTAL, label="Font Size", command=self.update_font_size)
        font_slider.set(self.font_size)
        font_slider.pack(pady=10)
        thickness_slider = tk.Scale(self.root, from_=1, to=10, orient=tk.HORIZONTAL, label="Cell Thickness", command=self.update_cell_thickness)
        thickness_slider.set(self.cell_thickness)
        thickness_slider.pack(pady=10)
        self.root.bind("<Escape>", self.exit_fullscreen)

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.display_image(file_path)

    def update_font_size(self, value):
        self.font_size = int(value)

    def update_cell_thickness(self, value):
        self.cell_thickness = int(value)

    def exit_fullscreen(self, event):
        self.root.attributes('-fullscreen', False)

    def display_image(self, image_path):
        img, predictions = process_image(
            image_path,
            self.model,
            self.device,
            self.classes,
            self.font_size,
            self.cell_thickness,
            self.confidence_threshold
        )
        photo = ImageTk.PhotoImage(img)
        self.img_label.config(image=photo)
        self.img_label.image = photo

        unique_labels = set(predictions['labels'])
        self.result_label.config(text=f"Detected: {len(unique_labels)} ({', '.join(unique_labels)})")


if __name__ == "__main__":
    train_dir = 'data/karp/train'
    test_dir = 'data/karp/test'
    device = torch.device('cpu')
    train_classes = get_classes(train_dir)
    test_classes = get_classes(test_dir)
    classes = list(set(train_classes + test_classes))
    model = get_object_detection_model(len(classes))
    model.load_state_dict(torch.load('logs/pytorch_model-e3.pt', weights_only=False))

    root = tk.Tk()
    root.title("Object Detection Inference")
    app = ObjectDetectionApp(root, model, device, classes)
    root.mainloop()
