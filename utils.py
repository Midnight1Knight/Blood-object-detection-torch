import cv2
import os
from xml.etree import ElementTree as et
from PIL import Image
import numpy as np

import torch
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


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
    for box, label, score in zip(
            predictions['boxes'].cpu().numpy(),
            predictions['labels'].cpu().numpy(),
            predictions['scores'].cpu().numpy()
    ):
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

    for box, label, score in zip(
            rescaled_predictions['boxes'], rescaled_predictions['labels'], rescaled_predictions['scores']
    ):
        xmin, ymin, xmax, ymax = map(int, box)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), cell_thickness)
        cv2.putText(image, f"{label} ({score:.2f})", (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size / 12, (0, 255, 0), 5)

    max_width, max_height = 800, 400
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
