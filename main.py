import torch
import tkinter as tk
from utils import get_classes, get_object_detection_model
from app import ObjectDetectionApp
import json

if __name__ == "__main__":
    device = torch.device('cpu')
    with open('data/karp/classes.json', 'r') as file:
        classes_idx = json.load(file)
    classes = list(classes_idx.keys())
    model = get_object_detection_model(len(classes) + 1)
    model.load_state_dict(torch.load('logs/pytorch_model-e4.pt', weights_only=False))

    root = tk.Tk()
    root.geometry("1000x800")
    root.title("Object Detection Inference")
    app = ObjectDetectionApp(root, model, device, classes)
    root.mainloop()
