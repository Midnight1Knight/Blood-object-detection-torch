import torch
import tkinter as tk
from utils import get_classes, get_object_detection_model
from app import ObjectDetectionApp
import json

if __name__ == "__main__":
    fish = 'osetr'
    device = torch.device('cpu')
    with open(f'data/{fish}/classes_idx.json', 'r') as file:
        classes_idx = json.load(file)
    classes = list(classes_idx.keys())
    model = get_object_detection_model(len(classes))
    model.load_state_dict(torch.load(f'logs/{fish}/pytorch_model-e3.pt', weights_only=False))

    root = tk.Tk()
    root.geometry("1000x800")
    root.title("Object Detection Inference")
    app = ObjectDetectionApp(root, model, device, classes)
    root.mainloop()
