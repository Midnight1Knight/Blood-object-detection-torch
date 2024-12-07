import torch
import tkinter as tk
from utils import get_classes, get_object_detection_model
from app import ObjectDetectionApp

if __name__ == "__main__":
    train_dir = 'data/karp/train'
    test_dir = 'data/karp/test'
    device = torch.device('cpu')
    train_classes = get_classes(train_dir)
    test_classes = get_classes(test_dir)
    classes = list(set(train_classes + test_classes))
    model = get_object_detection_model(len(classes))
    model.load_state_dict(torch.load('logs/pytorch_model-e10.pt', weights_only=False))

    root = tk.Tk()
    root.geometry("1000x800")
    root.title("Object Detection Inference")
    app = ObjectDetectionApp(root, model, device, classes)
    root.mainloop()
