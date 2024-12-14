import tkinter as tk
from app import ObjectDetectionApp
import torch
from load_data import load_up


if __name__ == "__main__":
    device = torch.device('cpu')
    data = load_up()

    root = tk.Tk()
    root.geometry("1000x1000")
    root.title("Object Detection Inference")
    app = ObjectDetectionApp(root, data, device)
    root.mainloop()
