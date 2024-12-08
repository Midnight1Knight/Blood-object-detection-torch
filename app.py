import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk
from utils import process_image


class ObjectDetectionApp:
    def __init__(self, root, model, device, classes):
        self.root = root
        self.model = model
        self.device = device
        self.classes = classes
        self.font_size = 30
        self.cell_thickness = 7
        self.confidence_threshold = 0.0
        self.img_label = tk.Label(root)
        self.img_label.pack(pady=10)
        self.result_label = tk.Label(root)
        self.result_label.pack(pady=10)
        self.init_ui()

    def init_ui(self):
        tk.Button(self.root, text="Load Single Image", command=self.select_file).pack(pady=10)
        font_slider = tk.Scale(
            self.root, from_=8, to=48, orient=tk.HORIZONTAL,
            label="Font Size", command=self.update_font_size
        )
        font_slider.set(self.font_size)
        font_slider.pack(pady=10)
        thickness_slider = tk.Scale(
            self.root, from_=1, to=10, orient=tk.HORIZONTAL,
            label="Cell Thickness", command=self.update_cell_thickness
        )
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

        result_text = "Detected:\n"
        for label in unique_labels:
            label_count = predictions['labels'].count(label)  # Count how many times this label appears
            result_text += f"{label}: {label_count}\n"  # Add result to the text

        self.result_label.config(text=result_text)
