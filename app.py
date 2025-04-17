import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk
from utils import process_image
from collections import Counter


class ObjectDetectionApp:
    def __init__(self, root, data, device):
        self.root = root
        self.data = data
        self.device = device
        self.classes = None
        self.model = None
        self.font_size = 30
        self.cell_thickness = 7
        self.confidence_threshold = 0.1
        self.selected_fish = None
        self.img_label = tk.Label(root)
        self.result_label = tk.Label(root)
        self.init_start_screen()

    def init_start_screen(self):
        """Initial screen to select the fish type."""
        title_label = tk.Label(self.root, text="Выберите клетки какой рыбы подсчитать", font=("Arial", 16))
        title_label.pack(pady=20)

        osetr_button = tk.Button(self.root, text="Осетр", command=lambda: self.select_fish("osetr"))
        osetr_button.pack(pady=10)

        carp_button = tk.Button(self.root, text="Карп", command=lambda: self.select_fish("karp"))
        carp_button.pack(pady=10)

    def select_fish(self, fish_type):
        """Handle the selection of fish and proceed to the main UI."""
        self.selected_fish = fish_type
        self.load_selected_model(fish_type)
        self.clear_screen()
        self.init_ui()

    def load_selected_model(self, fish_type):
        self.model = self.data[fish_type]["model"]
        self.classes = self.data[fish_type]["classes"]

    def clear_screen(self):
        """Clear the current widgets from the root window."""
        for widget in self.root.winfo_children():
            widget.destroy()

    def init_ui(self):
        """Main application interface."""
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

        self.img_label = tk.Label(self.root)
        self.img_label.pack(pady=10)

        self.result_label = tk.Label(self.root, text="", font=("Arial", 14))
        self.result_label.pack(pady=10)

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

        result_text = f"Detected for {self.selected_fish}:\n"  # Include the selected fish type in the result
        pred_counts = Counter(predictions['labels'])
        len_of_preds = len(predictions["labels"])
        for label in pred_counts:
            result_text += f"{label}: {pred_counts[label] / len_of_preds * 100:.2f}% ({pred_counts[label]} шт.)\n"
        self.result_label.config(text=result_text)
