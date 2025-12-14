import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import torch
from dotenv import load_dotenv

from predict import FullPipeline

load_dotenv()

DETECTOR_MODEL_PATH = os.getenv("DETECTOR_MODEL_PATH")
CLASSIFIER_CROPPED_PATH = "producer_classifier.pth"
CLASSIFIER_FULL_PATH = "producer_classifier_full.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 1.0

class App:
    def __init__(self, master):
        self.master = master
        master.title("Producer Classifier Testing")
        master.geometry("800x600")

        self.btn_open = tk.Button(
            master,
            text="Выбрать изображение…",
            command=self.open_image
        )
        self.btn_open.pack(pady=10)

        self.canvas = tk.Canvas(
            master,
            width=640,
            height=480,
            bg="#333333"
        )
        self.canvas.pack()

        self.lbl_result = tk.Label(
            master,
            text="",
            font=("Arial", 14)
        )
        self.lbl_result.pack(pady=10)

        if not DETECTOR_MODEL_PATH or not os.path.exists(DETECTOR_MODEL_PATH):
            messagebox.showerror(
                "Ошибка конфигурации",
                "Не найден путь к весам детектора.\n"
                "Проверьте переменную окружения DETECTOR_MODEL_PATH."
            )
            master.destroy()
            return

        try:
            self.pipeline = FullPipeline(
                detector_weights_path=DETECTOR_MODEL_PATH,
                classifier_cropped_path=CLASSIFIER_CROPPED_PATH,
                classifier_full_path=CLASSIFIER_FULL_PATH,
                threshold=THRESHOLD,
                device=DEVICE
            )
        except Exception as e:
            messagebox.showerror(
                "Ошибка инициализации",
                f"Не удалось загрузить модели:\n{e}"
            )
            master.destroy()


    def open_image(self):
        path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif")
            ]
        )
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            messagebox.showerror(
                "Ошибка",
                "Не удалось загрузить изображение"
            )
            return

        try:
            result = self.pipeline.predict(path)
        except Exception as e:
            messagebox.showerror(
                "Ошибка при предсказании",
                str(e)
            )
            return

        producer = result["producer"]
        confidence = result["confidence"]
        source = result["source"]

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img = pil_img.resize((640, 480))
        self.photo = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        self.canvas.create_image(
            0, 0,
            image=self.photo,
            anchor=tk.NW
        )

        text = (
            f"Producer: {producer}\n"
            f"Confidence: {confidence * 100:.1f}%\n"
            f"Source: {source}"
        )
        self.lbl_result.config(text=text)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
