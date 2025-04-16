import numpy as np
import os
import cv2
import json
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt5.QtGui import *
import sys
from segment_anything import SamPredictor, sam_model_registry
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--data_root", type=str, default="plate")
args = parser.parse_args()

device = args.device
data_root = args.data_root

sam = sam_model_registry["vit_h"]("../saved_pretrained/sam_vit_h_4b8939.pth").to(device)
predictor = SamPredictor(sam)


class ImageWindow(QDialog):
    """A window to display the full-sized image."""
    def __init__(self, folder_path, image_name,):
        QDialog.__init__(self,)
        self.setWindowTitle("{}".format(image_name))
        self.setGeometry(200, 200, 500, 500)
        self.folder_path = folder_path
        self.start_point = None
        self.end_point = None
        self.current_rect = None
        self.segm_save_path = os.path.join(folder_path, "{}_mask.png".format(image_name))
            
        self.image = cv2.imread(os.path.join(folder_path, image_name))
        self.display_image = self.image.copy()

        self.canvas = QLabel(self)
        self.canvas.setAlignment(Qt.AlignCenter)
        self.update_canvas()

        # self.text_prompt = QTextEdit(self)
        # self.text_prompt.setPlaceholderText("Enter your message here...")
        button_OK = QHBoxLayout()
        button_OK = QPushButton('OK', self)
        button_OK.clicked.connect(self.save_segm)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        # layout.addWidget(self.text_prompt)
        layout.addWidget(button_OK)
        self.setLayout(layout)
        self.image_name = image_name
        self.segm_mask = None
        self.updated_segm_mask = None
        # Variables for drawing
        self.drawing = True


    def save_segm(self,):
        cv2.imwrite(self.segm_save_path, self.segm_mask.astype(np.uint8) * 255)
        self.close()
    
    def _convert_to_image_coordinates(self, event):
        """Convert mouse event coordinates to image-relative coordinates."""
        canvas_width = self.canvas.width()
        canvas_height = self.canvas.height()

        img_height, img_width, _ = self.image.shape
        margin_left = (canvas_width - img_width) // 2
        margin_top = (canvas_height - img_height) // 2
        image_x = event.pos().x() - margin_left
        image_y = event.pos().y() - margin_top
        image_x = np.clip(image_x, a_min=0, a_max=img_width - 1)
        image_y = np.clip(image_y, a_min=0, a_max=img_height - 1)
        return image_x, image_y
    
    def update_canvas(self):
        """Update the QLabel with the current image."""
        display_image = self.display_image.copy()
        h, w, c = display_image.shape
        if self.start_point is not None and self.end_point is not None:
            start_x, start_y = self.start_point
            end_x, end_y = self.end_point
            cv2.rectangle(display_image, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)  # Red rectangle
        qimage = QImage(display_image.data, w, h, 3 * w, QImage.Format_RGB888)
        # Draw the bounding box if a start and end point are defined
        self.canvas.setPixmap(QPixmap.fromImage(qimage))

    def toggle_drawing(self):
        """Toggle the drawing mode."""
        self.drawing = self.toggle_button.isChecked()

    def mousePressEvent(self, event):
        """Start drawing a bounding box."""
        self.display_image = self.image.copy()
        self.update_canvas()
        if self.drawing and event.button() == Qt.LeftButton:
            self.start_point = self._convert_to_image_coordinates(event)
    
    def mouseMoveEvent(self, event):
        """Draw the bounding box as the mouse moves."""
        if self.drawing and self.start_point is not None:
            self.display_image = self.image.copy()
            self.end_point = self._convert_to_image_coordinates(event)
            self.update_canvas()

    def mouseReleaseEvent(self, event):
        """Complete the bounding box and apply segmentation."""
        if self.drawing and event.button() == Qt.LeftButton:
            x1, y1 = self.start_point
            x2, y2 = self.end_point

            # Apply the segmentation model
            bbox = np.array([x1, y1, x2, y2])
            predictor.set_image(self.image)
            mask, _, _ = predictor.predict(box=bbox, multimask_output=False)
            mask = mask[0]
            self.segm_mask = mask
            # Overlay the mask
            mask_colored = np.zeros_like(self.image)
            mask_colored[:, :, 0] = ((mask > 0) * 255).astype(np.uint8)
            self.display_image = cv2.addWeighted(self.image.copy(), 0.5, mask_colored, 0.5, 0)
            self.start_point = None
            self.end_point = None
            self.update_canvas()

app = QApplication(sys.argv)
image_names = os.listdir(data_root)

for image_name in image_names:
    if "mask" in image_name or not image_name.endswith("png"):
        continue 
    image_path = os.path.join(data_root, image_name)
    image_window = ImageWindow(folder_path=data_root, image_name=image_name)
    image_window.show()
    image_window.exec_()
