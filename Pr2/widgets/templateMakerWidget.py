import os
import glob
import cv2
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLabel, QFileDialog)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt
from .imageLabel import ImageLabel
from src.env import home_prefix


class TemplateMaker(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cv_image = None
        self.template = None
        self.source_images = []   # Список файлов исходных изображений
        self.current_source_index = -1

        main_layout = QVBoxLayout(self)
        # для исходного и шаблона
        images_layout = QHBoxLayout()
        # левая колонка для Исходное изображение с выделением шаблона
        left_layout = QVBoxLayout()
        self.source_label = ImageLabel()
        self.source_label.setFrameStyle(QLabel.Box | QLabel.Plain)
        self.source_label.setFixedSize(400, 300)
        self.source_label.setAlignment(Qt.AlignCenter)
        self.source_label.mouse_release_completed.connect(self.setTemplate)
        left_layout.addWidget(self.source_label)
        left_layout.addStretch()
        
        
        
        # правая колонка для предпоказа шаблона 
        right_layout = QVBoxLayout()
        self.template_label = QLabel()
        self.template_label.setFrameStyle(QLabel.Box | QLabel.Plain)
        self.template_label.setAlignment(Qt.AlignCenter)
        self.template_label.setFixedSize(150, 150)
        right_layout.addWidget(self.template_label)
        right_layout.addStretch()
        
        src_controls_layout = QHBoxLayout()
        self.select_source_dir_button = QPushButton("Папка с изображениями")
        self.select_source_dir_button.clicked.connect(self.selectSourceDir)
        src_controls_layout.addWidget(self.select_source_dir_button)
        self.prev_source_button = QPushButton("Предыдущая")
        self.prev_source_button.clicked.connect(self.prevSource)
        src_controls_layout.addWidget(self.prev_source_button)
        self.next_source_button = QPushButton("Следующая")
        self.next_source_button.clicked.connect(self.nextSource)
        src_controls_layout.addWidget(self.next_source_button)
        self.source_info_label = QLabel("Нет изображений")
        src_controls_layout.addWidget(self.source_info_label)
        buttons_layout = QHBoxLayout()
        self.open_button = QPushButton("Открыть один")
        self.open_button.clicked.connect(self.openImage)
        
        self.save_button = QPushButton("Сохранить фрагмент")
        self.save_button.clicked.connect(self.saveTemplate)
        
        buttons_layout.addWidget(self.open_button)
        buttons_layout.addWidget(self.save_button)
        right_layout.addLayout(buttons_layout)
        right_layout.addLayout(src_controls_layout)
        
        images_layout.addLayout(left_layout)
        images_layout.addLayout(right_layout)

        main_layout.addLayout(images_layout)
        main_layout.addLayout(QHBoxLayout(),stretch=1)
        main_layout.addStretch()

    def setTemplate(self):
        if self.cv_image is None:
            return
        rect = self.source_label.getSelectionRect()
        if rect is None or rect.width() <= 0 or rect.height() <= 0:
            return

        pixmap = self.source_label._pixmap
        if pixmap is None:
            return

        label_width = self.source_label.width()
        label_height = self.source_label.height()
        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()

        # Вычисляем смещения при центрировании изображения в QLabel
        offset_x = (label_width - pixmap_width) // 2
        offset_y = (label_height - pixmap_height) // 2

        scale_x = self.cv_image.shape[1] / pixmap_width
        scale_y = self.cv_image.shape[0] / pixmap_height

        x = int((rect.x() - offset_x) * scale_x)
        y = int((rect.y() - offset_y) * scale_y)
        w = int(rect.width() * scale_x)
        h = int(rect.height() * scale_y)

        if x < 0: x = 0
        if y < 0: y = 0
        if x + w > self.cv_image.shape[1]:
            w = self.cv_image.shape[1] - x
        if y + h > self.cv_image.shape[0]:
            h = self.cv_image.shape[0] - y

        # Сохраняем выделенную область в self.template (BGR)
        self.template = self.cv_image[y:y+h, x:x+w].copy()

        # Конвертируем в RGB для отображения в QLabel
        try:
            cv_rgb_template = cv2.cvtColor(self.template, cv2.COLOR_BGR2RGB)
        except Exception:
            return
        height_temp, width_temp, channels = cv_rgb_template.shape
        bytes_per_line = 3 * width_temp
        q_image_template = QImage(cv_rgb_template.data, width_temp, height_temp, bytes_per_line, QImage.Format_RGB888)
        template_pixmap = QPixmap.fromImage(q_image_template)
        self.template_label.setPixmap(template_pixmap.scaled(self.template_label.size(), Qt.KeepAspectRatio))

        # Сохранение в файл (добавлено)

    def openImage(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Открыть изображение", "",
                                                "Image Files (*.png *.jpg *.bmp *.ppm)")
        if file_name:
            self.cv_image = cv2.imread(file_name)
            if self.cv_image is None:
                return
            cv_rgb = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
            height, width, channels = cv_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(cv_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(self.source_label.size(), Qt.KeepAspectRatio)
            self.source_label.setPixmap(scaled_pixmap)

    def setPixmapToLabel(self, label, pixmap):
        """
        Если изображение больше размера label, масштабирует с сохранением пропорций.
        Если меньше – отображает оригинальный размер.
        """
        if pixmap.width() > label.width() or pixmap.height() > label.height():
            scaled = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled)
        else:
            label.setPixmap(pixmap)

    def selectSourceDir(self):
        dir_path = QFileDialog().getExistingDirectory(self, "Выбрать папку с изображениями",fr"{home_prefix}\public\faceRecognition\1.original")
        if dir_path:
            patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
            files = []
            for pat in patterns:
                files.extend(glob.glob(os.path.join(dir_path, pat)))
            files.sort()
            self.source_images = files
            if self.source_images:
                self.current_source_index = 0
                self.loadSourceImage()
            else:
                self.source_info_label.setText("Нет изображений")

    def loadSourceImage(self):
        if 0 <= self.current_source_index < len(self.source_images):
            file_path = self.source_images[self.current_source_index]
            self.cv_image = cv2.imread(file_path)
            if self.cv_image is None:
                return
            cv_rgb = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
            height, width, channels = cv_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(cv_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(self.source_label.size(), Qt.KeepAspectRatio)
            self.source_label.setPixmap(scaled_pixmap)
            
            # метод для установки изображения без увеличения, до размеров лейбла
            # self.setPixmapToLabel(self.source_label, pixmap)
            # self.source_info_label.setText(f"({self.current_source_index+1}/{len(self.source_images)})")

    def nextSource(self):
        if self.source_images:
            self.current_source_index = (self.current_source_index + 1) % len(self.source_images)
            self.loadSourceImage()

    def prevSource(self):
        if self.source_images:
            self.current_source_index = (self.current_source_index - 1) % len(self.source_images)
            self.loadSourceImage()

    def saveTemplate(self):
        if not hasattr(self, 'template') or self.template is None:
            return
        
        # Открываем диалог выбора файла
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить шаблон",
            fr"{home_prefix}\public\faceRecognition\3.customTemplatesAndResults",
            "PNG Image (*.png);;PPM Image (*.ppm)"
        )
        
        if filename:
            try:
                cv2.imwrite(filename, self.template)
                print(f"Шаблон сохранен как {filename}")
            except Exception as e:
                print(f"Ошибка сохранения: {e}")
