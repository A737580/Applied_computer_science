import os
import glob
import cv2
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QFileDialog, QFrame, QSizePolicy)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt
from env import prefix

class ViolaJonesWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cv_image = None      # Исходное изображение (OpenCV)
        self.source_images = []   # Список файлов исходных изображений
        self.current_source_index = -1
        
        # Основной горизонтальный layout
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Левая часть: два больших окна, расположенные вертикально
        left_layout = QVBoxLayout()
        left_layout.setSpacing(5)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Заголовок и исходное изображение
        source_title = QLabel("Исходное изображение")
        source_title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(source_title)
        self.source_label = QLabel()
        self.source_label.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.source_label.setFixedSize(400, 300)
        self.source_label.setAlignment(Qt.AlignCenter)
        self.source_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.source_label, stretch=1)

        # Заголовок и результат
        result_title = QLabel("Результат")
        result_title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(result_title)
        self.result_label = QLabel()
        self.result_label.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.result_label.setFixedSize(400, 300)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.result_label, stretch=1)

        main_layout.addLayout(left_layout, stretch=1)

        # Правая часть: вверху горизонтальный layout с кнопками
        right_layout = QVBoxLayout()
        right_layout.setSpacing(5)
        right_layout.setContentsMargins(0, 0, 0, 0)

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(5)
        buttons_layout.setContentsMargins(0, 0, 0, 0)

        self.detect_button = QPushButton("Детектировать (Viola–Jones)")
        self.detect_button.clicked.connect(self.detectFaces)
        buttons_layout.addWidget(self.detect_button)
        src_controls_layout = QHBoxLayout()
        self.select_source_dir_button = QPushButton("Папка с изображениями")
        self.select_source_dir_button.clicked.connect(self.selectSourceDir)
        src_controls_layout.addWidget(self.select_source_dir_button)
        self.prev_source_button = QPushButton("Предыдущая")
        self.prev_source_button.clicked.connect(self.prevSource)
        self.prev_source_button.clicked.connect(self.detectFaces)
        src_controls_layout.addWidget(self.prev_source_button)
        self.next_source_button = QPushButton("Следующая")
        self.next_source_button.clicked.connect(self.nextSource)
        self.next_source_button.clicked.connect(self.detectFaces)
        src_controls_layout.addWidget(self.next_source_button)
        self.source_info_label = QLabel("Нет изображений")
        src_controls_layout.addWidget(self.source_info_label)
        right_layout.addLayout(src_controls_layout)
        right_layout.addLayout(buttons_layout)
        right_layout.addStretch()  # чтобы кнопки оставались вверху

        main_layout.addLayout(right_layout, stretch=1)

    def detectFaces(self):
        if self.cv_image is None:
            return
        gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            print("Ошибка загрузки каскада:", cascade_path)
            return
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        result_img = self.cv_image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        height, width, channels = result_rgb.shape
        bytes_per_line = 3 * width
        q_image_result = QImage(result_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        result_pixmap = QPixmap.fromImage(q_image_result)
        self.setPixmapToLabel(self.result_label, result_pixmap)

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
        dir_path = QFileDialog().getExistingDirectory(self, "Выбрать папку с изображениями",fr"{prefix}\public\faceRecognition\facesOnly")
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
            self.setPixmapToLabel(self.source_label, pixmap)
            self.source_info_label.setText(f"({self.current_source_index+1}/{len(self.source_images)})")

    def prevSource(self):
        if self.source_images:
            self.current_source_index = (self.current_source_index - 1) % len(self.source_images)
            self.loadSourceImage()

    def nextSource(self):
        if self.source_images:
            self.current_source_index = (self.current_source_index + 1) % len(self.source_images)
            self.loadSourceImage()
