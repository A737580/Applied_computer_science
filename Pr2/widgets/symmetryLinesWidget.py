import os
import glob
import cv2
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QFileDialog, QFrame, QSizePolicy)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt
from src.env import home_prefix

class SymmetryLinesWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cv_image = None      # Исходное изображение (OpenCV)
        self.resultImage = None
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

        # Верхний виджет: исходное изображение
        source_title = QLabel("Исходное изображение")
        source_title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(source_title)
        self.source_label = QLabel()
        self.source_label.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.source_label.setFixedSize(400, 300)
        self.source_label.setAlignment(Qt.AlignCenter)
        self.source_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.source_label, stretch=1)

        # Нижний виджет: результат
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

        # Правая часть: вверху горизонтальный layout с двумя кнопками
        right_layout = QVBoxLayout()
        right_layout.setSpacing(5)
        right_layout.setContentsMargins(0, 0, 0, 0)

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(5)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.detect_button = QPushButton("Определить линии симметрии")
        self.detect_button.clicked.connect(self.detectSymmetryLines)
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
        buttons_layout.addWidget(self.detect_button)
        right_layout.addLayout(src_controls_layout)
        right_layout.addLayout(buttons_layout)
        
        # Добавляем вертикальный растягивающий отступ, чтобы кнопки оказались вверху
        right_layout.addStretch()

        main_layout.addLayout(right_layout, stretch=1)



    def saveTemplate(self):
        if not hasattr(self, 'resultImage') or self.resultImage is None:
            return
        
        # Открываем диалог выбора файла
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить шаблон",
            fr"{home_prefix}\public\faceRecognition\3.customTemplatesAndResults",
            "PPM Image (*.ppm)"
        )
        
        if filename:
            try:
                cv2.imwrite(filename, self.resultImage)
                print(f"Шаблон сохранен как {filename}")
            except Exception as e:
                print(f"Ошибка сохранения: {e}")


    def detectSymmetryLines(self):
        if self.cv_image is None:
            return

        result_img = self.cv_image.copy()
        gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)

        # Загрузка каскада для лица
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if face_cascade.empty():
            print("Ошибка загрузки каскада для лица")
            return

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            print("Лицо не обнаружено")
            return

        # Берем первое обнаруженное лицо
        (x, y, w, h) = faces[0]
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Центральная линия через центр лица
        center_x = x + w // 2
        cv2.line(result_img, (center_x, y), (center_x, y+h), (0, 0, 255), 2)

        # Детектируем глаза для локальных линий
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        if eye_cascade.empty():
            print("Ошибка загрузки каскада для глаз")
            return

        face_roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=5)
        for (ex, ey, ew, eh) in eyes[:2]:
            eye_center_x = x + ex + ew // 2
            cv2.line(result_img, (eye_center_x, y), (eye_center_x, y+h), (255, 0, 0), 2)

        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        height_res, width_res, channels = result_rgb.shape
        bytes_per_line = 3 * width_res
        q_image_result = QImage(result_rgb.data, width_res, height_res, bytes_per_line, QImage.Format_RGB888)
        result_pixmap = QPixmap.fromImage(q_image_result)
        self.resultImage = result_img.copy()
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
        dir_path = QFileDialog().getExistingDirectory(self, "Выбрать папку с изображениями",fr"{home_prefix}\public\faceRecognition\facesOnly")
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
