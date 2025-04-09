import os
import glob
import cv2
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QFileDialog, QComboBox, QFrame)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt
from src.env import src_prefix, home_prefix

class TemplateMatchingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cv_image = None      # Исходное изображение (OpenCV)
        self.template = None      # Шаблонное изображение (OpenCV)
        self.resultImage = None
        self.detecredParts = None
        self.source_images = []   # Список файлов исходных изображений
        self.template_images = [] # Список файлов шаблонов
        self.current_source_index = -1
        self.current_template_index = -1
        # Основная вертикальная компоновка
        main_layout = QVBoxLayout(self)

        # Горизонтальная компоновка для исходного изображения и шаблонов
        images_layout = QHBoxLayout()

        # Левая колонка – исходное изображение и управление им
        source_layout = QVBoxLayout()
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
        self.source_label = QLabel()
        self.source_label.setFrameStyle(QFrame.Box | QFrame.Plain)
        # Фиксированный размер для исходного изображения
        self.source_label.setFixedSize(400, 300)
        self.source_label.setAlignment(Qt.AlignCenter)
        source_layout.addWidget(self.source_label)
        images_layout.addLayout(source_layout)

        # Правая колонка – шаблоны: управление и превью шаблона
        template_layout = QVBoxLayout()
        tmpl_controls_layout = QHBoxLayout()
        self.select_template_dir_button = QPushButton("Папка с шаблонами")
        self.select_template_dir_button.clicked.connect(self.selectTemplateDir)
        tmpl_controls_layout.addWidget(self.select_template_dir_button)
        self.prev_template_button = QPushButton("Предыдущий шаблон")
        self.prev_template_button.clicked.connect(self.prevTemplate)
        tmpl_controls_layout.addWidget(self.prev_template_button)
        self.next_template_button = QPushButton("Следующий шаблон")
        self.next_template_button.clicked.connect(self.nextTemplate)
        tmpl_controls_layout.addWidget(self.next_template_button)
        self.template_info_label = QLabel("Нет шаблонов")
        tmpl_controls_layout.addWidget(self.template_info_label)
        self.template_label = QLabel()
        self.template_label.setFrameStyle(QFrame.Box | QFrame.Plain)
        # Фиксированный размер для шаблона
        self.template_label.setFixedSize(150, 150)
        self.template_label.setAlignment(Qt.AlignCenter)
        template_layout.addWidget(self.template_label)
        template_layout2 = QVBoxLayout()
        template_layout2.addWidget(QLabel(),stretch=1)
        template_layout.addLayout(template_layout2)
        template_layout.addLayout(tmpl_controls_layout)
        template_layout.addLayout(src_controls_layout)
        images_layout.addLayout(template_layout)

        main_layout.addLayout(images_layout)

        # Компоновка для выбора метода и кнопки запуска
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Метод:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "TM_CCOEFF", "TM_CCOEFF_NORMED",
            "TM_CCORR", "TM_CCORR_NORMED",
            "TM_SQDIFF", "TM_SQDIFF_NORMED"
        ])
        controls_layout.addWidget(self.method_combo)
        self.match_button = QPushButton("Выполнить Template Matching")
        self.match_button.clicked.connect(self.matchTemplate)
        controls_layout.addWidget(self.match_button)
        self.detect_parts_button = QPushButton("Детектировать части лица")
        self.detect_parts_button.clicked.connect(self.detectFaceParts)
        controls_layout.addWidget(self.detect_parts_button)
        main_layout.addLayout(controls_layout)

        
        
        # Отображение результата
        labels_layout = QHBoxLayout()
        labels_layout.addWidget(QLabel("Результат"))
        labels_layout.addWidget(QLabel("Нормализованная матрица совпадений | детекция частей лица"))
        main_layout.addLayout(labels_layout)
        self.result_label = QLabel()
        self.result_label.setFrameStyle(QFrame.Box | QFrame.Plain)
        # Фиксированный размер для результата
        self.result_label.setFixedSize(400, 300)
        self.result_label.setAlignment(Qt.AlignCenter)
        
        result_layout = QHBoxLayout()
        self.mathing_result_label = QLabel()
        self.mathing_result_label.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.mathing_result_label.setFixedSize(400, 300)
        self.mathing_result_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.mathing_result_label)
        main_layout.addLayout(result_layout)

    def saveDetectedParts(self):
        if not hasattr(self, 'detecredParts') or self.detecredParts is None:
            return
        
        # Открываем диалог выбора файла
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить результат",
            fr"{home_prefix}\public\faceRecognition\3.customTemplatesAndResults",
            "PPM Image (*.ppm)"
        )
        
        if filename:
            try:
                cv2.imwrite(filename, self.detecredParts)
                print(f"Шаблон сохранен как {filename}")
            except Exception as e:
                print(f"Ошибка сохранения: {e}")


    def saveTemplate(self):
        if not hasattr(self, 'resultImage') or self.resultImage is None:
            return
        
        # Открываем диалог выбора файла
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить шаблон",
            fr"{home_prefix}\public\faceRecognition\customTemplates",
            "PPM Image (*.ppm)"
        )
        
        if filename:
            try:
                cv2.imwrite(filename, self.resultImage)
                print(f"Шаблон сохранен как {filename}")
            except Exception as e:
                print(f"Ошибка сохранения: {e}")

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

    # Методы для исходных изображений
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
            self.setPixmapToLabel(self.source_label, pixmap)
            self.source_info_label.setText(f"({self.current_source_index+1}/{len(self.source_images)})")
            # {os.path.basename(file_path)} 
            # self.result_label.clear()

    def nextSource(self):
        if self.source_images:
            self.current_source_index = (self.current_source_index + 1) % len(self.source_images)
            self.loadSourceImage()

    def prevSource(self):
        if self.source_images:
            self.current_source_index = (self.current_source_index - 1) % len(self.source_images)
            self.loadSourceImage()

    # Методы для шаблонных изображений
    def selectTemplateDir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Выбрать папку с шаблонами",fr"{home_prefix}\public\faceRecognition\templates")
        if dir_path:
            patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp","*.ppm"]
            files = []
            for pat in patterns:
                files.extend(glob.glob(os.path.join(dir_path, pat)))
            files.sort()
            self.template_images = files
            if self.template_images:
                self.current_template_index = 0
                self.loadTemplateImage()
            else:
                self.template_info_label.setText("Нет шаблонов")

    def loadTemplateImage(self):
        if 0 <= self.current_template_index < len(self.template_images):
            file_path = self.template_images[self.current_template_index]
            self.template = cv2.imread(file_path)
            if self.template is None:
                return
            cv_rgb = cv2.cvtColor(self.template, cv2.COLOR_BGR2RGB)
            height, width, channels = cv_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(cv_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.setPixmapToLabel(self.template_label, pixmap)
            self.template_info_label.setText(f"({self.current_template_index+1}/{len(self.template_images)})")
            # {os.path.basename(file_path)} 
            
    def nextTemplate(self):
        if self.template_images:
            self.current_template_index = (self.current_template_index + 1) % len(self.template_images)
            self.loadTemplateImage()

    def prevTemplate(self):
        if self.template_images:
            self.current_template_index = (self.current_template_index - 1) % len(self.template_images)
            self.loadTemplateImage()

    def matchTemplate(self):
        if self.cv_image is None or self.template is None:
            return

        method_str = self.method_combo.currentText()
        method = getattr(cv2, method_str)
        # Получаем матрицу совпадений
        res = cv2.matchTemplate(self.cv_image, self.template, method)
        
        # Нормализуем результат для отображения (0-255)
        norm_res = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX)
        norm_res = norm_res.astype('uint8')
        # Преобразуем нормализованную матрицу в QImage (формат Grayscale)
        q_image_map = QImage(norm_res.data, norm_res.shape[1], norm_res.shape[0],
                            norm_res.strides[0], QImage.Format_Grayscale8)
        map_pixmap = QPixmap.fromImage(q_image_map)
        # Устанавливаем матрицу совпадений в mathing_result_label
        self.setPixmapToLabel(self.mathing_result_label, map_pixmap)

        # Далее ищем оптимальное совпадение
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        h_temp, w_temp = self.template.shape[:2]
        bottom_right = (top_left[0] + w_temp, top_left[1] + h_temp)

        result_img = self.cv_image.copy()
        cv2.rectangle(result_img, top_left, bottom_right, (0, 0, 255), 2)
        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        height_res, width_res, channels = result_rgb.shape
        bytes_per_line = 3 * width_res
        q_image_result = QImage(result_rgb.data, width_res, height_res, bytes_per_line, QImage.Format_RGB888)
        result_pixmap = QPixmap.fromImage(q_image_result)
        self.resultImage = result_img.copy()
        self.setPixmapToLabel(self.result_label, result_pixmap)

    def detectFaceParts(self):
        if self.cv_image is None:
            return
        result_img = self.cv_image.copy()
        gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)

        # Детекция лица
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if not face_cascade.empty():
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Лицо – зелёный

        # Детекция глаз
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        if not eye_cascade.empty():
            eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in eyes:
                cv2.rectangle(result_img, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Глаза – синий

        # Детекция улыбки
        smile_cascade = cv2.CascadeClassifier(fr"{src_prefix}\haarcascade_smile.xml")
        if not smile_cascade.empty():
            smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=20, flags=cv2.CASCADE_SCALE_IMAGE)
            for (x, y, w, h) in smiles:
                cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 165, 255), 2)  # Улыбка – оранжевый
        
        # Детекция носа
        nose_cascade = cv2.CascadeClassifier(fr"{src_prefix}\haarcascade_nose.xml")
        if not nose_cascade.empty():
            smiles = nose_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=8, flags=cv2.CASCADE_SCALE_IMAGE)
            for (x, y, w, h) in smiles:
                cv2.rectangle(result_img, (x, y), (x+w, y+h), (123, 104, 238), 2)  # Нос – сине-фиолетовый
        
        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        height_res, width_res, channels = result_rgb.shape
        bytes_per_line = 3 * width_res
        q_image_result = QImage(result_rgb.data, width_res, height_res, bytes_per_line, QImage.Format_RGB888)
        result_pixmap = QPixmap.fromImage(q_image_result)
        self.detecredParts = result_img.copy()
        self.setPixmapToLabel(self.mathing_result_label, result_pixmap)
