import logging
from pathlib import Path

import numpy as np
import cv2

# GUI
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QLabel,
    QMessageBox,
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, Slot

# Наши модули
import src.feature_extractor as feature_extractor
import src.model as model_utils 


class TestModelWidget(QWidget):
    def __init__(self, shared_state):
        super().__init__()
        self.shared_state = shared_state
        self.init_ui()
    
        # --- init_inference_ui ---
    def init_ui(self):
        """Инициализирует UI вкладки 'Инференс (Тест)' для работы с папками."""
        layout = QVBoxLayout(self)

        # --- Верхний блок: Загрузка модели ---
        model_load_layout = QHBoxLayout()
        self.btn_load_inference_model = QPushButton("Загрузить модель для инференса (.joblib)")
        self.btn_load_inference_model.clicked.connect(self.load_inference_model) # Старый слот остается
        self.lbl_inference_model_status = QLabel("Модель не загружена")
        model_load_layout.addWidget(self.btn_load_inference_model)
        model_load_layout.addWidget(self.lbl_inference_model_status, 1)
        layout.addLayout(model_load_layout)

        # --- Блок выбора директории ---
        dir_select_layout = QHBoxLayout()
        self.btn_select_inference_dir = QPushButton("Выбрать директорию с изображениями")
        self.btn_select_inference_dir.clicked.connect(self.select_inference_dir) 
        self.lbl_inference_dir_status = QLabel("Папка не выбрана")
        dir_select_layout.addWidget(self.btn_select_inference_dir)
        dir_select_layout.addWidget(self.lbl_inference_dir_status, 1)
        layout.addLayout(dir_select_layout)

        # --- Блок навигации ---
        navigation_layout = QHBoxLayout()
        self.btn_prev_image = QPushButton("Предыдущее")
        self.btn_prev_image.setEnabled(False)
        self.btn_prev_image.clicked.connect(self.show_prev_image)  

        self.lbl_current_image_info = QLabel("Файл: N/A (0/0)")
        self.lbl_current_image_info.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.btn_next_image = QPushButton("Следующее")
        self.btn_next_image.setEnabled(False)
        self.btn_next_image.clicked.connect(self.show_next_image)  

        navigation_layout.addWidget(self.btn_prev_image)
        navigation_layout.addWidget(self.btn_next_image)
        navigation_layout.addWidget(self.lbl_current_image_info, 1) 
        layout.addLayout(navigation_layout)

        # --- Кнопка предсказания ---
        self.btn_run_prediction = QPushButton("Проверить предсказание")
        self.btn_run_prediction.setEnabled(False)
        self.btn_run_prediction.clicked.connect(self.run_prediction_for_current_image)  
        layout.addWidget(self.btn_run_prediction)

        # --- Отображение результата (как и раньше) ---
        results_display_layout = QHBoxLayout()
        # Место для картинки
        self.lbl_inference_image = QLabel("Выберите папку для начала")
        self.lbl_inference_image.setMinimumSize(300, 300) # Можно настроить размер
        self.lbl_inference_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_inference_image.setStyleSheet("border: 1px solid gray; color: gray;")
        # Место для предсказания
        self.lbl_inference_prediction = QLabel("Предсказанный автор: ?")
        font = self.lbl_inference_prediction.font(); font.setPointSize(14); self.lbl_inference_prediction.setFont(font); self.lbl_inference_prediction.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop) # Выравниваем текст сверху

        results_display_layout.addWidget(self.lbl_inference_image, 1) # Даем картинке растягиваться
        results_display_layout.addWidget(self.lbl_inference_prediction, 1) # И предсказанию

        layout.addLayout(results_display_layout)
        layout.addStretch() # Добавляем растяжение в конец
    
    
        # --- Методы для вкладки "Инференс" ---
    @Slot()
    def load_inference_model(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Загрузить модель для инференса", "", "Joblib Files (*.joblib)"
        )
        if filepath:
            try:
                # Загружаем ТОЛЬКО модель и имена классов для инференса
                model, names, _ = model_utils.load_model(filepath)
                if not isinstance(model, model_utils.Pipeline):
                    raise TypeError("Model is not Pipeline")
                if not isinstance(names, list):
                    raise TypeError("Class names not a list")

                self.shared_state.loaded_inference_model_path = filepath
                self.shared_state.loaded_inference_model = model
                self.shared_state.loaded_inference_class_names = names

                self.lbl_inference_model_status.setText(
                    f"Загружено: {Path(filepath).name} ({len(names)} кл.)"
                )
                logging.info(f"Модель для инференса загружена: {filepath}")
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Ошибка загрузки модели",
                    f"Не удалось загрузить модель для инференса:\n{e}",
                )
                self.shared_state.loaded_inference_model = None
                self.shared_state.loaded_inference_class_names = None
                self.lbl_inference_model_status.setText("Ошибка загрузки")

    @Slot()
    def run_inference_on_image(self):
        if not self.shared_state.loaded_inference_model:
            return
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение для теста",
            "",
            "Image Files (*.png *.jpg *.jpeg)",
        )
        if filepath:
            try:
                img = cv2.imread(filepath)
                if img is None:
                    raise ValueError("Не удалось загрузить изображение.")
                img_size = self.shared_state.current_img_size

                img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                img_normalized = img_rgb.astype(np.float32) / 255.0

                q_img = QImage(
                    img_rgb.data,
                    img_rgb.shape[1],
                    img_rgb.shape[0],
                    img_rgb.strides[0],
                    QImage.Format_RGB888,
                )
                pixmap = QPixmap.fromImage(q_img)
                self.lbl_inference_image.setPixmap(
                    pixmap.scaled(
                        self.lbl_inference_image.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )

                self.lbl_inference_prediction.setText("Извлечение признаков...")
                QApplication.processEvents()
                features = feature_extractor.extract_features(
                    np.array([img_normalized])
                )

                self.lbl_inference_prediction.setText("Предсказание...")
                QApplication.processEvents()
                pred_idx = self.shared_state.loaded_inference_model.predict(features)[0]
                pred_author = self.shared_state.loaded_inference_class_names[pred_idx]
                try:
                    probs = self.shared_state.loaded_inference_model.predict_proba(features)[0]
                    conf = probs[pred_idx]
                    pred_text = f"Предсказанный автор:\n{pred_author}\n(Уверенность: {conf:.2f})"
                except Exception:
                    pred_text = f"Предсказанный автор:\n{pred_author}"
                self.lbl_inference_prediction.setText(pred_text)
                logging.info(f"Инференс для {filepath}: Предсказано '{pred_author}'")
            except Exception as e:
                QMessageBox.critical(
                    self, "Ошибка инференса", f"Произошла ошибка:\n{e}"
                )
                logging.error(f"Ошибка инференса: {e}", exc_info=True)
                self.lbl_inference_image.setText("Ошибка обработки")
                self.lbl_inference_prediction.setText("Предсказание не удалось")

        # --- Метод для кнопки "Выбрать изображение для теста" ---
    
    @Slot()
    def select_inference_dir(self):
        """Слот для кнопки выбора директории с изображениями для инференса."""
        dir_path = QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями")
        if dir_path:
            self.shared_state.inference_dir = Path(dir_path)
            self.lbl_inference_dir_status.setText(f"Папка: ...{str(self.shared_state.inference_dir)[-50:]}")
            logging.info(f"Выбрана папка для инференса: {self.shared_state.inference_dir}")

            # Ищем файлы изображений
            valid_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
            self.shared_state.inference_image_files = []
            for ext in valid_extensions:
                self.shared_state.inference_image_files.extend(sorted(self.shared_state.inference_dir.glob(ext)))

            if not self.shared_state.inference_image_files:
                self.shared_state.current_inference_image_index = -1
                QMessageBox.warning(self, "Нет изображений", f"В папке {self.shared_state.inference_dir} не найдено поддерживаемых изображений ({', '.join(valid_extensions)}).")
                self.lbl_current_image_info.setText("Файл: N/A (0/0)")
                self.lbl_inference_image.setText("Изображения не найдены")
                self.lbl_inference_image.setStyleSheet("border: 1px solid gray; color: red;")
                self.btn_prev_image.setEnabled(False)
                self.btn_next_image.setEnabled(False)
                self.btn_run_prediction.setEnabled(False)
            else:
                self.shared_state.current_inference_image_index = 0 # Начинаем с первого
                self.display_current_inference_image() # Отображаем первое изображение

        else:
            logging.warning("Выбор папки для инференса отменен.")

    def display_current_inference_image(self):
        """Отображает текущее изображение из выбранной папки и обновляет UI."""
        if 0 <= self.shared_state.current_inference_image_index < len(self.shared_state.inference_image_files):
            filepath = self.shared_state.inference_image_files[self.shared_state.current_inference_image_index]
            total_files = len(self.shared_state.inference_image_files)
            current_num = self.shared_state.current_inference_image_index + 1

            self.lbl_current_image_info.setText(f"Файл: {filepath.name} ({current_num}/{total_files})")
            self.lbl_current_image_info.setToolTip(str(filepath)) # Полный путь во всплывающей подсказке

            try:
                pixmap = QPixmap(str(filepath))
                if pixmap.isNull(): raise ValueError("Не удалось загрузить QPixmap")

                # Масштабируем для отображения
                scaled_pixmap = pixmap.scaled(self.lbl_inference_image.size(),
                                              Qt.AspectRatioMode.KeepAspectRatio,
                                              Qt.TransformationMode.SmoothTransformation)
                self.lbl_inference_image.setPixmap(scaled_pixmap)
                self.lbl_inference_image.setStyleSheet("border: 1px solid gray;") # Убираем красный цвет, если был

                # Очищаем предыдущее предсказание
                self.lbl_inference_prediction.setText("Предсказанный автор: ?")

                # Обновляем состояние кнопок навигации и предсказания
                self.btn_prev_image.setEnabled(self.shared_state.current_inference_image_index > 0)
                self.btn_next_image.setEnabled(self.shared_state.current_inference_image_index < total_files - 1)
                # Кнопка предсказания активна, если загружена модель
                self.btn_run_prediction.setEnabled(self.shared_state.loaded_inference_model is not None)

            except Exception as e:
                logging.error(f"Ошибка загрузки изображения {filepath}: {e}", exc_info=True)
                self.lbl_inference_image.setText(f"Ошибка загрузки:\n{filepath.name}")
                self.lbl_inference_image.setStyleSheet("border: 1px solid red; color: red;")
                self.lbl_inference_prediction.setText("Предсказание невозможно")
                self.btn_run_prediction.setEnabled(False)
        else:
            self.lbl_current_image_info.setText("Файл: N/A (0/0)")
            self.lbl_inference_image.setText("Изображение не выбрано")
            self.lbl_inference_image.setStyleSheet("border: 1px solid gray; color: gray;")
            self.btn_prev_image.setEnabled(False)
            self.btn_next_image.setEnabled(False)
            self.btn_run_prediction.setEnabled(False)

    @Slot()
    def show_next_image(self):
        """Переключается на следующее изображение в папке."""
        if self.shared_state.current_inference_image_index < len(self.shared_state.inference_image_files) - 1:
            self.shared_state.current_inference_image_index += 1
            self.display_current_inference_image()

    @Slot()
    def show_prev_image(self):
        """Переключается на предыдущее изображение в папке."""
        if self.shared_state.current_inference_image_index > 0:
            self.shared_state.current_inference_image_index -= 1
            self.display_current_inference_image()

    @Slot()
    def run_prediction_for_current_image(self):
        """Запускает инференс для текущего отображаемого изображения."""
        # Проверяем все условия
        if self.shared_state.loaded_inference_model is None:
            QMessageBox.warning(self, "Модель не загружена", "Сначала загрузите модель.")
            return
        if not (0 <= self.shared_state.current_inference_image_index < len(self.shared_state.inference_image_files)):
            QMessageBox.warning(self, "Изображение не выбрано", "Нет текущего изображения для предсказания.")
            return

        filepath = self.shared_state.inference_image_files[self.shared_state.current_inference_image_index]
        self.lbl_inference_prediction.setText("Обработка...")
        QApplication.processEvents() # Даем UI обновиться

        try:
            # 1. Загрузка и предобработка (повторяется, можно оптимизировать)
            img = cv2.imread(str(filepath))
            if img is None: raise ValueError(f"Не удалось загрузить изображение: {filepath}")

            # Используем размер по умолчанию или из параметров модели, если они были бы сохранены
            img_size = self.shared_state.current_img_size

            img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_normalized = img_rgb.astype(np.float32) / 255.0

            # 2. Извлечение признаков
            self.lbl_inference_prediction.setText("Извлечение признаков...")
            QApplication.processEvents()
            features = feature_extractor.extract_features(np.array([img_normalized]))

            # 3. Предсказание
            self.lbl_inference_prediction.setText("Предсказание...")
            QApplication.processEvents()
            pred_idx = self.shared_state.loaded_inference_model.predict(features)[0]

            # 4. Отображение результата
            if 0 <= pred_idx < len(self.shared_state.loaded_inference_class_names):
                pred_author = self.shared_state.loaded_inference_class_names[pred_idx]
                try:
                    probabilities = self.shared_state.loaded_inference_model.predict_proba(features)[0]
                    confidence = probabilities[pred_idx]
                    pred_text = f"Предсказанный автор:\n{pred_author}\n(Уверенность: {confidence:.2f})"
                except Exception:
                    pred_text = f"Предсказанный автор:\n{pred_author}"
            else:
                pred_text = f"Ошибка: Индекс ({pred_idx}) вне диапазона ({len(self.shared_state.loaded_inference_class_names)})"
                logging.error(pred_text)

            self.lbl_inference_prediction.setText(pred_text)
            logging.info(f"Инференс для {filepath}: Предсказано '{pred_author}'")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка инференса", f"Произошла ошибка:\n{e}")
            logging.error(f"Ошибка инференса для {filepath}: {e}", exc_info=True)
            self.lbl_inference_prediction.setText("Предсказание не удалось")
    
    @Slot()
    def run_inference_on_image(self):
        if not self.shared_state.loaded_inference_model:
            QMessageBox.warning(
                self,
                "Модель не загружена",
                "Сначала загрузите модель на вкладке 'Инференс (Тест)'.",
            )
            return

        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение для теста",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)",
        )
        if filepath:
            try:
                # 1. Загрузка и предобработка изображения
                img = cv2.imread(filepath)
                if img is None:
                    raise ValueError(f"Не удалось загрузить изображение: {filepath}")

                img_size = self.shared_state.current_img_size

                img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                img_normalized = img_rgb.astype(np.float32) / 255.0

                # Отображаем загруженное изображение
                h, w, ch = img_rgb.shape
                bytes_per_line = ch * w
                q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                self.lbl_inference_image.setPixmap(
                    pixmap.scaled(
                        self.lbl_inference_image.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )

                # 2. Извлечение признаков
                self.lbl_inference_prediction.setText("Извлечение признаков...")
                QApplication.processEvents()  # Обновить UI
                features = feature_extractor.extract_features(
                    np.array([img_normalized])
                )

                # 3. Предсказание
                self.lbl_inference_prediction.setText("Предсказание...")
                QApplication.processEvents()

                pred_idx = self.shared_state.loaded_inference_model.predict(features)[0]
                if 0 <= pred_idx < len(self.shared_state.loaded_inference_class_names):
                    pred_author = self.shared_state.loaded_inference_class_names[pred_idx]
                    try:
                        probabilities = self.shared_state.loaded_inference_model.predict_proba(
                            features
                        )[0]
                        confidence = probabilities[
                            pred_idx
                        ]  # Берем вероятность предсказанного класса
                        pred_text = f"Предсказанный автор:\n{pred_author}\n(Уверенность: {confidence:.2f})"
                    except Exception:
                        pred_text = f"Предсказанный автор:\n{pred_author}"
                else:
                    pred_text = f"Ошибка: Индекс предсказания ({pred_idx}) вне диапазона классов ({len(self.shared_state.loaded_inference_class_names)})."
                    logging.error(pred_text)

                self.lbl_inference_prediction.setText(pred_text)
                logging.info(f"Инференс для {filepath}: Предсказано '{pred_author}'")

            except Exception as e:
                QMessageBox.critical(
                    self, "Ошибка инференса", f"Произошла ошибка:\n{e}"
                )
                logging.error(f"Ошибка инференса для {filepath}: {e}", exc_info=True)
                self.lbl_inference_image.setText("Ошибка обработки")
                self.lbl_inference_prediction.setText("Предсказание не удалось")
