import sys
import logging
import random
from pathlib import Path
from datetime import datetime
import traceback

import numpy as np
import cv2

# GUI
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QLabel,
    QListWidget,
    QTabWidget,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
    QSizePolicy,
    QTextEdit,
    QFrame,  # Добавлен QFrame для разделителя
)
from PySide6.QtGui import QPixmap, QImage, QColor
from PySide6.QtCore import Qt, Signal, Slot, QThread, QObject

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

# Sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Наши модули
import data_loader
import feature_extractor
import model as model_utils
import env

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- Worker class ---
class Worker(QObject):
    finished = Signal()
    progress = Signal(int)
    log_message = Signal(str)
    results = Signal(object)
    error = Signal(str)

    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self._is_running = True

    @Slot()
    def run(self):
        try:
            if "progress_callback" in self.function.__code__.co_varnames:
                self.kwargs["progress_callback"] = self.progress.emit
            if "log_callback" in self.function.__code__.co_varnames:
                self.kwargs["log_callback"] = self.log_message.emit

            result = self.function(*self.args, **self.kwargs)
            if self._is_running and result is not None:
                self.results.emit(result)
        except Exception as e:
            logging.error(f"Error in worker thread: {e}", exc_info=True)
            if self._is_running:
                self.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        finally:
            if self._is_running:
                self.finished.emit()

    def stop(self):
        self._is_running = False


# --- Основное окно приложения ---
class PainterAttributionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Атрибуция картин по автору")  # Версия
        self.setGeometry(100, 100, 1150, 800)  # Еще немного увеличим

        # --- Переменные состояния ---
        self.root_dir = None
        self.selected_genre = None
        self.current_img_size = (128, 128)  # Размер изображения по умолчанию
        self.current_test_size = 0.25
        self.current_random_state = 42

        self.class_names = []
        self.class_to_idx = {}

        # Результаты последнего УСПЕШНОГО обучения или оценки
        self.last_run_results = {}
        self.trained_model = None

        self.loaded_inference_model_path = None
        self.loaded_inference_model = None
        self.loaded_inference_class_names = None

        # --- Создание вкладок ---
        self.tabs = QTabWidget()
        self.tab_dataset = QWidget()
        self.tab_training_results = QWidget()
        self.tab_training_plots = QWidget()
        self.tab_3d_vis = QWidget()
        self.tab_inference = QWidget()

        self.tabs.addTab(self.tab_dataset, "1. Датасет")
        self.tabs.addTab(self.tab_training_results, "2. Обучение и Результаты")
        self.tabs.addTab(self.tab_training_plots, "3. Графики Обучения")
        self.tabs.addTab(self.tab_3d_vis, "4. 3D Визуализация")
        self.tabs.addTab(self.tab_inference, "5. Инференс (Тест)")

        # --- Инициализация UI ---
        self.init_dataset_ui()
        self.init_training_results_ui()
        self.init_training_plots_ui()
        self.init_3d_vis_ui()
        self.init_inference_ui()

        self.setCentralWidget(self.tabs)
        self.thread = None
        self.worker = None

    # --- init_dataset_ui ---
    def init_dataset_ui(self):
        layout = QVBoxLayout(self.tab_dataset)
        # Код идентичен предыдущей версии...
        dir_layout = QHBoxLayout()
        self.btn_load_dir = QPushButton("Выбрать корневую папку датасета")
        self.btn_load_dir.clicked.connect(self.select_root_directory)
        self.lbl_root_dir = QLabel("Корневая папка не выбрана")
        dir_layout.addWidget(self.btn_load_dir)
        dir_layout.addWidget(self.lbl_root_dir, 1)
        layout.addLayout(dir_layout)
        genre_layout = QHBoxLayout()
        self.btn_load_genre = QPushButton("Загрузить жанр")
        self.btn_load_genre.setEnabled(False)
        self.btn_load_genre.clicked.connect(self.load_genre_data)
        self.lbl_selected_genre = QLabel("Жанр не загружен")
        self.list_authors = QListWidget()
        self.list_authors.setMaximumHeight(150)
        genre_layout.addWidget(self.btn_load_genre)
        genre_layout.addWidget(self.lbl_selected_genre, 1)
        layout.addLayout(genre_layout)
        layout.addWidget(QLabel("Обнаруженные авторы:"))
        layout.addWidget(self.list_authors)
        sample_layout = QHBoxLayout()
        self.btn_show_samples = QPushButton("Показать примеры автора")
        self.btn_show_samples.setEnabled(False)
        self.btn_show_samples.clicked.connect(self.show_author_samples)
        self.list_authors.currentItemChanged.connect(
            lambda: self.btn_show_samples.setEnabled(
                True if self.list_authors.currentItem() else False
            )
        )
        sample_layout.addWidget(self.btn_show_samples)
        layout.addLayout(sample_layout)
        self.sample_images_layout = QHBoxLayout()
        layout.addLayout(self.sample_images_layout)
        layout.addStretch()

    # --- ИНИЦИАЛИЗАЦИЯ ВКЛАДКИ "ОБУЧЕНИЕ И РЕЗУЛЬТАТЫ" ---

    def init_training_results_ui(self):
        main_layout = QVBoxLayout(self.tab_training_results)

        # Верхняя часть: Кнопки управления и Статус/Прогресс
        top_layout = QHBoxLayout()

        # Кнопки слева
        self.btn_train = QPushButton("Начать обучение")
        self.btn_train.setEnabled(False)
        self.btn_train.clicked.connect(self.start_training_process)

        self.btn_save_model = QPushButton("Сохранить текущую модель")
        self.btn_save_model.setEnabled(False)
        self.btn_save_model.clicked.connect(self.save_trained_model)

        top_layout.addWidget(self.btn_train)
        top_layout.addWidget(self.btn_save_model)
        top_layout.addStretch()

        # Статус и прогресс-бар справа (вертикально)
        status_progress_layout = QVBoxLayout()
        self.lbl_train_status = QLabel("Статус: Ожидание данных")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumWidth(250)
        self.lbl_train_status.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        status_progress_layout.addWidget(self.lbl_train_status)
        status_progress_layout.addWidget(self.progress_bar)

        top_layout.addLayout(status_progress_layout)

        main_layout.addLayout(top_layout)

        # Разделитель
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(line)

        # Средняя часть: Матрица ошибок и Отчет классификации
        results_display_layout = QHBoxLayout()
        cm_layout = QVBoxLayout()
        cm_layout.addWidget(QLabel("Матрица ошибок (Confusion Matrix)"))
        self.table_confusion_matrix = QTableWidget()
        self.table_confusion_matrix.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        cm_layout.addWidget(self.table_confusion_matrix)
        results_display_layout.addLayout(cm_layout, 1)
        report_layout = QVBoxLayout()
        report_layout.addWidget(
            QLabel("Отчет по классификации (Classification Report)")
        )
        self.report_display = QTextEdit()
        self.report_display.setReadOnly(True)
        self.report_display.setFontFamily("Courier New")
        self.report_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        report_layout.addWidget(self.report_display)
        results_display_layout.addLayout(report_layout, 1)
        main_layout.addLayout(results_display_layout)

        # Нижняя часть: График уверенности (Box Plot)
        confidence_layout = QVBoxLayout()
        
        confidence_layout.addWidget(
            QLabel(
                "График уверенности модели (Вероятности для истинных классов на тесте)"
            )
        )
        self.fig_confidence, self.ax_confidence = plt.subplots()
        self.canvas_confidence = FigureCanvas(self.fig_confidence)
        self.toolbar_confidence = NavigationToolbar(self.canvas_confidence, self)
        confidence_layout.addWidget(self.toolbar_confidence)
        confidence_layout.addWidget(self.canvas_confidence)
        
        
        ###  Скрытие Box Plot  ###
        for i in range(confidence_layout.count()): #
            widget = confidence_layout.itemAt(i).widget() #
            if widget: # 
                widget.hide() #
                
                
        main_layout.addLayout(confidence_layout)

    # --- ИНИЦИАЛИЗАЦИЯ ВКЛАДКИ "ГРАФИКИ ОБУЧЕНИЯ" ---
    def init_training_plots_ui(self):
        main_layout = QVBoxLayout(self.tab_training_plots)

        # Верхняя часть: График Loss/Validation Score
        metrics_layout = QVBoxLayout()
        metrics_layout.addWidget(QLabel("График обучения (Loss / Validation Score)"))
        self.fig_metrics, self.ax_metrics = plt.subplots()
        self.canvas_metrics = FigureCanvas(self.fig_metrics)
        self.toolbar_metrics = NavigationToolbar(self.canvas_metrics, self)
        metrics_layout.addWidget(self.toolbar_metrics)
        metrics_layout.addWidget(self.canvas_metrics)
        main_layout.addLayout(metrics_layout)

        # Нижняя часть: Компактная информация об обучении
        info_layout = QHBoxLayout()
        self.lbl_final_iter_label = QLabel("Итераций до остановки:")
        self.lbl_final_iter = QLabel("N/A")
        self.lbl_final_loss_label = QLabel("Финальное значение Loss:")
        self.lbl_final_loss = QLabel("N/A")

        info_layout.addWidget(self.lbl_final_iter_label)
        info_layout.addWidget(self.lbl_final_iter)
        info_layout.addStretch()
        info_layout.addWidget(self.lbl_final_loss_label)
        info_layout.addWidget(self.lbl_final_loss)
        info_layout.addStretch()

        self.lbl_final_iter_label.setVisible(False)
        self.lbl_final_iter.setVisible(False)
        self.lbl_final_loss_label.setVisible(False)
        self.lbl_final_loss.setVisible(False)

        main_layout.addLayout(info_layout)
        main_layout.addStretch()

    # --- ИНИЦИАЛИЗАЦИЯ ВКЛАДКИ "3D ВИЗУАЛИЗАЦИЯ" ---
    def init_3d_vis_ui(self):
        main_layout = QVBoxLayout(self.tab_3d_vis)
        main_layout.addWidget(QLabel("3D Проекция признаков обучающей выборки (PCA)"))

        self.fig_projection = plt.figure(figsize=(8, 8))
        self.ax_projection = self.fig_projection.add_subplot(111, projection="3d")
        self.canvas_projection = FigureCanvas(self.fig_projection)
        self.toolbar_projection = NavigationToolbar(self.canvas_projection, self)

        main_layout.addWidget(self.toolbar_projection)
        main_layout.addWidget(self.canvas_projection)

    # --- init_inference_ui ---
    def init_inference_ui(self):
        layout = QVBoxLayout(self.tab_inference)
        model_load_layout = QHBoxLayout()
        self.btn_load_inference_model = QPushButton(
            "Загрузить модель для инференса (.joblib)"
        )
        self.btn_load_inference_model.clicked.connect(self.load_inference_model)
        self.lbl_inference_model_status = QLabel("Модель не загружена")
        model_load_layout.addWidget(self.btn_load_inference_model)
        model_load_layout.addWidget(self.lbl_inference_model_status, 1)
        layout.addLayout(model_load_layout)
        test_layout = QHBoxLayout()
        self.btn_test_image = QPushButton("Выбрать изображение для теста")
        self.btn_test_image.setEnabled(False)
        self.btn_test_image.clicked.connect(self.run_inference_on_image)
        test_layout.addWidget(self.btn_test_image)
        layout.addLayout(test_layout)
        results_layout = QHBoxLayout()
        self.lbl_inference_image = QLabel("Здесь будет тестовое изображение")
        self.lbl_inference_image.setMinimumSize(256, 256)
        self.lbl_inference_image.setAlignment(Qt.AlignCenter)
        self.lbl_inference_image.setStyleSheet("border: 1px solid gray;")
        self.lbl_inference_prediction = QLabel("Предсказанный автор: ?")
        font = self.lbl_inference_prediction.font()
        font.setPointSize(14)
        self.lbl_inference_prediction.setFont(font)
        self.lbl_inference_prediction.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.lbl_inference_image)
        results_layout.addWidget(self.lbl_inference_prediction, 1)
        layout.addLayout(results_layout)
        layout.addStretch()

    # --- Методы для вкладки "Датасет" ---
    @Slot()
    def select_root_directory(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Выберите корневую папку датасета"
        )
        if dir_path:
            self.root_dir = Path(dir_path)
            self.lbl_root_dir.setText(f"Выбрано: ...{str(self.root_dir)[-40:]}")
            self.btn_load_genre.setEnabled(True)
            self.selected_genre = None
            self.class_names = []
            self.class_to_idx = {}
            self.list_authors.clear()
            self.btn_train.setEnabled(False)
            self.btn_save_model.setEnabled(False)
            self.lbl_train_status.setText(
                "Статус: Корневая папка выбрана. Загрузите жанр."
            )
            self.clear_sample_images()
            self.clear_all_results_and_plots()
            logging.info(f"Выбрана корневая директория: {self.root_dir}")

    # --- Метод для кнопки "Загрузить жанр" ---
    @Slot()
    def load_genre_data(self):
        if not self.root_dir:
            QMessageBox.warning(
                self, "Ошибка", "Сначала выберите корневую папку датасета."
            )
            return

        genres = (
            [d.name for d in self.root_dir.iterdir() if d.is_dir()]
            if self.root_dir.is_dir()
            else []
        )
        genre_name, ok = self.get_genre_from_user(genres)

        if not ok or not genre_name:
            logging.warning("Загрузка жанра отменена.")
            return

        self.selected_genre = genre_name
        genre_path = self.root_dir / self.selected_genre

        if not genre_path.is_dir():
            QMessageBox.warning(
                self,
                "Ошибка",
                f"Папка жанра '{self.selected_genre}' не найдена в {self.root_dir}.",
            )
            self.selected_genre = None
            self.btn_train.setEnabled(False)
            self.btn_save_model.setEnabled(False)
            return

        try:
            author_dirs = sorted([d for d in genre_path.iterdir() if d.is_dir()])
            self.class_names = [d.name for d in author_dirs]
            self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

            if not self.class_names:
                QMessageBox.warning(
                    self,
                    "Внимание",
                    f"В папке жанра '{self.selected_genre}' не найдено папок авторов.",
                )
                self.list_authors.clear()
                self.btn_train.setEnabled(False)
                self.btn_load_evaluate.setEnabled(False)
                self.btn_save_model.setEnabled(False)
                return

            self.list_authors.clear()
            self.list_authors.addItems(self.class_names)
            self.lbl_selected_genre.setText(
                f"Загружен жанр: {self.selected_genre} ({len(self.class_names)} авторов)"
            )
            # Активируем кнопки обучения и загрузки/оценки
            self.btn_train.setEnabled(True)
            self.btn_save_model.setEnabled(False)
            self.lbl_train_status.setText(
                f"Статус: Данные жанра '{self.selected_genre}' готовы."
            )
            self.btn_show_samples.setEnabled(False)
            self.clear_sample_images()
            self.clear_all_results_and_plots()
            logging.info(
                f"Загружен жанр '{self.selected_genre}'. Авторы: {self.class_names}"
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Ошибка", f"Ошибка при сканировании папки жанра: {e}"
            )
            logging.error(f"Ошибка сканирования {genre_path}: {e}", exc_info=True)
            self.selected_genre = None
            self.list_authors.clear()
            # Сбрасываем кнопки
            self.btn_train.setEnabled(False)
            # self.btn_load_evaluate.setEnabled(False)
            self.btn_save_model.setEnabled(False)

    # --- Вспомогательный метод для выбора жанра ---
    def get_genre_from_user(self, available_genres):
        if available_genres:
            logging.info(f"Автоматически выбран первый жанр: {available_genres[0]}")
            return available_genres[0], True
        elif self.selected_genre:
            return self.selected_genre, True
        else:
            QMessageBox.warning(
                self,
                "Нет жанров",
                "Не найдено подпапок (жанров) в выбранной директории.",
            )
            return None, False

    # --- Метод для кнопки "Показать примеры автора" ---
    @Slot()
    def show_author_samples(self):
        current_item = self.list_authors.currentItem()
        if not current_item:
            return
        author_name = current_item.text()

        if not self.root_dir or not self.selected_genre:
            logging.warning("Не выбрана папка или жанр для показа примеров.")
            return

        author_path = self.root_dir / self.selected_genre / author_name
        try:
            image_files = (
                list(author_path.glob("*.jpg"))
                + list(author_path.glob("*.jpeg"))
                + list(author_path.glob("*.png"))
            )

            if not image_files:
                QMessageBox.information(
                    self,
                    "Нет изображений",
                    f"Не найдено файлов изображений для автора {author_name}.",
                )
                self.clear_sample_images()
                return

            # Показываем до 4 случайных примеров
            num_samples = min(len(image_files), 4)
            selected_files = random.sample(image_files, num_samples)

            self.clear_sample_images()

            logging.info(f"Показ {num_samples} примеров для {author_name}")
            max_thumb_size = 150  # Макс размер превью

            for img_path in selected_files:
                pixmap = QPixmap(str(img_path))
                label = QLabel()
                if pixmap.isNull():
                    logging.warning(f"Не удалось загрузить pixmap: {img_path}")
                    label.setText(f"Error\n{img_path.name}")
                    label.setStyleSheet("border: 1px dashed red; color: red;")
                else:
                    scaled_pixmap = pixmap.scaled(
                        max_thumb_size,
                        max_thumb_size,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                    label.setPixmap(scaled_pixmap)
                    label.setToolTip(str(img_path.name))  # Подсказка с именем файла
                label.setFixedSize(max_thumb_size, max_thumb_size)
                label.setAlignment(Qt.AlignCenter)
                self.sample_images_layout.addWidget(label)

        except Exception as e:
            logging.error(
                f"Ошибка при показе примеров для {author_name}: {e}", exc_info=True
            )
            QMessageBox.critical(self, "Ошибка", f"Не удалось показать примеры: {e}")
            self.clear_sample_images()

    def clear_sample_images(self):
        for i in reversed(range(self.sample_images_layout.count())):
            widget_to_remove = self.sample_images_layout.itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.deleteLater()
        self.loaded_images_sample = []
        self.loaded_author_sample = None

    # --- Метод для кнопки "Выбрать изображение для теста" ---
    @Slot()
    def run_inference_on_image(self):
        if not self.loaded_inference_model:
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

                img_size = self.current_img_size

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

                pred_idx = self.loaded_inference_model.predict(features)[0]
                if 0 <= pred_idx < len(self.loaded_inference_class_names):
                    pred_author = self.loaded_inference_class_names[pred_idx]
                    try:
                        probabilities = self.loaded_inference_model.predict_proba(
                            features
                        )[0]
                        confidence = probabilities[
                            pred_idx
                        ]  # Берем вероятность предсказанного класса
                        pred_text = f"Предсказанный автор:\n{pred_author}\n(Уверенность: {confidence:.2f})"
                    except Exception:
                        pred_text = f"Предсказанный автор:\n{pred_author}"
                else:
                    pred_text = f"Ошибка: Индекс предсказания ({pred_idx}) вне диапазона классов ({len(self.loaded_inference_class_names)})."
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

    # --- Методы для вкладки "Обучение и Результаты" ---
    @Slot()
    def start_training_process(self):
        """Запускает полный цикл обучения и оценки."""
        if not self.selected_genre or not self.root_dir:
            QMessageBox.warning(
                self, "Ошибка", "Сначала выберите датасет и загрузите жанр."
            )
            return

        self.clear_all_results_and_plots()
        self.set_ui_state(training=True)

        params = {
            "root_dir": str(self.root_dir),
            "genre": self.selected_genre,
            "img_size": self.current_img_size,
            "test_size": self.current_test_size,
            "hidden_layers": (128,),
            "alpha": 1e-4,
            "random_state": self.current_random_state,
            "max_iter": 300,
        }

        self.worker = Worker(self.training_thread_func, **params)
        self.thread = QThread()
        self._connect_worker_signals(
            self.worker, self.thread, self.display_training_results
        )
        self.thread.start()

    @Slot()
    def load_and_evaluate_model(self):
        """Загружает модель и оценивает ее на текущем датасете."""
        if not self.selected_genre or not self.root_dir:
            QMessageBox.warning(
                self, "Ошибка", "Сначала выберите датасет и загрузите жанр."
            )
            return

        model_path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить модель для оценки", "", "Joblib Files (*.joblib)"
        )
        if not model_path:
            return

        self.clear_all_results_and_plots()
        self.set_ui_state(training=True)
        self.lbl_train_status.setText("Статус: Загрузка модели и оценка...")

        # Запускаем оценку в потоке
        self.worker = Worker(
            self.evaluation_thread_func,
            model_path=model_path,
            root_dir=str(self.root_dir),
            genre=self.selected_genre,
        )
        self.thread = QThread()
        # Используем другой слот для отображения результатов!
        self._connect_worker_signals(
            self.worker, self.thread, self.display_evaluation_results
        )
        self.thread.start()

    def _connect_worker_signals(self, worker, thread, result_slot):
        """Вспомогательный метод для подключения сигналов воркера."""
        worker.moveToThread(thread)
        worker.progress.connect(self.update_progress)
        worker.log_message.connect(self.update_status_label)
        worker.results.connect(result_slot)  # Подключаем нужный слот для результатов
        worker.finished.connect(self.on_task_finished)
        worker.error.connect(self.on_task_error)
        thread.started.connect(worker.run)

    def training_thread_func(
        self,
        root_dir,
        genre,
        img_size,
        test_size,
        hidden_layers,
        alpha,
        random_state,
        max_iter,
        progress_callback,
        log_callback,
    ):
        """Поток для ПОЛНОГО цикла обучения."""
        try:
            log_callback("1/5 Загрузка данных...")
            progress_callback(5)
            X_imgs, y, class_names, _ = data_loader.load_data(
                root_dir, genre, img_size=img_size
            )
            if X_imgs.size == 0:
                raise ValueError("Нет данных для обучения.")
            if len(class_names) < 2:
                raise ValueError("Менее 2 классов для обучения.")
            log_callback(
                f"Загружено {len(X_imgs)} изображений, {len(class_names)} классов."
            )
            progress_callback(20)

            log_callback("2/5 Извлечение признаков...")
            progress_callback(30)
            X_features = feature_extractor.extract_features(X_imgs)
            log_callback(f"Признаки извлечены. Форма: {X_features.shape}")
            progress_callback(60)

            log_callback("3/5 Разделение данных...")
            progress_callback(65)
            X_train, X_test, y_train, y_test = train_test_split(
                X_features,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
            log_callback(f"Обучение: {len(X_train)}, Тест: {len(X_test)}")
            progress_callback(70)

            log_callback("4/5 Обучение MLP...")
            progress_callback(75)
            # Сохраняем параметры, которые будут записаны с моделью
            training_params = {
                "random_state": random_state,
                "test_size": test_size,
                "img_size": img_size,
            }
            model = model_utils.build_classifier(
                hidden_layer_sizes=hidden_layers,
                alpha=alpha,
                random_state=random_state,
                max_iter=max_iter,
            )
            model.fit(X_train, y_train)
            log_callback("Обучение завершено.")
            progress_callback(90)

            log_callback("5/5 Оценка и подготовка результатов...")
            progress_callback(95)
            y_pred = model.predict(X_test)
            probabilities = model.predict_proba(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report_str = classification_report(
                y_test, y_pred, target_names=class_names, zero_division=0
            )
            cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names)))

            loss_curve = model.named_steps["mlp"].loss_curve_
            validation_scores = None
            if model.named_steps["mlp"].early_stopping and hasattr(
                model.named_steps["mlp"], "validation_scores_"
            ):
                validation_scores = model.named_steps["mlp"].validation_scores_
            n_iter_ = model.named_steps["mlp"].n_iter_
            loss_ = model.named_steps["mlp"].loss_

            scaler = model.named_steps["scaler"]
            X_train_scaled = scaler.transform(X_train)
            pca = PCA(n_components=3, random_state=random_state)  # 3 компоненты
            X_train_pca = pca.fit_transform(X_train_scaled)

            progress_callback(100)
            log_callback("Анализ обучения завершен.")

            results = {
                "type": "training",
                "model": model,
                "class_names": class_names,
                "training_params": training_params,  # Параметры для сохранения
                # Результаты для вкладки "Обучение и Результаты"
                "cm": cm,
                "report": report_str,
                "probabilities": probabilities,
                "y_test": y_test,
                "accuracy": accuracy,
                # Результаты для вкладки "Графики Обучения"
                "loss_curve": loss_curve,
                "validation_scores": validation_scores,
                "n_iter": n_iter_,
                "final_loss": loss_,
                # Результаты для вкладки "3D Визуализация"
                "X_projected": X_train_pca,
                "y_train": y_train,
            }
            return results

        except Exception as e:
            log_callback(f"Ошибка обучения: {e}")
            raise

    def evaluation_thread_func(
        self, model_path, root_dir, genre, progress_callback, log_callback
    ):
        """Поток ТОЛЬКО для оценки загруженной модели."""
        try:
            log_callback("1/4 Загрузка модели...")
            progress_callback(10)
            model, class_names, training_params = model_utils.load_model(model_path)
            log_callback(
                f"Модель загружена. Классы: {class_names}. Параметры: {training_params}"
            )

            # Получаем параметры для пересоздания выборки
            random_state = training_params.get("random_state", 42)
            test_size = training_params.get("test_size", 0.2)
            img_size = training_params.get("img_size", (128, 128))
            log_callback(
                f"Используются параметры: random_state={random_state}, test_size={test_size}, img_size={img_size}"
            )
            progress_callback(25)

            log_callback("2/4 Загрузка данных для оценки...")
            progress_callback(30)
            # Загружаем ВЕСЬ датасет заново
            X_imgs, y, current_class_names, _ = data_loader.load_data(
                root_dir, genre, img_size=img_size
            )
            if X_imgs.size == 0:
                raise ValueError(f"Нет данных для оценки в {root_dir}/{genre}")
            # Проверка совместимости классов
            if set(class_names) != set(current_class_names):
                log_callback(
                    f"ВНИМАНИЕ: Классы в модели {class_names} не совпадают с классами в датасете {current_class_names}!"
                )
            log_callback(f"Загружено {len(X_imgs)} изображений.")
            progress_callback(50)

            log_callback("3/4 Извлечение признаков...")
            progress_callback(60)
            X_features = feature_extractor.extract_features(X_imgs)
            log_callback("Признаки извлечены.")
            progress_callback(80)

            log_callback("4/4 Пересоздание тестовой выборки и оценка...")
            progress_callback(85)
            # Пересоздаем ТУ ЖЕ САМУЮ тестовую выборку
            _, X_test, _, y_test = train_test_split(
                X_features,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
            log_callback(
                f"Тестовая выборка ({len(X_test)} примеров) создана. Оценка..."
            )

            y_pred = model.predict(X_test)
            probabilities = model.predict_proba(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report_str = classification_report(
                y_test, y_pred, target_names=class_names, zero_division=0
            )
            cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names)))

            progress_callback(100)
            log_callback("Оценка завершена.")

            # Возвращаем ТОЛЬКО результаты оценки
            results = {
                "type": "evaluation",  # Тип результата
                "model": model,  # Возвращаем загруженную модель для возможного сохранения
                "class_names": class_names,
                "training_params": training_params,  # Для возможного сохранения
                "cm": cm,
                "report": report_str,
                "probabilities": probabilities,
                "y_test": y_test,
                "accuracy": accuracy,
            }
            return results

        except Exception as e:
            log_callback(f"Ошибка оценки: {e}")
            raise

    @Slot(int)
    def update_progress(self, value):
        self.progress_bar.setValue(value)

    @Slot(str)
    def update_status_label(self, message):
        self.lbl_train_status.setText(f"Статус: {message}")
        logging.info(f"GUI Status: {message}")

    @Slot(object)
    def display_training_results(self, results):
        """Отображает ВСЕ результаты ПОСЛЕ ОБУЧЕНИЯ."""
        self.last_run_results = results
        self.trained_model = results.get("model")

        # 1. Обновляем вкладку "Обучение и Результаты"
        self.update_confusion_matrix(results["cm"], results["class_names"])
        self.report_display.setText(results.get("report", "N/A"))
        self.update_confidence_plot(results)

        # 2. Обновляем вкладку "Графики Обучения"
        self.update_metrics_plot(results)
        self.update_compact_training_info(results)

        # 3. Обновляем вкладку "3D Визуализация"
        self.update_projection_plot(results)

        # 4. Обновляем статус и кнопки
        accuracy = results.get("accuracy", -1)
        self.lbl_train_status.setText(
            f"Обучение завершено. Точность на тесте: {accuracy:.4f}"
        )
        self.set_ui_state(
            training=False, model_ready=True
        )  # Разблокируем UI, модель готова

    @Slot(object)
    def display_evaluation_results(self, results):
        """Отображает ТОЛЬКО результаты оценки ПОСЛЕ ЗАГРУЗКИ модели."""
        self.last_run_results = results
        self.trained_model = results.get("model")

        # 1. Обновляем вкладку "Обучение и Результаты"
        self.update_confusion_matrix(results["cm"], results["class_names"])
        self.report_display.setText(results.get("report", "N/A"))
        self.update_confidence_plot(results)

        # 2. Очищаем вкладки "Графики Обучения" и "3D Визуализация"
        self.clear_training_plots_and_info()
        self.clear_3d_plot()

        # 3. Обновляем статус и кнопки
        accuracy = results.get("accuracy", -1)
        self.lbl_train_status.setText(
            f"Оценка загруженной модели завершена. Точность на тесте: {accuracy:.4f}"
        )
        self.set_ui_state(
            training=False, model_ready=True
        )  # Разблокируем UI, модель готова для сохранения

    @Slot()
    def on_task_finished(self):
        """Вызывается при завершении любого потока (обучение/оценка)."""
        self.set_ui_state(training=False, model_ready=(self.trained_model is not None))
        # Очистка потока
        if self.thread:
            self.thread.quit()
            self.thread.wait()
            self.thread.deleteLater()
            if self.worker:
                self.worker.deleteLater()
        self.thread = None
        self.worker = None
        logging.info("Фоновый поток завершен и очищен.")

    @Slot(str)
    def on_task_error(self, error_message):
        """Вызывается при ошибке в любом потоке."""
        self.set_ui_state(
            training=False, model_ready=False
        )  # Ошибка -> модель не готова
        self.lbl_train_status.setText(
            f"Статус: Ошибка - {error_message.splitlines()[0]}"
        )
        QMessageBox.critical(self, "Ошибка", f"Произошла ошибка:\n{error_message}")
        # Очистка потока
        if self.thread:
            self.thread.quit()
            self.thread.wait()
            self.thread.deleteLater()
            if self.worker:
                self.worker.deleteLater()
        self.thread = None
        self.worker = None
        logging.info("Фоновый поток остановлен из-за ошибки.")

    def set_ui_state(self, training: bool, model_ready: bool = False):
        """Управляет активностью кнопок и прогресс-бара."""
        self.btn_train.setEnabled(not training and self.selected_genre is not None)
        self.btn_save_model.setEnabled(not training and model_ready)
        self.progress_bar.setVisible(training)
        # Блокируем смену датасета во время работы
        self.tab_dataset.setEnabled(not training)

    # --- Методы обновления и очистки для вкладки "Обучение и Результаты" ---
    def update_confusion_matrix(self, cm, class_names):
        self.table_confusion_matrix.clear()
        if cm is None or class_names is None:
            return
        n_classes = len(class_names)
        self.table_confusion_matrix.setRowCount(n_classes)
        self.table_confusion_matrix.setColumnCount(n_classes)
        self.table_confusion_matrix.setHorizontalHeaderLabels(class_names)
        self.table_confusion_matrix.setVerticalHeaderLabels(class_names)
        for i in range(n_classes):
            for j in range(n_classes):
                item = QTableWidgetItem(str(cm[i, j]))
                item.setTextAlignment(Qt.AlignCenter)
                if i == j:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                    item.setBackground(QColor(220, 255, 220))
                self.table_confusion_matrix.setItem(i, j, item)
        self.table_confusion_matrix.resizeColumnsToContents()
        self.table_confusion_matrix.resizeRowsToContents()

    def update_confidence_plot(self, results):
        try:
            self.ax_confidence.clear()
            probabilities = results.get("probabilities")
            y_test = results.get("y_test")
            class_names = results.get("class_names")
            if probabilities is None or y_test is None or class_names is None:
                self.ax_confidence.set_title("Нет данных для графика уверенности")
            else:
                n_classes = len(class_names)
                probs_per_true_class = [[] for _ in range(n_classes)]
                for i in range(len(y_test)):
                    true_class_idx = y_test[i]
                    prob_for_true_class = probabilities[i, true_class_idx]
                    probs_per_true_class[true_class_idx].append(prob_for_true_class)
                # [...] остальной код отрисовки boxplot [...]
                bp = self.ax_confidence.boxplot(
                    probs_per_true_class,
                    labels=class_names,
                    vert=True,
                    patch_artist=True,
                    showfliers=False,
                )
                cmap = plt.get_cmap("viridis", n_classes)
                for patch, color in zip(
                    bp["boxes"], [cmap(i / n_classes) for i in range(n_classes)]
                ):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
                self.ax_confidence.set_title(
                    "Распределение уверенности модели для истинных классов"
                )
                self.ax_confidence.set_ylabel(
                    "Вероятность, присвоенная истинному классу"
                )
                self.ax_confidence.set_xlabel("Истинный класс (Автор)")
                self.ax_confidence.set_ylim(0, 1.05)
                self.ax_confidence.yaxis.grid(
                    True, linestyle="-", which="major", color="lightgrey", alpha=0.5
                )
                if n_classes > 5:
                    plt.setp(
                        self.ax_confidence.get_xticklabels(), rotation=30, ha="right"
                    )
                self.fig_confidence.tight_layout()
            self.canvas_confidence.draw()
        except Exception as e:
            logging.error(f"Ошибка отрисовки графика уверенности: {e}", exc_info=True)
            self.ax_confidence.clear()
            self.ax_confidence.set_title("Ошибка отрисовки")
            self.canvas_confidence.draw()

    def clear_results_outputs(self):
        """Очищает элементы на вкладке Обучение и Результаты."""
        self.table_confusion_matrix.clear()
        self.table_confusion_matrix.setRowCount(0)
        self.table_confusion_matrix.setColumnCount(0)
        self.report_display.clear()
        self.ax_confidence.clear()
        self.ax_confidence.set_title("Ожидание данных...")
        self.canvas_confidence.draw()

    # --- Методы обновления и очистки для вкладки "Графики Обучения" ---
    def update_metrics_plot(self, results):
        try:
            self.ax_metrics.clear()
            loss_curve = results.get("loss_curve")
            validation_scores = results.get("validation_scores")
            if loss_curve is None:
                self.ax_metrics.set_title("Нет данных для графика обучения")
            else:
                # [...] код отрисовки loss_curve и validation_scores [...]
                epochs = np.arange(1, len(loss_curve) + 1)
                color = "tab:red"
                self.ax_metrics.set_xlabel("Эпохи")
                self.ax_metrics.set_ylabel("Loss", color=color)
                (line1,) = self.ax_metrics.plot(
                    epochs, loss_curve, color=color, label="Training Loss"
                )
                self.ax_metrics.tick_params(axis="y", labelcolor=color)
                lines = [line1]
                labels = [line1.get_label()]
                if validation_scores is not None:
                    ax2 = self.ax_metrics.twinx()
                    color = "tab:blue"
                    ax2.set_ylabel("Validation Score (Accuracy)", color=color)
                    val_epochs = np.arange(1, len(validation_scores) + 1)
                    (line2,) = ax2.plot(
                        val_epochs,
                        validation_scores,
                        color=color,
                        label="Validation Score",
                    )
                    ax2.tick_params(axis="y", labelcolor=color)
                    ax2.set_ylim(0, 1.05)
                    lines.append(line2)
                    labels.append(line2.get_label())
                    self.fig_metrics.tight_layout()
                self.ax_metrics.set_title("Кривая потерь и оценка валидации")
                self.ax_metrics.grid(True, axis="y", linestyle="--", alpha=0.6)
                self.ax_metrics.legend(lines, labels, loc="best")
            self.canvas_metrics.draw()
        except Exception as e:
            logging.error(f"Ошибка отрисовки графика обучения: {e}", exc_info=True)
            self.ax_metrics.clear()
            self.ax_metrics.set_title("Ошибка отрисовки")
            self.canvas_metrics.draw()

    def update_compact_training_info(self, results):
        """Обновляет текстовую информацию об обучении."""
        n_iter = results.get("n_iter", "N/A")
        final_loss = results.get("final_loss", "N/A")
        self.lbl_final_iter.setText(str(n_iter))
        self.lbl_final_loss.setText(
            f"{final_loss:.4f}"
            if isinstance(final_loss, (int, float))
            else str(final_loss)
        )
        # Показываем метки
        self.lbl_final_iter_label.setVisible(True)
        self.lbl_final_iter.setVisible(True)
        self.lbl_final_loss_label.setVisible(True)
        self.lbl_final_loss.setVisible(True)

    def clear_training_plots_and_info(self):
        """Очищает элементы на вкладке Графики Обучения."""
        self.ax_metrics.clear()
        self.ax_metrics.set_title("Ожидание данных...")
        self.canvas_metrics.draw()
        self.lbl_final_iter.setText("N/A")
        self.lbl_final_loss.setText("N/A")
        self.lbl_final_iter_label.setVisible(False)
        self.lbl_final_iter.setVisible(False)
        self.lbl_final_loss_label.setVisible(False)
        self.lbl_final_loss.setVisible(False)

    # --- Методы обновления и очистки для вкладки "3D Визуализация" ---
    def update_projection_plot(self, results):
        """
        Обновляет 3D Class Connection Plot:
        - Вертикальные линии вдоль оси Z от общей минимальной Z до медианного Z класса.
        - Соединительные пунктирные линии от каждой точки к медианному ref_point.
        """
        try:
            self.ax_projection.clear()

            X_projected = results.get("X_projected")
            y_train = results.get("y_train")
            class_names = results.get("class_names")

            # Проверяем, что есть хотя бы 3 ПК
            if (
                X_projected is None
                or y_train is None
                or class_names is None
                or X_projected.shape[1] < 3
            ):
                self.ax_projection.set_title("Нет данных для 3D визуализации")
                self.canvas_projection.draw()
                return

            n_classes = len(class_names)
            cmap = plt.get_cmap("viridis", n_classes)

            # Глобальная минимальная Z для всех точек
            z_min_global = np.min(X_projected[:, 2])

            # Задаём границы осей с отступами
            mins = X_projected.min(axis=0)
            maxs = X_projected.max(axis=0)
            margins = (maxs - mins) * 0.05
            self.ax_projection.set_xlim(mins[0] - margins[0], maxs[0] + margins[0])
            self.ax_projection.set_ylim(mins[1] - margins[1], maxs[1] + margins[1])
            self.ax_projection.set_zlim(z_min_global - margins[2], maxs[2] + margins[2])

            handles = []
            for class_idx in range(n_classes):
                mask = y_train == class_idx
                pts = X_projected[mask]
                if pts.size == 0:
                    continue

                color = cmap(class_idx)

                # Центры по X и Y
                center_x = pts[:, 0].mean()
                center_y = pts[:, 1].mean()
                # Медианное Z для класса
                median_z = np.median(pts[:, 2])

                # 1) Вертикальная линия вдоль Z от z_min_global до median_z
                self.ax_projection.plot(
                    [center_x, center_x],  # X не меняется
                    [center_y, center_y],  # Y не меняется
                    [z_min_global, median_z],  # Z: от общего минимума до медианы
                    color=color,
                    linewidth=2,
                )

                # Точка назначения для соединений
                ref_point = (center_x, center_y, median_z)

                # 2) Соединительные линии от каждой точки к ref_point
                for pt in pts:
                    self.ax_projection.plot(
                        [pt[0], ref_point[0]],
                        [pt[1], ref_point[1]],
                        [pt[2], ref_point[2]],
                        color=color,
                        linestyle=":",
                        linewidth=0.8,
                        alpha=0.4,
                    )

                # 3) Точки PCA
                sc = self.ax_projection.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    color=color,
                    s=10,
                    alpha=0.6,
                    label=class_names[class_idx],
                )
                handles.append(sc)

            # Оформление
            self.ax_projection.set_title("3D Class Connection Plot (PCA)")
            self.ax_projection.set_xlabel("PC 1 (X)")
            self.ax_projection.set_ylabel("PC 2 (Y)")
            self.ax_projection.set_zlabel("PC 3 (Z)")
            self.ax_projection.legend(
                handles=handles,
                title="Авторы",
                loc="center left",
                bbox_to_anchor=(1.05, 0.5),
                fontsize="small",
            )

            self.canvas_projection.draw()

        except Exception as e:
            logging.error(f"Ошибка 3D визуализации: {e}", exc_info=True)
            self.ax_projection.clear()
            self.ax_projection.set_title("Ошибка отрисовки 3D визуализации")
            self.canvas_projection.draw()

    # вар2  - 3d график в виде куста
    # ef update_projection_plot(self, results):
    #     """
    #     Обновляет график 3D визуализации (теперь Class Connection Plot).
    #     Отображает точки PCA и соединяет их с верхней точкой
    #     вертикальной линии, представляющей Y-диапазон класса в его XZ-центре.
    #     """
    #     try:
    #         self.ax_projection.clear() # Очищаем оси перед отрисовкой

    #         X_projected = results.get('X_projected')
    #         y_train = results.get('y_train')
    #         class_names = results.get('class_names')

    #         if X_projected is None or y_train is None or class_names is None \
    #            or X_projected.shape[1] < 3:
    #              self.ax_projection.set_title("Нет данных для 3D визуализации")
    #              self.canvas_projection.draw()
    #              return

    #         n_classes = len(class_names)
    #         cmap = plt.get_cmap('viridis', n_classes)

    #         # Устанавливаем границы осей заранее по всем данным
    #         x_min, x_max = np.min(X_projected[:, 0]), np.max(X_projected[:, 0])
    #         y_min, y_max = np.min(X_projected[:, 1]), np.max(X_projected[:, 1])
    #         z_min, z_max = np.min(X_projected[:, 2]), np.max(X_projected[:, 2])
    #         margin_x = (x_max - x_min) * 0.05
    #         margin_y = (y_max - y_min) * 0.05
    #         margin_z = (z_max - z_min) * 0.05
    #         self.ax_projection.set_xlim(x_min - margin_x, x_max + margin_x)
    #         self.ax_projection.set_ylim(y_min - margin_y, y_max + margin_y)
    #         self.ax_projection.set_zlim(z_min - margin_z, z_max + margin_z)

    #         plot_handles = [] # Для легенды

    #         # Рисуем для каждого класса
    #         for class_idx in range(n_classes):
    #             # Выбираем данные текущего класса
    #             class_mask = (y_train == class_idx)
    #             class_points = X_projected[class_mask] # (n_points, 3)

    #             if class_points.shape[0] == 0: continue # Пропускаем, если нет точек

    #             color = cmap(class_idx / n_classes)

    #             # Находим центр в XZ и диапазон Y
    #             center_x = np.mean(class_points[:, 0])
    #             center_z = np.mean(class_points[:, 2])
    #             min_y_class = np.min(class_points[:, 1])
    #             max_y_class = np.max(class_points[:, 1])
    #             # mid_y_class = (min_y_class + max_y_class) / 2 # Середина диапазона Y

    #             # 1. Рисуем центральную вертикальную линию для класса
    #             # Линия от минимального Y до максимального Y класса в XZ-центре
    #             line, = self.ax_projection.plot(
    #                 [center_x, center_x],       # X координаты (не меняются)
    #                 [min_y_class, max_y_class], # Y координаты (от min до max)
    #                 [center_z, center_z],       # Z координаты (не меняются)
    #                 color=color,
    #                 linestyle='-',
    #                 linewidth=2,
    #                 marker='_', markersize=8 # Маркеры на концах линии
    #             )
    #             # Добавляем маркер линии в легенду (если еще не добавлен для этого цвета)
    #             # Это немного хак, т.к. line plot не имеет facecolor, но для легенды сойдет
    #             # if class_idx == 0: # Примерно так, но лучше создать отдельный хэндл
    #             #    plot_handles.append(line) # Добавляем только одну линию для легенды

    #             # Верхняя точка центральной линии, к которой будем тянуть линии
    #             top_ref_point = (center_x, max_y_class, center_z)

    #             # 2. Рисуем точки и соединительные линии
    #             for i in range(class_points.shape[0]):
    #                 point = class_points[i] # (px, py, pz)
    #                 # Соединительная линия от точки к верху центральной линии
    #                 self.ax_projection.plot(
    #                     [point[0], top_ref_point[0]], # X: от точки до центра
    #                     [point[1], top_ref_point[1]], # Y: от точки до max_y
    #                     [point[2], top_ref_point[2]], # Z: от точки до центра
    #                     color=color,
    #                     linestyle=':', # Пунктир
    #                     linewidth=0.8,
    #                     alpha=0.4      # Полупрозрачная
    #                 )

    #             # 3. Рисуем сами точки данных (небольшие)
    #             # Сохраняем объект scatter для использования в легенде
    #             scatter = self.ax_projection.scatter(
    #                 class_points[:, 0], class_points[:, 1], class_points[:, 2],
    #                 color=color,
    #                 marker='o',
    #                 s=10, # Маленький размер точек
    #                 alpha=0.6,
    #                 label=class_names[class_idx] # Метка для легенды
    #             )
    #             plot_handles.append(scatter) # Добавляем scatter в хэндлы легенды

    #         # Настраиваем оси и заголовок
    #         self.ax_projection.set_title('3D Class Connection Plot (PCA)')
    #         self.ax_projection.set_xlabel('PC 1')
    #         self.ax_projection.set_ylabel('PC 2 (Высота)')
    #         self.ax_projection.set_zlabel('PC 3')

    #         # Создаем и размещаем легенду
    #         # Используем хэндлы от scatter объектов
    #         self.ax_projection.legend(handles=plot_handles, labels=class_names, title="Авторы",
    #                                    loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize='small')

    #         self.canvas_projection.draw()
    #         logging.info("График 3D Class Connection Plot обновлен.")

    #     except Exception as e:
    #         logging.error(f"Ошибка отрисовки 3D визуализации: {e}", exc_info=True)
    #         self.ax_projection.clear()
    #         self.ax_projection.set_title("Ошибка отрисовки 3D визуализации")
    #         self.canvas_projection.draw()
    # исходный  - 3d в виде разброса точек
    # def update_projection_plot(self, results):
    #     """Обновляет график 3D PCA проекции."""
    #     try:
    #         self.ax_projection.clear() # Очищаем оси перед отрисовкой
    #         X_projected = results.get('X_projected')
    #         y_train = results.get('y_train')
    #         class_names = results.get('class_names')

    #         if X_projected is None or y_train is None or class_names is None or X_projected.shape[1] < 3:
    #              self.ax_projection.set_title("Нет данных для 3D PCA")
    #         else:
    #             n_classes = len(class_names)
    #             cmap = plt.get_cmap('viridis', n_classes)
    #             scatter = self.ax_projection.scatter(
    #                 X_projected[:, 0], X_projected[:, 1], X_projected[:, 2],
    #                 c=y_train, cmap=cmap, alpha=0.7, edgecolors='w', linewidth=0.5, depthshade=True
    #             )
    #             self.ax_projection.set_title('3D PCA проекция обучающих данных')
    #             self.ax_projection.set_xlabel('PC 1'); self.ax_projection.set_ylabel('PC 2'); self.ax_projection.set_zlabel('PC 3')

    #             # Легенда: Попробуем разместить справа внизу относительно осей
    #             handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i / n_classes), markersize=8, alpha=0.7) for i in range(n_classes)]
    #             # bbox_to_anchor(x, y, width, height) - якорь относительно осей (1,0) - нижний правый угол
    #             # loc='lower right' - какая часть легенды привязана к якорю
    #             # Уменьшим размер шрифта легенды
    #             self.ax_projection.legend(handles=handles, labels=class_names, title="Авторы",
    #                                        loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize='small')
    #             # Важно! Может потребоваться подогнать bbox_to_anchor и вызвать fig.tight_layout()
    #             # self.fig_projection.tight_layout(rect=[0, 0, 0.9, 1]) # Оставить место справа для легенды

    #         self.canvas_projection.draw()
    #         logging.info("График 3D PCA обновлен.")
    #     except Exception as e: logging.error(f"Ошибка отрисовки 3D PCA: {e}", exc_info=True); self.ax_projection.clear(); self.ax_projection.set_title("Ошибка отрисовки"); self.canvas_projection.draw()

    def clear_3d_plot(self):
        """Очищает 3D график."""
        self.ax_projection.clear()
        self.ax_projection.set_title("Ожидание данных...")
        self.canvas_projection.draw()

    # --- Общие методы очистки и сохранения ---
    @Slot()
    def save_trained_model(self):
        """Сохраняет последнюю успешно обученную или загруженную модель."""
        if not self.trained_model:  # Проверяем наличие модели
            QMessageBox.warning(
                self,
                "Модель не готова",
                "Нет модели для сохранения. Сначала обучите или загрузите и оцените модель.",
            )
            return

        model_to_save = self.trained_model
        # Используем параметры из последнего запуска (обучения или оценки)
        params_to_save = self.last_run_results.get("training_params", {})
        class_names_to_save = self.last_run_results.get("class_names", [])

        model_dir = Path(f"{env.get_parent_dir(env.prefix)}/models")
        model_dir.mkdir(parents=True, exist_ok=True)
        genre_name = self.selected_genre if self.selected_genre else "unknown_genre"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = model_dir / f"{genre_name}_model_{timestamp}.joblib"

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить модель", str(default_filename), "Joblib Files (*.joblib)"
        )
        if save_path:
            try:
                model_utils.save_model(
                    model_to_save,
                    class_names_to_save,
                    save_path,
                    training_params=params_to_save,
                )
                QMessageBox.information(
                    self, "Успех", f"Модель успешно сохранена в:\n{save_path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Ошибка сохранения", f"Не удалось сохранить модель:\n{e}"
                )

    def clear_all_results_and_plots(self):
        """Очищает все результаты и графики во всех вкладках."""
        logging.debug("Очистка всех результатов и графиков...")
        self.clear_results_outputs()
        self.clear_training_plots_and_info()
        self.clear_3d_plot()
        self.last_run_results = {}
        self.trained_model = None
        self.btn_save_model.setEnabled(False)

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

                self.loaded_inference_model_path = filepath
                self.loaded_inference_model = model
                self.loaded_inference_class_names = names

                self.lbl_inference_model_status.setText(
                    f"Загружено: {Path(filepath).name} ({len(names)} кл.)"
                )
                self.btn_test_image.setEnabled(True)
                logging.info(f"Модель для инференса загружена: {filepath}")
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Ошибка загрузки модели",
                    f"Не удалось загрузить модель для инференса:\n{e}",
                )
                self.loaded_inference_model = None
                self.loaded_inference_class_names = None
                self.lbl_inference_model_status.setText("Ошибка загрузки")
                self.btn_test_image.setEnabled(False)

    @Slot()
    def run_inference_on_image(self):
        if not self.loaded_inference_model:
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
                img_size = self.current_img_size

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
                pred_idx = self.loaded_inference_model.predict(features)[0]
                pred_author = self.loaded_inference_class_names[pred_idx]
                try:
                    probs = self.loaded_inference_model.predict_proba(features)[0]
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PainterAttributionApp()
    window.show()
    sys.exit(app.exec())
