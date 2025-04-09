import os
import cv2

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QSpinBox,
    QMessageBox,
)
from PySide6.QtGui import QPixmap, QImage, QPainter
from PySide6.QtCore import Qt, QSize

def cv_image_to_qpixmap(cv_img, target_width=-1, target_height=-1):
    """
    Конвертирует OpenCV изображение (BGR или Grayscale) в QPixmap для отображения.
    Масштабирует для вписывания в target_width ИЛИ target_height, если задано, сохраняя пропорции.
    """
    # Проверка входного изображения
    if (
        cv_img is None
        or cv_img.size == 0
        or cv_img.shape[0] <= 0
        or cv_img.shape[1] <= 0
    ):
        ph_size = 50  # Размер плейсхолдера
        pixmap = QPixmap(ph_size, ph_size)
        pixmap.fill(Qt.GlobalColor.lightGray)
        painter = QPainter(pixmap)
        painter.setPen(Qt.GlobalColor.red)
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "No/Empty\nImage")
        painter.end()
        return pixmap

    try:
        height, width = cv_img.shape[:2]
        bytes_per_line = 0
        qt_format = QImage.Format.Format_Invalid

        if len(cv_img.shape) == 3:  # Color BGR assumed
            if cv_img.shape[2] == 3:
                rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                qt_format = QImage.Format.Format_RGB888
                bytes_per_line = 3 * width
                qt_image = QImage(
                    rgb_image.data, width, height, bytes_per_line, qt_format
                )
            else:  # Handle other cases like BGRA if needed, currently raises error
                raise ValueError(f"Unsupported number of channels: {cv_img.shape[2]}")
        elif len(cv_img.shape) == 2:  # Grayscale
            qt_format = QImage.Format.Format_Grayscale8
            bytes_per_line = width
            qt_image = QImage(cv_img.data, width, height, bytes_per_line, qt_format)
        else:
            raise ValueError("Unsupported image dimensions")

        if qt_image.isNull():
            raise ValueError("Failed to create QImage from data.")

        pixmap = QPixmap.fromImage(qt_image)
        if pixmap.isNull():
            raise ValueError("Failed to create QPixmap from QImage")

        # --- Логика масштабирования ---
        current_size = pixmap.size()
        orig_w = current_size.width()
        orig_h = current_size.height()

        if orig_w <= 0 or orig_h <= 0:  # Проверка на всякий случай
            print("Warning: Original pixmap size is invalid.")
            return pixmap  # Возвращаем как есть

        # Убедимся, что целевые размеры положительны, если заданы
        target_width = max(1, target_width) if target_width > 0 else -1
        target_height = max(1, target_height) if target_height > 0 else -1

        # Если нет ограничений или изображение уже меньше ограничений, масштабирование не нужно
        if (target_width == -1 or orig_w <= target_width) and (
            target_height == -1 or orig_h <= target_height
        ):
            return pixmap  # Возвращаем оригинал

        # Рассчитываем новый размер с сохранением пропорций
        new_w = float(orig_w)
        new_h = float(orig_h)
        ratio = new_w / new_h if new_h > 0 else 1.0

        # Применяем ограничение по ширине
        if target_width > 0 and new_w > target_width:
            new_w = float(target_width)
            new_h = new_w / ratio if ratio != 0 else 0  # Avoid division by zero

        # Применяем ограничение по высоте (к возможно уже измененному размеру)
        if target_height > 0 and new_h > target_height:
            new_h = float(target_height)
            new_w = new_h * ratio

        # Округляем и проверяем минимальный размер
        final_w = max(1, int(round(new_w)))
        final_h = max(1, int(round(new_h)))

        # Создаем целевой QSize
        scaled_qsize = QSize(final_w, final_h)

        # Выполняем масштабирование QPixmap один раз
        # Проверяем, нужно ли реально масштабировать (изменился ли размер)
        if scaled_qsize != current_size:
            pixmap = pixmap.scaled(
                scaled_qsize,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        # --- Конец логики масштабирования ---

        return pixmap

    except Exception as e:
        print(f"Error converting image to QPixmap: {e}")
        # Создаем плейсхолдер с ошибкой
        err_size = 50
        error_pixmap = QPixmap(err_size, err_size)
        error_pixmap.fill(Qt.GlobalColor.gray)
        painter = QPainter(error_pixmap)
        painter.setPen(Qt.GlobalColor.red)
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        error_text = f"Display\nError:\n{type(e).__name__}"
        painter.drawText(
            error_pixmap.rect(),
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
            error_text,
        )
        painter.end()
        return error_pixmap

class LsbEmbeddingWidget(QWidget):
    """
    Виджет для встраивания изображения PAP QR-кода в LSB слоя
    изображения-контейнера.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.container_image_path = None
        self.container_image = None # Оригинальный контейнер (cv2 BGR)
        self.pap_code_path = None
        self.pap_code_image = None  # Изображение PAP QR-кода (cv2 BGR)
        self.result_image = None    # Изображение с встроенным кодом (cv2 BGR)

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # --- Секция загрузки файлов ---
        load_layout = QHBoxLayout()
        self.load_container_button = QPushButton("1. Загрузить контейнер")
        self.load_container_button.clicked.connect(self._load_container_image)
        self.load_pap_button = QPushButton("2. Загрузить Bio QR-code PAP")
        self.load_pap_button.clicked.connect(self._load_pap_image)

        load_layout.addWidget(self.load_container_button)
        load_layout.addWidget(self.load_pap_button)
        main_layout.addLayout(load_layout)

        # --- Отображение путей к файлам (опционально) ---
        path_layout = QHBoxLayout()
        self.container_path_label = QLabel("Контейнер: (не выбран)")
        self.container_path_label.setWordWrap(True)
        self.pap_path_label = QLabel("PAP QR-code: (не выбран)")
        self.pap_path_label.setWordWrap(True)
        path_layout.addWidget(self.container_path_label, 1) # Растягиваем
        path_layout.addWidget(self.pap_path_label, 1)     # Растягиваем
        main_layout.addLayout(path_layout)

        # --- Секция предпросмотра изображений ---
        preview_layout = QHBoxLayout()
        preview_min_size = 250

        # Контейнер
        container_group = QVBoxLayout()
        container_title = QLabel("Контейнер (Оригинал)")
        container_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.container_preview_label = QLabel()
        self.container_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.container_preview_label.setMinimumSize(preview_min_size, preview_min_size)
        self.container_preview_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.container_preview_label.setText("[Изображение контейнера]")
        container_group.addWidget(container_title)
        container_group.addWidget(self.container_preview_label)
        preview_layout.addLayout(container_group)

        # PAP QR-код
        pap_group = QVBoxLayout()
        pap_title = QLabel("PAP QR-code")
        pap_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pap_preview_label = QLabel()
        self.pap_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pap_preview_label.setMinimumSize(preview_min_size, preview_min_size)
        self.pap_preview_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.pap_preview_label.setText("[PAP QR-code]")
        pap_group.addWidget(pap_title)
        pap_group.addWidget(self.pap_preview_label)
        preview_layout.addLayout(pap_group)

        # Результат
        result_group = QVBoxLayout()
        result_title = QLabel("Результат (Контейнер + PAP LSB)")
        result_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_preview_label = QLabel()
        self.result_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_preview_label.setMinimumSize(preview_min_size, preview_min_size)
        self.result_preview_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.result_preview_label.setText("[Результат встраивания]")
        result_group.addWidget(result_title)
        result_group.addWidget(self.result_preview_label)
        preview_layout.addLayout(result_group)

        main_layout.addLayout(preview_layout)

        # --- Секция управления и встраивания ---
        control_layout = QHBoxLayout()
        self.lsb_label = QLabel("Количество LSB бит для встраивания:")
        self.lsb_spinbox = QSpinBox()
        self.lsb_spinbox.setRange(1, 8) # От 1 до 8 бит
        self.lsb_spinbox.setValue(1)    # По умолчанию 1 бит
        self.lsb_spinbox.valueChanged.connect(self._update_embedding_state) # Обновляем кнопку при смене

        self.embed_button = QPushButton("3. Встроить PAP в LSB")
        self.embed_button.setEnabled(False) # Включится после загрузки
        self.embed_button.clicked.connect(self._perform_embedding)

        self.save_result_button = QPushButton("Сохранить результат")
        self.save_result_button.setEnabled(False) # Включится после встраивания
        self.save_result_button.clicked.connect(self._save_result_image)

        control_layout.addWidget(self.lsb_label)
        control_layout.addWidget(self.lsb_spinbox)
        control_layout.addStretch(1) # Добавляем растяжение между элементами
        control_layout.addWidget(self.embed_button)
        control_layout.addWidget(self.save_result_button)
        main_layout.addLayout(control_layout)

        # --- Статус бар (опционально) ---
        self.status_label = QLabel("Статус: Готов к загрузке файлов.")
        main_layout.addWidget(self.status_label)

    def _update_path_labels(self):
        """Обновляет текстовые метки с путями к файлам."""
        if self.container_image_path:
            self.container_path_label.setText(f"Контейнер: ...{os.path.basename(self.container_image_path)}")
        else:
            self.container_path_label.setText("Контейнер: (не выбран)")

        if self.pap_code_path:
            self.pap_path_label.setText(f"PAP QR-code: ...{os.path.basename(self.pap_code_path)}")
        else:
            self.pap_path_label.setText("PAP QR-code: (не выбран)")

    def _update_previews(self):
        """Обновляет все три метки предпросмотра изображений."""
        if self.container_image is not None:
            pixmap = cv_image_to_qpixmap(self.container_image,
                                         self.container_preview_label.width() - 5,
                                         self.container_preview_label.height() - 5)
            self.container_preview_label.setPixmap(pixmap)
        else:
            self.container_preview_label.clear()
            self.container_preview_label.setText("[Изображение контейнера]")

        if self.pap_code_image is not None:
            pixmap = cv_image_to_qpixmap(self.pap_code_image,
                                         self.pap_preview_label.width() - 5,
                                         self.pap_preview_label.height() - 5)
            self.pap_preview_label.setPixmap(pixmap)
        else:
            self.pap_preview_label.clear()
            self.pap_preview_label.setText("[PAP QR-code]")

        if self.result_image is not None:
             pixmap = cv_image_to_qpixmap(self.result_image,
                                         self.result_preview_label.width() - 5,
                                         self.result_preview_label.height() - 5)
             self.result_preview_label.setPixmap(pixmap)
        else:
            self.result_preview_label.clear()
            self.result_preview_label.setText("[Результат встраивания]")

    def _update_embedding_state(self):
        """Включает/выключает кнопку встраивания."""
        can_embed = (self.container_image is not None and
                     self.pap_code_image is not None and
                     self.lsb_spinbox.value() > 0) # Добавим проверку бит
        self.embed_button.setEnabled(can_embed)
        # Сбрасываем результат, если изменились входные данные или параметры
        if not can_embed or self.result_image is not None:
             self.result_image = None
             self.save_result_button.setEnabled(False)
             self._update_previews() # Обновить, чтобы убрать старый результат

    def _load_image(self, title, is_container):
        """Общая функция загрузки изображения."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, title, "", "Изображения (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if file_path:
            try:
                # Загружаем как BGR
                img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("Не удалось загрузить изображение. Возможно, файл поврежден или не поддерживается.")
                if len(img.shape) != 3 or img.shape[2] != 3:
                     # Попробуем конвертировать, если серое
                     if len(img.shape) == 2:
                         print(f"Предупреждение: Загружено одноканальное изображение ({'контейнер' if is_container else 'PAP'}), конвертируется в BGR.")
                         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                     else:
                         raise ValueError(f"Ожидается цветное (BGR) изображение, получено {img.shape}")

                if is_container:
                    self.container_image_path = file_path
                    self.container_image = img
                    self.status_label.setText("Статус: Контейнер загружен.")
                else:
                    self.pap_code_path = file_path
                    self.pap_code_image = img
                    self.status_label.setText("Статус: PAP QR-code загружен.")

                self._update_path_labels()
                self._update_previews()
                self._update_embedding_state() # Проверяем, можно ли встраивать

            except Exception as e:
                QMessageBox.critical(self, "Ошибка загрузки", f"Не удалось загрузить изображение:\n{e}")
                self.status_label.setText(f"Статус: Ошибка загрузки {'контейнера' if is_container else 'PAP'}.")
                if is_container:
                    self.container_image_path = None
                    self.container_image = None
                else:
                    self.pap_code_path = None
                    self.pap_code_image = None
                self._update_path_labels()
                self._update_previews()
                self._update_embedding_state()


    def _load_container_image(self):
        self._load_image("Выберите изображение-контейнер", is_container=True)

    def _load_pap_image(self):
        self._load_image("Выберите изображение Bio QR-code PAP", is_container=False)

    def _embed_lsb(self, container, pap, n_bits):
        """
        Встраивает изображение pap в n_bits младших бит изображения container.
        Возвращает новое изображение с встроенными данными.
        """
        if container is None or pap is None:
            raise ValueError("Контейнер и PAP изображение должны быть загружены.")
        if not (1 <= n_bits <= 8):
             raise ValueError("Количество бит LSB должно быть от 1 до 8.")

        c_h, c_w, c_ch = container.shape
        p_h, p_w, p_ch = pap.shape

        if p_h > c_h or p_w > c_w:
            raise ValueError(f"PAP изображение ({p_w}x{p_h}) больше, чем контейнер ({c_w}x{c_h}). Встраивание невозможно.")
        if c_ch != 3 or p_ch != 3:
             raise ValueError("Оба изображения должны быть в формате BGR (3 канала).")

        # Создаем копию контейнера для модификации
        result = container.copy()

        # Маска для очистки n_bits LSB контейнера
        # Пример: n_bits=1 -> mask=11111110 (0xFE)
        # Пример: n_bits=2 -> mask=11111100 (0xFC)
        # Пример: n_bits=8 -> mask=00000000 (0x00)
        clear_mask = (0xFF << n_bits) & 0xFF

        # Итерация по области встраивания (верхний левый угол)
        for y in range(p_h):
            for x in range(p_w):
                for ch in range(3): # Каналы B, G, R
                    # --- Значение пикселя контейнера ---
                    container_pixel = result[y, x, ch]

                    # --- Значение пикселя PAP ---
                    pap_pixel = pap[y, x, ch]

                    # 1. Очищаем LSB биты в пикселе контейнера
                    cleared_container_pixel = container_pixel & clear_mask

                    # 2. Берем старшие n_bits из пикселя PAP
                    # Сдвигаем вправо так, чтобы нужные биты оказались в LSB позиции
                    # Пример: n_bits=1, pap=10110101 -> pap_bits = 00000001
                    # Пример: n_bits=2, pap=10110101 -> pap_bits = 00000010
                    # Пример: n_bits=8, pap=10110101 -> pap_bits = 10110101
                    pap_bits_to_embed = pap_pixel >> (8 - n_bits)

                    # 3. Объединяем очищенный пиксель контейнера и биты PAP
                    result[y, x, ch] = cleared_container_pixel | pap_bits_to_embed

        return result


    def _perform_embedding(self):
        """Выполняет процесс встраивания."""
        if self.container_image is None or self.pap_code_image is None:
            QMessageBox.warning(self, "Ошибка", "Загрузите оба изображения перед встраиванием.")
            return

        n_lsb = self.lsb_spinbox.value()
        self.status_label.setText(f"Статус: Выполняется встраивание с {n_lsb} LSB битами...")
        QApplication.processEvents() # Обновляем интерфейс

        try:
            self.result_image = self._embed_lsb(self.container_image, self.pap_code_image, n_lsb)
            self.status_label.setText(f"Статус: Встраивание с {n_lsb} LSB завершено.")
            self.save_result_button.setEnabled(True)
            self._update_previews() # Показать результат

        except ValueError as ve:
             QMessageBox.critical(self, "Ошибка встраивания", f"Не удалось выполнить встраивание:\n{ve}")
             self.status_label.setText(f"Статус: Ошибка встраивания LSB.")
             self.result_image = None
             self.save_result_button.setEnabled(False)
             self._update_previews()
        except Exception as e:
            QMessageBox.critical(self, "Неизвестная ошибка", f"Произошла ошибка:\n{e}")
            self.status_label.setText(f"Статус: Неизвестная ошибка при встраивании.")
            self.result_image = None
            self.save_result_button.setEnabled(False)
            self._update_previews()

    def _save_result_image(self):
        """Сохраняет изображение с встроенным кодом."""
        if self.result_image is None:
            QMessageBox.warning(self, "Нет данных", "Нет результата для сохранения.")
            return

        # Предлагаем имя файла по умолчанию
        default_name = "result_embedded.png"
        if self.container_image_path:
            base, ext = os.path.splitext(os.path.basename(self.container_image_path))
            default_name = f"{base}_lsb_embedded{ext}"
            # Если контейнер был не png, меняем расширение
            if not default_name.lower().endswith(".png"):
                 default_name = os.path.splitext(default_name)[0] + ".png"


        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результат", default_name, "PNG Image (*.png);;Все файлы (*)"
        )

        if file_path:
            # Убедимся, что расширение .png (LSB лучше сохранять без потерь)
            if not file_path.lower().endswith(".png"):
                print("Предупреждение: Результат будет сохранен в формате PNG для избежания потерь LSB.")
                file_path = os.path.splitext(file_path)[0] + ".png"

            try:
                success = cv2.imwrite(file_path, self.result_image)
                if not success:
                    raise IOError("cv2.imwrite вернул False.")
                QMessageBox.information(self, "Успешно", f"Результат сохранен в:\n{file_path}")
                self.status_label.setText(f"Статус: Результат сохранен.")
            except Exception as e:
                 QMessageBox.critical(self, "Ошибка сохранения", f"Не удалось сохранить файл:\n{e}")
                 self.status_label.setText(f"Статус: Ошибка сохранения результата.")
