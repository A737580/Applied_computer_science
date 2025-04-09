import os
import cv2
import numpy as np

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QSizePolicy,
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


class SplitPapVisWidget(QWidget):
    def __init__(self, main_window_ref):
        super().__init__()
        self.main_window = main_window_ref
        self.pap_r_channel = None  # Scrambled R
        self.pap_g_channel = None  # QR
        self.pap_b_channel = None  # Scrambled B
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        vis_layout = QHBoxLayout()

        # --- Middle section: Preview of selected PAP ---
        self.visualize_pap_preview_label = QLabel("PAP Code Preview")
        self.visualize_pap_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.visualize_pap_preview_label.setMaximumHeight(150)
        self.visualize_pap_preview_label.setStyleSheet("border: 1px solid lightgray; background-color: #f8f8f8;")
        self.visualize_pap_preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        main_layout.addWidget(self.visualize_pap_preview_label)
        
        layer_min_width = 200
        layer_min_height = 200
        
        # --- Группа Канала R  ---
        r_group = QVBoxLayout()
        self.r_label = QLabel("1. R Channel (Scrambled R)")
        self.r_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.r_label.setMinimumSize(layer_min_width, layer_min_height)
        self.r_label.setStyleSheet("border: 1px solid gray; background-color: #ffeeee;")
        self.save_r_button = QPushButton("Save R Channel")
        self.save_r_button.setEnabled(False)
        self.save_r_button.clicked.connect(self.save_r)
        r_group.addWidget(self.r_label)
        r_group.addWidget(self.save_r_button)
        vis_layout.addLayout(r_group)

        # --- Группа Канала G (HCC2D QR-code) ---
        g_group = QVBoxLayout()
        self.g_label = QLabel("2. G Channel (HCC2D QR-code)")
        self.g_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.r_label.setMinimumSize(layer_min_width, layer_min_height)
        self.g_label.setStyleSheet("border: 1px solid gray; background-color: #eeffee;")
        self.save_g_button = QPushButton("Save G Channel (HCC2D QR-code)")
        self.save_g_button.setEnabled(False)
        self.save_g_button.clicked.connect(self.save_g)
        g_group.addWidget(self.g_label)
        g_group.addWidget(self.save_g_button)
        vis_layout.addLayout(g_group)

        # --- Группа Канала B (Scrambled B) ---
        b_group = QVBoxLayout()
        self.b_label = QLabel("3. B Channel (Scrambled B)")
        self.b_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.r_label.setMinimumSize(layer_min_width, layer_min_height)
        self.b_label.setStyleSheet("border: 1px solid gray; background-color: #eeeeff;")
        self.save_b_button = QPushButton("Save B Channel")
        self.save_b_button.setEnabled(False)
        self.save_b_button.clicked.connect(self.save_b)
        b_group.addWidget(self.b_label)
        b_group.addWidget(self.save_b_button)
        vis_layout.addLayout(b_group)

        main_layout.addLayout(vis_layout)

        # Кнопка обновления данных
        self.update_button = QPushButton("Load/Update PAP from Encoder/Decoder")
        self.update_button.clicked.connect(self.update_data)
        main_layout.addWidget(self.update_button)


    def update_data(self):
        pap_image_to_split = None
        source_name = None
        
        # Пытаемся взять из кодера сначала
        if self.main_window.pap_code_image is not None:
            pap_image_to_split = self.main_window.pap_code_image
            source_name = "Encoder"
        # Если нет, пытаемся загрузить из пути декодера
        elif self.main_window.pap_code_path and os.path.exists(self.main_window.pap_code_path):
            pap_image_to_split = cv2.imread(self.main_window.pap_code_path)
            source_name = "Decoder Path"
            if pap_image_to_split is None:
                QMessageBox.warning(
                    self,
                    "Load Error",
                    f"Failed to load PAP image from:\n{self.main_window.pap_code_path}",
                )
                source_name = None  
        if pap_image_to_split is not None and len(pap_image_to_split.shape) == 3:
            print(f"Splitting PAP image from {source_name} for visualization.")
            
            pixmap_m = cv_image_to_qpixmap(
               pap_image_to_split, self.visualize_pap_preview_label.width() - 5, self.visualize_pap_preview_label.height() - 5
            )
            self.visualize_pap_preview_label.setPixmap(pixmap_m)
            try:
                self.pap_b_channel, self.pap_g_channel, self.pap_r_channel = cv2.split(
                    pap_image_to_split
                )
                h, w = self.pap_b_channel.shape
                if h <= 0 or w <= 0: raise ValueError("PAP image channels have invalid dimensions.")
                zeros = np.zeros((h, w), dtype=self.pap_b_channel.dtype) 
                
                pixmap_r = cv_image_to_qpixmap(
                    cv2.merge([zeros, zeros, self.pap_r_channel]),
                    self.r_label.width() - 5,
                    self.r_label.height() - 5,
                )
                self.r_label.setPixmap(pixmap_r)
                self.save_r_button.setEnabled(True)

                pixmap_g = cv_image_to_qpixmap(
                    cv2.merge([zeros, self.pap_g_channel, zeros]),
                    self.g_label.width() - 5,
                    self.g_label.height() - 5,
                )
                self.g_label.setPixmap(pixmap_g)
                self.save_g_button.setEnabled(True)

                pixmap_b = cv_image_to_qpixmap(
                    cv2.merge([self.pap_b_channel, zeros, zeros]),
                    self.b_label.width() - 5,
                    self.b_label.height() - 5,
                )
                self.b_label.setPixmap(pixmap_b)
                self.save_b_button.setEnabled(True)

            except Exception as e:
                QMessageBox.critical(
                    self, "Split Error", f"Error splitting PAP image: {e}"
                )
                self._clear_all()
        else:
            QMessageBox.information(
                self, "No Data", "No valid PAP code found in Encoder or Decoder tabs."
            )
            self.visualize_pap_preview_label.clear()
            self.visualize_pap_preview_label.setText("PAP Code preview\n[No Data]")
            self._clear_all()

    def _clear_all(self):
        self.pap_r_channel = None
        self.pap_g_channel = None
        self.pap_b_channel = None
        self.r_label.clear()
        self.r_label.setText("1. R Channel (Scrambled R)\n[No Data]")
        self.g_label.clear()
        self.g_label.setText("2. G Channel (QR)\n[No Data]")
        self.b_label.clear()
        self.b_label.setText("3. B Channel (Scrambled B)\n[No Data]")
        self.save_r_button.setEnabled(False)
        self.save_g_button.setEnabled(False)
        self.save_b_button.setEnabled(False)

    def _save_channel(self, channel_data, default_suffix):
        # Аналогично _save_channel из PreScrambleVisWidget
        if channel_data is None:
            QMessageBox.warning(self, "Save Error", "No data available to save.")
            return
        base_name = "pap_split"
        # Попробуем взять имя из пути декодера, если он есть
        if self.main_window.pap_code_path:
            base_name = os.path.splitext(
                os.path.basename(self.main_window.pap_code_path)
            )[0]
        elif self.main_window.input_image_path:  # Иначе из исходного
            base_name = os.path.splitext(
                os.path.basename(self.main_window.input_image_path)
            )[0]

        default_path = f"{base_name}_{default_suffix}.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self, f"Save {default_suffix} Channel", default_path, "PNG Image (*.png)"
        )
        if file_path:
            if not file_path.lower().endswith(".png"):
                file_path += ".png"
            try:
                success = cv2.imwrite(file_path, channel_data)
                if not success:
                    raise IOError("cv2.imwrite returned false.")
                QMessageBox.information(
                    self, "Success", f"{default_suffix} channel saved to:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Save Error", f"Failed to save {default_suffix} channel:\n{e}"
                )

    def save_r(self):
        self._save_channel(self.pap_r_channel, "R_ScrambledR")

    def save_g(self):
        self._save_channel(self.pap_g_channel, "G_QR")

    def save_b(self):
        self._save_channel(self.pap_b_channel, "B_ScrambledB")

