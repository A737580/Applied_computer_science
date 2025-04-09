import os
import cv2

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


class PreScrambleVisWidget(QWidget):
    def __init__(self, main_window_ref):  # Передаем ссылку на главное окно
        super().__init__()
        self.main_window = main_window_ref  # Сохраняем ссылку
        self.r_channel = None  # Original R
        self.qr_channel = None  # Generated QR (Centered) 
        self.b_channel = None  # Original B
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        vis_layout = QHBoxLayout()
        r_group = QVBoxLayout()
        
        # --- Middle section ---
        self.visualize_pap_preview_label = QLabel("GRAY Photo Preview")
        self.visualize_pap_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.visualize_pap_preview_label.setMaximumHeight(150)
        self.visualize_pap_preview_label.setStyleSheet("border: 1px solid lightgray; background-color: #f8f8f8;")
        self.visualize_pap_preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        main_layout.addWidget(self.visualize_pap_preview_label)
        
        layer_min_width = 200
        layer_min_height = 200
        
        self.r_label = QLabel("1. R Channel (Original)")
        self.r_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.r_label.setMinimumSize(layer_min_width, layer_min_height)
        self.r_label.setStyleSheet("border: 1px solid gray; background-color: #ffeeee;")
        self.save_r_button = QPushButton("Save R Channel")
        self.save_r_button.setEnabled(False)
        self.save_r_button.clicked.connect(self.save_r)
        r_group.addWidget(self.r_label)
        r_group.addWidget(self.save_r_button)
        vis_layout.addLayout(r_group)

        qr_group = QVBoxLayout()
        self.qr_label = QLabel("2. QR Channel (Centered)")   
        self.qr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.r_label.setMinimumSize(layer_min_width, layer_min_height)

        self.qr_label.setStyleSheet(
            "border: 1px solid gray; background-color: #eeffee;"
        )
        self.save_qr_button = QPushButton("Save G Channel (HCC2D QR-code)")
        self.save_qr_button.setEnabled(False)
        self.save_qr_button.clicked.connect(self.save_qr)
        qr_group.addWidget(self.qr_label)
        qr_group.addWidget(self.save_qr_button)
        vis_layout.addLayout(qr_group)

        b_group = QVBoxLayout()
        self.b_label = QLabel("3. B Channel (Original)")
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
        self.update_button = QPushButton("Load/Update Data from Encoder Tab")
        self.update_button.clicked.connect(self.update_data)
        main_layout.addWidget(self.update_button)

    def update_data(self):
        self.input_image = getattr(self.main_window, "input_image", None) 
        self.r_channel = getattr(self.main_window, "r_channel_prescramble", None)
        self.qr_channel = getattr(
            self.main_window, "qr_channel_prescramble", None
        )   
        self.b_channel = getattr(self.main_window, "b_channel_prescramble", None)
        updated = False
        if self.input_image is not None:
            img_gray_norm = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2GRAY)
            
            pixmap_m = cv_image_to_qpixmap(
                img_gray_norm, self.visualize_pap_preview_label.width() - 5, self.visualize_pap_preview_label.height() - 5
            )
            self.visualize_pap_preview_label.setPixmap(pixmap_m)
            updated = True
        else:
            self.visualize_pap_preview_label.clear()
            self.visualize_pap_preview_label.setText("Photo preview\n[No Data]")
        
        if self.r_channel is not None:
            pixmap_r = cv_image_to_qpixmap(
                self.r_channel, self.r_label.width() - 5, self.r_label.height() - 5
            )
            self.r_label.setPixmap(pixmap_r)
            self.save_r_button.setEnabled(True)
            updated = True
        else:
            self.r_label.clear()
            self.r_label.setText("1. R Channel\n[No Data]")
            self.save_r_button.setEnabled(False)
        if self.qr_channel is not None:
            pixmap_qr = cv_image_to_qpixmap(
                self.qr_channel, self.qr_label.width() - 5, self.qr_label.height() - 5
            )
            self.qr_label.setPixmap(pixmap_qr)
            self.save_qr_button.setEnabled(True)
            updated = True
        else:
            self.qr_label.clear()
            self.qr_label.setText("2. QR Channel (Centered)\n[No Data]")
            self.save_qr_button.setEnabled(False)  # Обновлен текст
        if self.b_channel is not None:
            pixmap_b = cv_image_to_qpixmap(
                self.b_channel, self.b_label.width() - 5, self.b_label.height() - 5
            )
            self.b_label.setPixmap(pixmap_b)
            self.save_b_button.setEnabled(True)
            updated = True
        else:
            self.b_label.clear()
            self.b_label.setText("3. B Channel\n[No Data]")
            self.save_b_button.setEnabled(False)
        if not updated:
            QMessageBox.information(
                self, "Load Info", "Generate data in 'Encode' tab first."
            )

    def _save_channel(self, channel_data, default_suffix):
        if channel_data is None:
            QMessageBox.warning(self, "Save Error", "No data.")
            return
        base_name = "prescramble_rgb"
        if self.main_window.input_image_path:
            base_name = os.path.splitext(
                os.path.basename(self.main_window.input_image_path)
            )[0]
        default_path = f"{base_name}_{default_suffix}.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self, f"Save {default_suffix}", default_path, "PNG (*.png)"
        )
        if file_path:
            if not file_path.lower().endswith(".png"):
                file_path += ".png"
            try:
                if not cv2.imwrite(file_path, channel_data):
                    raise IOError("imwrite failed")
                QMessageBox.information(self, "Success", f"Saved:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed save:\n{e}")

    def save_r(self):
        self._save_channel(self.r_channel, "R_original")

    def save_qr(self):
        self._save_channel(self.qr_channel, "QR_centered") 

    def save_b(self):
        self._save_channel(self.b_channel, "B_original")

