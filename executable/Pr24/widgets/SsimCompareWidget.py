
import cv2
from skimage.metrics import structural_similarity  # Для SSIM

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
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

class SsimCompareWidget(QWidget):
    def __init__(self, main_window_ref):
        super().__init__()
        self.main_window = main_window_ref
        self.original_image_for_ssim = None
        self.decoded_image_for_ssim = None  # Без АПТ 
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        compare_layout = QHBoxLayout()

        # --- Группа Оригинального Изображения ---
        orig_group = QVBoxLayout()
        self.original_label = QLabel("Original Input Image")
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_label.setMinimumSize(350, 350)
        self.original_label.setStyleSheet(
            "border: 1px solid gray; background-color: #e8e8e8;"
        )
        orig_group.addWidget(self.original_label)
        compare_layout.addLayout(orig_group)

        # --- Группа Декодированного Изображения ---
        decoded_group = QVBoxLayout()
        self.decoded_label = QLabel("Decoded Image (No Landmarks)")
        self.decoded_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.decoded_label.setMinimumSize(350, 350)
        self.decoded_label.setStyleSheet(
            "border: 1px solid gray; background-color: #e8e8e8;"
        )
        decoded_group.addWidget(self.decoded_label)
        compare_layout.addLayout(decoded_group)
        main_layout.addLayout(compare_layout)
        
        # --- Кнопки и Результат ---
        control_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Images for Comparison")
        self.load_button.clicked.connect(self.load_images)
        self.calculate_button = QPushButton("Calculate SSIM")
        self.calculate_button.setEnabled(False)  # Включится после загрузки
        self.calculate_button.clicked.connect(self.calculate_ssim)
        self.result_label = QLabel("Similarity (SSIM): --.--%")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = self.result_label.font()
        font.setPointSize(12)
        font.setBold(True) # Optional
        self.result_label.setFont(font)

        control_layout.addWidget(self.load_button)
        control_layout.addWidget(self.calculate_button)
        control_layout.addWidget(self.result_label, 1)  # Растянуть метку результата

        main_layout.addLayout(control_layout)

    def load_images(self):
        # Получаем ссылки на изображения из главного окна
        self.original_image_for_ssim = getattr(self.main_window, "input_image", None)
        self.decoded_image_for_ssim = getattr(
            self.main_window, "decoded_image_no_landmarks", None
        )

        loaded_orig = False
        loaded_decoded = False

        if self.original_image_for_ssim is not None:
            pixmap_orig = cv_image_to_qpixmap(
                self.original_image_for_ssim,
                self.original_label.width() - 5,
                self.original_label.height() - 5,
            )
            self.original_label.setPixmap(pixmap_orig)
            loaded_orig = True
        else:
            self.original_label.clear()
            self.original_label.setText("Original Input Image\n[No Data]")

        if self.decoded_image_for_ssim is not None:
            pixmap_decoded = cv_image_to_qpixmap(
                self.decoded_image_for_ssim,
                self.decoded_label.width() - 5,
                self.decoded_label.height() - 5,
            )
            self.decoded_label.setPixmap(pixmap_decoded)
            loaded_decoded = True
        else:
            self.decoded_label.clear()
            self.decoded_label.setText("Decoded Image (No Landmarks)\n[No Data]")

        can_calculate = loaded_orig and loaded_decoded
        self.calculate_button.setEnabled(can_calculate)
        self.result_label.setText("Similarity (SSIM): --.--%")  # Сброс результата

        if not can_calculate:
            # Информируем пользователя, если чего-то не хватает
            missing = []
            if not loaded_orig:
                missing.append("original input image (from Encode tab)")
            if not loaded_decoded:
                missing.append("decoded image (run Decode tab first)")
            QMessageBox.information(
                self,
                "Load Info",
                f"Missing data for comparison:\n- {'\n- '.join(missing)}",
            )

    def calculate_ssim(self):
        if self.original_image_for_ssim is None or self.decoded_image_for_ssim is None:
            QMessageBox.warning(
                self, "Error", "Both original and decoded images must be loaded."
            )
            self.result_label.setText("Similarity (SSIM): Error")
            return

        img1 = self.original_image_for_ssim
        img2 = self.decoded_image_for_ssim

        # Проверка размеров
        if img1.shape != img2.shape:
            QMessageBox.warning(
                self,
                "Dimension Mismatch",
                f"Images have different dimensions:\nOriginal: {img1.shape}\nDecoded: {img2.shape}\n"
                "Cannot calculate SSIM.",
            )
            self.result_label.setText("Similarity (SSIM): Size Error")
            return

        # SSIM обычно вычисляется на grayscale изображениях
        try:
            # Проверка количества каналов перед конвертацией
            if len(img1.shape) < 3 or img1.shape[2] != 3:
                img1_gray = (
                    img1
                    if len(img1.shape) == 2
                    else cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                )  # На всякий случай
            else:
                img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

            if len(img2.shape) < 3 or img2.shape[2] != 3:
                img2_gray = (
                    img2
                    if len(img2.shape) == 2
                    else cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                )
            else:
                img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # --- Вычисление SSIM ---
            # data_range рекомендуется указывать, для uint8 это 255
            # win_size должен быть нечетным и <= min(img_height, img_width)
            # Убедимся, что win_size не больше размеров изображения
            min_dim = min(img1_gray.shape)
            # Стандартное значение 7 или 11, но нужно адаптировать, если изображение меньше
            win_size = min(
                7, min_dim if min_dim % 2 != 0 else min_dim - 1
            )  # Должен быть нечетным
            if win_size < 3:  # Минимально допустимый размер окна, кажется, 3
                QMessageBox.warning(
                    self,
                    "Calculation Warning",
                    f"Image dimension ({min_dim}) too small for standard SSIM window size. Result might be less reliable.",
                )
                # Можно либо остановить, либо продолжить с win_size=min_dim (если нечетное)
                win_size = (
                    min_dim if min_dim % 2 != 0 else min_dim - 1
                )  # Попробуем с максимально возможным нечетным

            if win_size >= 3:
                ssim_value, diff = structural_similarity(
                    img1_gray, img2_gray, full=True, data_range=255, win_size=win_size
                )

                # Отображение результата
                result_text = f"Similarity (SSIM): {ssim_value * 100:.2f}%"
                self.result_label.setText(result_text)
                print(f"SSIM Calculation: {result_text} (win_size={win_size})")
            else:
                QMessageBox.critical(
                    self,
                    "Calculation Error",
                    f"Image dimension ({min_dim}) too small to calculate SSIM.",
                )
                self.result_label.setText("Similarity (SSIM): Size Error")

            # Опционально: показать изображение разницы (diff)
            # diff_vis = (diff * 255).astype("uint8")
            # cv2.imshow("SSIM Difference", diff_vis) # Или показать в отдельном QLabel

        except ImportError:
            QMessageBox.critical(
                self,
                "Import Error",
                "Scikit-image library not found. Please install it (`pip install scikit-image`).",
            )
            self.result_label.setText("Similarity (SSIM): Lib Error")
        except Exception as e:
            QMessageBox.critical(
                self, "SSIM Calculation Error", f"An error occurred: {e}"
            )
            self.result_label.setText("Similarity (SSIM): Calc Error")
            print(f"Error during SSIM calculation: {e}")
