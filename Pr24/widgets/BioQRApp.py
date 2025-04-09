import os
import cv2
import dlib
import numpy as np
import json  # For serializing landmark data
import qrcode  # For generating QR code for Layer 2
from qrcode.image.pil import PilImage  # Specify PIL factory
from pyzbar import pyzbar  # For decoding QR code from Layer 2
from src.env import prefix

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QTabWidget,
    QSizePolicy,
    QMessageBox,
)
from PySide6.QtGui import QPixmap, QImage, QPainter
from PySide6.QtCore import Qt, QSize, QTimer

from widgets.PreScrambleVisWidget import PreScrambleVisWidget 
from widgets.SplitPapVisWidget import SplitPapVisWidget
from widgets.SsimCompareWidget import SsimCompareWidget

dlib_detector = None
dlib_predictor = None

DLIB_MODEL_FILENAME = fr"{prefix}\shape_predictor_68_face_landmarks.dat"

SCRAMBLE_SEED_R = 42 + 0
SCRAMBLE_SEED_B = 42 + 2

TARGET_IMG_SIZE = (250, 250)  # Желаемый размер выходного изображения (ширина, высота)
TARGET_IED = 80  # Желаемое расстояние между глазами в пикселях на выходном изображении
QR_BACKGROUND_COLOR = 255  # Цвет фона для G-канала вокруг QR-кода (128 - серый)



def initialize_dlib():
    global dlib_detector, dlib_predictor
    try:
        print("Initializing dlib detector...")
        dlib_detector = dlib.get_frontal_face_detector()
        print("Initializing dlib predictor...")
        dlib_predictor = dlib.shape_predictor(DLIB_MODEL_FILENAME)
        print("dlib initialized successfully.")
        return True
    except Exception as e:
        # Use print/log
        print(f"Error: Failed to initialize dlib detector/predictor: {e}")
        dlib_detector = None  # Ensure they are None on failure
        dlib_predictor = None
        return False


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


def scramble_channel(channel, seed):
    """Scrambles rows and columns of a 2D numpy array using a seed."""
    if channel is None or channel.size == 0:
        print("Warning: Attempting to scramble empty channel.")
        return None, None, None
    h, w = channel.shape
    np.random.seed(seed)  # Ensure reproducibility
    row_perm = np.random.permutation(h)
    col_perm = np.random.permutation(w)
    scrambled = channel[row_perm][:, col_perm]
    return scrambled, row_perm, col_perm


def unscramble_channel(scrambled_channel, row_perm_indices, col_perm_indices):
    """Unscrambles rows and columns using the inverse permutation indices."""
    if (
        scrambled_channel is None
        or row_perm_indices is None
        or col_perm_indices is None
    ):
        print("Error: Missing data for unscrambling.")
        return None
    # Проверка соответствия размеров перед созданием инверсий
    if scrambled_channel.shape[0] != len(row_perm_indices) or scrambled_channel.shape[
        1
    ] != len(col_perm_indices):
        print(
            f"Error: Dimension mismatch during unscramble. Scrambled: {scrambled_channel.shape}, RowPerms: {len(row_perm_indices)}, ColPerms: {len(col_perm_indices)}"
        )
        return None

    h, w = scrambled_channel.shape
    if h == 0 or w == 0:
        print("Warning: Attempting to unscramble empty channel.")
        return None  # Невозможно расшифровать пустое изображение

    try:
        # Calculate inverse permutations
        inv_row_perm = np.argsort(row_perm_indices)
        inv_col_perm = np.argsort(col_perm_indices)

        # Apply inverse permutations
        # Убедимся, что инверсные индексы соответствуют размерностям (хотя argsort должен гарантировать это)
        if len(inv_col_perm) != w or len(inv_row_perm) != h:
            print("Error: Inverse permutation index length mismatch (unexpected).")
            return None
        unscrambled = scrambled_channel[:, inv_col_perm][inv_row_perm]
        return unscrambled
    except IndexError as e:
        print(
            f"Error during unscrambling (IndexError likely due to size mismatch): {e}"
        )
        return None
    except Exception as e:
        print(f"Unexpected error during unscrambling: {e}")
        return None


def preprocess_iso19794_geometric(image, target_size=(250, 250), target_ied=80):
    """
    Выполняет геометрическую нормализацию лица на основе положения глаз.
    Args:
        image: Входное BGR изображение (NumPy array).
        target_size: Кортеж (ширина, высота) для выходного изображения.
        target_ied: Желаемое межзрачковое расстояние в пикселях.
    Returns:
        Нормализованное BGR изображение (NumPy array) или None в случае ошибки.
    """
    if dlib_detector is None or dlib_predictor is None:
        print("Error: Dlib not initialized for preprocessing.")
        return None
    if image is None or image.size == 0:
        print("Preprocessing Error: Input image is empty.")
        return None

    target_w, target_h = target_size
    img_gray = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY
    )  # Dlib работает с grayscale быстрее
    # Используем dlib для поиска лица и ориентиров
    faces = dlib_detector(img_gray)

    if not faces:
        print("Preprocessing Warning: No face detected.")
        # Можно вернуть оригинал или None. Возвращаем None, чтобы сигнализировать о неудаче.
        return None

    # Берем первое найденное лицо
    landmarks = dlib_predictor(img_gray, faces[0])
    parts = landmarks.parts()

    # Индексы для глаз в модели 68 точек
    left_eye_indices = list(range(36, 42))
    right_eye_indices = list(range(42, 48))

    # Вычисляем центры глаз
    left_eye_pts = np.array(
        [(p.x, p.y) for i, p in enumerate(parts) if i in left_eye_indices],
        dtype=np.float32,
    )
    right_eye_pts = np.array(
        [(p.x, p.y) for i, p in enumerate(parts) if i in right_eye_indices],
        dtype=np.float32,
    )

    if len(left_eye_pts) == 0 or len(right_eye_pts) == 0:
        print("Preprocessing Warning: Could not find all eye landmarks.")
        return None  # Не можем выровнять без глаз

    left_eye_center = left_eye_pts.mean(axis=0)
    right_eye_center = right_eye_pts.mean(axis=0)

    # Вычисляем угол наклона и текущее расстояние между глазами
    dx = right_eye_center[0] - left_eye_center[0]
    dy = right_eye_center[1] - left_eye_center[1]
    if dx == 0 and dy == 0:
        return None  # Точки совпадают

    dist = np.sqrt(dx * dx + dy * dy)
    angle = np.degrees(np.arctan2(dy, dx))  # Угол в градусах

    # Желаемый центр лица на выходном изображении (середина по ширине, чуть выше центра по высоте)
    desired_face_center_x = target_w / 2.0
    desired_face_center_y = target_h * 0.45  # Можно настроить

    # Центр между глазами в оригинальном изображении
    eyes_center = (
        (left_eye_center[0] + right_eye_center[0]) / 2.0,
        (left_eye_center[1] + right_eye_center[1]) / 2.0,
    )

    # Масштаб для достижения target_ied
    scale = float(target_ied) / dist

    # Получаем матрицу поворота вокруг центра глаз
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

    # Корректируем компонент трансляции в матрице M, чтобы центр глаз оказался в желаемой точке
    M[0, 2] += desired_face_center_x - eyes_center[0]
    M[1, 2] += desired_face_center_y - eyes_center[1]

    # Применяем аффинное преобразование
    try:
        normalized_face = cv2.warpAffine(
            image,
            M,
            (target_w, target_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        print(f"Geometric normalization applied. Angle={angle:.1f}, Scale={scale:.2f}")
        return normalized_face
    except Exception as e:
        print(f"Error during warpAffine: {e}")
        return None


class BioQRApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bio QR Code Encoder/Decoder/Visualizer (PAP)")  
        self.setGeometry(100, 100, 1050, 700)

        self.input_image = None
        self.input_image_path = None
        self.pap_code_image = None
        self.pap_code_path = None  

        # Атрибуты для вкладки "Визуализация до скремблирования"
        self.r_channel_prescramble = None
        self.qr_channel_prescramble = None
        self.b_channel_prescramble = None
        # Атрибут для вкладки "Сравнение SSIM"
        self.decoded_image_no_landmarks = None
        # Данные для разделенного PAP хранятся внутри SplitPapVisWidget

        self.decoded_image = None  

        # --- Создание вкладок ---
        self.tabs = QTabWidget()
        self.encoder_tab = QWidget()
        self.decoder_tab = QWidget()
        self.prescramble_vis_tab = PreScrambleVisWidget(self)
        self.split_pap_vis_tab = SplitPapVisWidget(self)
        self.ssim_compare_tab = SsimCompareWidget(self)

        self.tabs.addTab(self.encoder_tab, "Encode PAP")
        self.tabs.addTab(self.decoder_tab, "Decode PAP")
        self.tabs.addTab(self.prescramble_vis_tab, "Visualize Pre-Scramble")
        self.tabs.addTab(self.split_pap_vis_tab, "Visualize Split PAP")
        self.tabs.addTab(self.ssim_compare_tab, "Compare SSIM")

        # --- Настройка UI для основных вкладок ---
        self.setup_encoder_ui()
        self.setup_decoder_ui()

        self.setCentralWidget(self.tabs)

        # Инициализация dlib с задержкой
        QTimer.singleShot(50, self._initialize_dlib_safe)

    def _initialize_dlib_safe(self):
        """Initialize dlib after Qt event loop has started."""
        print("Attempting safe dlib initialization...")
        if not (dlib_detector and dlib_predictor):
            if not initialize_dlib():
                QMessageBox.warning(
                    self,
                    "Dlib Initialization Failed",
                    "Facial landmark features will be unavailable.\n"
                    "Ensure dlib library is installed and the model file\n"
                    f"'{DLIB_MODEL_FILENAME}' is present or downloadable.",
                )
            else:
                print("Dlib check/initialization successful.")
        else:
            print("Dlib already initialized.")

    def setup_encoder_ui(self):
        layout = QVBoxLayout(self.encoder_tab)
        display_layout = QHBoxLayout()

        # Input Image Side
        input_group = QVBoxLayout()
        self.input_image_label = QLabel("Select Input Image")
        self.input_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_image_label.setMinimumSize(400, 400)
        self.input_image_label.setStyleSheet(
            "border: 1px solid gray; background-color: #f0f0f0;"
        )
        self.input_image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.select_input_button = QPushButton("Select Image...")
        self.select_input_button.clicked.connect(self.select_input_image)
        input_group.addWidget(self.input_image_label)
        input_group.addWidget(self.select_input_button)

        # PAP Code Side
        pap_group = QVBoxLayout()
        self.pap_image_label = QLabel("PAP Code (R->R scrambled, anthropometric points->HCC2D QR-code, B->B scrambled)")
        self.pap_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pap_image_label.setMinimumSize(400, 400)
        self.pap_image_label.setStyleSheet(
            "border: 1px solid gray; background-color: #f0f0f0;"
        )
        self.pap_image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.save_pap_button = QPushButton("Save PAP Code...") 
        self.save_pap_button.clicked.connect(self.save_pap_code)
        self.save_pap_button.setEnabled(False)
        pap_group.addWidget(self.pap_image_label)
        pap_group.addWidget(self.save_pap_button)

        display_layout.addLayout(input_group)
        display_layout.addLayout(pap_group)

        self.generate_button = QPushButton(
            "Generate Bio QR Code (PAP)"
        )
        self.generate_button.clicked.connect(self.generate_pap_code)
        self.generate_button.setEnabled(False)

        layout.addLayout(display_layout)
        layout.addWidget(self.generate_button)

    def select_input_image(self):
        """Opens file dialog to select input image for encoding."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Input Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.input_image_original_path = file_path 
            self.input_image = None  
            
            self.r_channel_prescramble = None
            self.qr_channel_prescramble = None
            self.b_channel_prescramble = None
            self.decoded_image_no_landmarks = None

            # 1. Загрузка
            loaded_image = cv2.imread(self.input_image_original_path)
            if loaded_image is None:
                QMessageBox.warning(self, "Load Error", f"Failed load: {file_path}")
                self._reset_encoder_state()
                return 

            # 2. <<< ПРЕДОБРАБОТКА >>>
            print("Applying geometric normalization...")
            normalized_image = preprocess_iso19794_geometric(
                loaded_image, TARGET_IMG_SIZE, TARGET_IED
            )

            if normalized_image is None:
                QMessageBox.warning(
                    self,
                    "Preprocessing Failed",
                    "Could not detect face/landmarks or normalize image. Please select a clear, frontal face image.",
                )
                self._reset_encoder_state()
                return 

            # 3. Используем нормализованное изображение
            self.input_image = normalized_image
            print(f"Using normalized image of size: {self.input_image.shape}")

            # 4. Отображение нормализованного изображения
            label_w = max(10, self.input_image_label.width() - 10)
            label_h = max(10, self.input_image_label.height() - 10)
            pixmap = cv_image_to_qpixmap(self.input_image, label_w, label_h)
            self.input_image_label.setPixmap(pixmap)

            # 5. Включаем кнопку генерации и сбрасываем PAP панель
            self.generate_button.setEnabled(True)
            self.pap_image_label.clear()
            self.pap_image_label.setText("PAP Code (ScR->R, QR(Cntr)->G, ScB->B)")
            self.pap_code_image = None
            self.save_pap_button.setEnabled(False)

    def generate_pap_code(self):
        if self.input_image is None:
            QMessageBox.warning(self, "Input Error", "Input image needed.")
            return
        if dlib_detector is None:
            QMessageBox.warning(self, "Dlib Error", "Dlib not ready.")
            return
        # Сброс перед генерацией
        self.r_channel_prescramble = None
        self.qr_channel_prescramble = None
        self.b_channel_prescramble = None
        self.pap_code_image = None
        print("Generating PAP Code (RGB, Normalized Input)...")
        # Размеры берем из нормализованного изображения
        h, w = self.input_image.shape[:2]

        # Переменные для промежуточных шагов
        b_channel_orig, g_channel_orig, r_channel_orig = None, None, None
        g_channel_final_qr = None  # Канал G с центрированным QR
        scrambled_r, scrambled_b = None, None

        try:
            # 1. Split Normalized BGR Image
            b_channel_orig, g_channel_orig, r_channel_orig = cv2.split(self.input_image)
            print(
                f"Split Normalized BGR: {b_channel_orig.shape}"
            )  # Размер уже TARGET_IMG_SIZE

            # 2. Detect Landmarks (на нормализованном изображении)
            # Используем grayscale нормализованного для dlib
            img_gray_norm = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2GRAY)
            landmark_coords = []
            # Переопределяем лицо на нормализованном изображении (должно быть одно и по центру)
            faces = dlib_detector(img_gray_norm)
            if faces:
                landmarks = dlib_predictor(img_gray_norm, faces[0])
                landmark_coords = [(p.x, p.y) for p in landmarks.parts()]
                print(f"Detected {len(landmark_coords)} landmarks on normalized image.")
            else:
                # Этого не должно случиться, если нормализация прошла успешно, но проверим
                print("Warning: No face detected on normalized image ?!")
                # Можно либо прервать, либо продолжить с пустым QR

            # 3. <<< Prepare Layer G (QR Code Centered) >>>
            landmark_data_str = json.dumps(landmark_coords)
            qr = qrcode.QRCode(
                version=None,
                error_correction=qrcode.constants.ERROR_CORRECT_M,
                box_size=2,
                border=2,
            )  # Можно настроить box_size/border для контроля размера QR
            qr.add_data(landmark_data_str)
            qr.make(fit=True)
            qr_img_pil = qr.make_image(image_factory=PilImage).convert("L")
            qr_img_np = np.array(qr_img_pil)
            qr_h, qr_w = qr_img_np.shape
            print(f"QR generated: {qr_h}x{qr_w}")

            # Создаем фон для G канала
            g_channel_final_qr = np.full((h, w), QR_BACKGROUND_COLOR, dtype=np.uint8)

            # Вычисляем позицию для центрирования
            if qr_h < h and qr_w < w:  # Убедимся, что QR меньше фона
                y_offset = (h - qr_h) // 2
                x_offset = (w - qr_w) // 2
                # Вставляем QR в центр
                g_channel_final_qr[
                    y_offset : y_offset + qr_h, x_offset : x_offset + qr_w
                ] = qr_img_np
                print(f"QR centered in G channel at ({x_offset}, {y_offset})")
            else:
                # Если QR больше или равен фону (маловероятно при норм параметрах), просто используем QR
                print(
                    "Warning: Generated QR is too large, resizing it to fit G channel."
                )
                g_channel_final_qr = cv2.resize(
                    qr_img_np, (w, h), interpolation=cv2.INTER_NEAREST
                )

            # 4. Prepare Layers R & B (Scramble Original R & B)
            print(f"Scrambling R with seed: {SCRAMBLE_SEED_R}")
            scrambled_r, _, _ = scramble_channel(r_channel_orig, SCRAMBLE_SEED_R)
            if scrambled_r is None:
                raise ValueError("Failed scramble R.")
            print(f"Scrambling B with seed: {SCRAMBLE_SEED_B}")
            scrambled_b, _, _ = scramble_channel(b_channel_orig, SCRAMBLE_SEED_B)
            if scrambled_b is None:
                raise ValueError("Failed scramble B.")

            # --- Сохранение для визуализации (R ориг, Центр. QR, B ориг) ---
            self.r_channel_prescramble = r_channel_orig.copy()
            self.qr_channel_prescramble = (
                g_channel_final_qr.copy()
            )  # Сохраняем канал с центрированным QR
            self.b_channel_prescramble = b_channel_orig.copy()
            print("Stored pre-scramble channels (R, QR Cntr, B) for visualization.")

            # 5. Merge PAP image (BGR: ScB, QR(Cntr), ScR)
            if not (
                scrambled_b.shape
                == g_channel_final_qr.shape
                == scrambled_r.shape
                == (h, w)
            ):
                raise ValueError("Shape mismatch before merge.")
            self.pap_code_image = cv2.merge(
                [scrambled_b, g_channel_final_qr, scrambled_r]
            )
            print(f"Merged PAP (RGB): {self.pap_code_image.shape}")

            # 6. Display Preview
            label_w = max(10, self.pap_image_label.width() - 10)
            label_h = max(10, self.pap_image_label.height() - 10)
            pixmap = cv_image_to_qpixmap(self.pap_code_image, label_w, label_h)
            self.pap_image_label.setPixmap(pixmap)
            self.save_pap_button.setEnabled(True)
            print("PAP Code (RGB, Norm, QR Cntr) Generation Complete.")

        except Exception as e:
            QMessageBox.critical(self, "Encoding Error", f"Error: {e}")
            print(f"Error during encoding: {e}")
            self._reset_encoder_state()  # Полный сброс при ошибке

    def save_pap_code(self):
        if self.pap_code_image is None or self.pap_code_image.size == 0:
            QMessageBox.warning(
                self, "Save Error", "No valid PAP code generated to save."
            )
            return

        default_name = "pap_combined_chroma.png"
        if self.input_image_path:
            base = os.path.splitext(os.path.basename(self.input_image_path))[0]
            default_name = f"{base}_pap_combo.png"

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save PAP Code (Combined Chroma)", default_name, "PNG Image (*.png)"
        )
        if file_path:
            if not file_path.lower().endswith(".png"):
                file_path += ".png"
            try:
                success = cv2.imwrite(file_path, self.pap_code_image)
                if success:
                    print(f"PAP Code saved to: {file_path}")
                    QMessageBox.information(
                        self, "Success", f"PAP Code saved to:\n{file_path}"
                    )
                else:
                    raise IOError(
                        f"cv2.imwrite returned False. Check file path/permissions: {file_path}"
                    )
            except Exception as e:
                QMessageBox.critical(
                    self, "Save Error", f"Failed to save PAP code:\n{e}"
                )
                print(f"Error saving PAP code: {e}")

    def setup_decoder_ui(self):
        layout = QVBoxLayout(self.decoder_tab)
        display_layout = QHBoxLayout()

        # PAP Code Input Side
        pap_input_group = QVBoxLayout()
        self.decode_input_label = QLabel("Select PAP Code Image")
        self.decode_input_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.decode_input_label.setMinimumSize(400, 400)
        self.decode_input_label.setStyleSheet(
            "border: 1px solid gray; background-color: #f0f0f0;"
        )
        self.decode_input_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.select_pap_button = QPushButton(
            "Select PAP Code..."
        )  # Selects the single file
        self.select_pap_button.clicked.connect(self.select_pap_for_decode)
        pap_input_group.addWidget(self.decode_input_label)
        pap_input_group.addWidget(self.select_pap_button)

        # Decoded Output Side
        decoded_output_group = QVBoxLayout()
        self.decoded_image_label = QLabel("Decoded Image")
        self.decoded_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.decoded_image_label.setMinimumSize(400, 400)
        self.decoded_image_label.setStyleSheet(
            "border: 1px solid gray; background-color: #f0f0f0;"
        )
        self.decoded_image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.save_decoded_button = QPushButton("Save Decoded Image...")
        self.save_decoded_button.clicked.connect(self.save_decoded_image)
        self.save_decoded_button.setEnabled(False)
        decoded_output_group.addWidget(self.decoded_image_label)
        decoded_output_group.addWidget(self.save_decoded_button)

        display_layout.addLayout(pap_input_group)
        display_layout.addLayout(decoded_output_group)

        self.decode_button = QPushButton("Decode PAP Code")
        self.decode_button.clicked.connect(self.decode_pap_code)
        self.decode_button.setEnabled(False)

        layout.addLayout(display_layout)
        layout.addWidget(self.decode_button)

    def select_pap_for_decode(self):
        """Opens file dialog to select the single PAP code file for decoding."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select PAP Code Image", "", "PNG Images (*.png)"
        )
        if file_path:
            self.pap_code_path = file_path   

            self.decoded_image_no_landmarks = None   

            pap_img_temp = cv2.imread(self.pap_code_path)
            if pap_img_temp is None or pap_img_temp.size == 0:
                QMessageBox.warning(
                    self,
                    "Load Error",
                    f"Could not load PAP code image or image is empty:\n{file_path}",
                )
                self.pap_code_path = None
                self.decode_input_label.clear()
                self.decode_input_label.setText("Load Error")
                self.decode_button.setEnabled(False)
                self.decoded_image_label.clear()
                self.decoded_image_label.setText(
                    "Decoded Image (Combined Chroma Recon)"
                )
                self.decoded_image = None
                self.save_decoded_button.setEnabled(False)
                return

            label_w = max(10, self.decode_input_label.width() - 10)
            label_h = max(10, self.decode_input_label.height() - 10)
            pixmap = cv_image_to_qpixmap(
                pap_img_temp, target_width=label_w, target_height=label_h
            )
            self.decode_input_label.setPixmap(pixmap)

            self.decode_button.setEnabled(True)
            self.decoded_image_label.clear()
            self.decoded_image_label.setText("Decoded Image (Combined Chroma Recon)")
            self.decoded_image = None
            self.save_decoded_button.setEnabled(False)

    def decode_pap_code(self):
        if self.pap_code_path is None:
            QMessageBox.warning(self, "Input Error", "Select PAP code.")
            return
        if not os.path.exists(self.pap_code_path):
            QMessageBox.critical(
                self, "File Error", f"Not found:\n{self.pap_code_path}"
            )
            return
        self.decoded_image = None
        self.decoded_image_no_landmarks = None
        print(f"Decoding PAP (RGB) from: {self.pap_code_path}")
        pap_img = cv2.imread(self.pap_code_path)
        if pap_img is None:
            QMessageBox.critical(
                self, "Load Error", f"Failed read: {self.pap_code_path}"
            )
            return
        decoded_landmarks = []
        reconstructed_bgr = None
        try:
            if len(pap_img.shape) != 3:
                raise ValueError("PAP not 3-channel.")
            pap_b, pap_g, pap_r = cv2.split(pap_img)
            h, w = pap_b.shape
            if h == 0 or w == 0:
                raise ValueError("PAP channels empty.")
            print(f"Split PAP (RGB): {pap_b.shape}")
            print("Decoding QR...")
            try:  # QR decode
                pap_g_uint8 = (
                    pap_g.astype(np.uint8) if pap_g.dtype != np.uint8 else pap_g
                )
                decoded_objects = pyzbar.decode(pap_g_uint8)
                if decoded_objects:
                    decoded_landmarks = json.loads(
                        decoded_objects[0].data.decode("utf-8")
                    )
                if not isinstance(decoded_landmarks, list):
                    decoded_landmarks = []
                print(f"QR decoded: {len(decoded_landmarks)} landmarks.")
            except Exception as qr_err:
                print(f"QR decode/parse error: {qr_err}")
                decoded_landmarks = []
            np.random.seed(SCRAMBLE_SEED_R)
            r_rp = np.random.permutation(h)
            r_cp = np.random.permutation(w)
            np.random.seed(SCRAMBLE_SEED_B)
            b_rp = np.random.permutation(h)
            b_cp = np.random.permutation(w)
            print("Regenerated R & B permutations.")
            print("Unscrambling R and B...")
            unscrambled_r = unscramble_channel(pap_r, r_rp, r_cp)
            if unscrambled_r is None:
                raise ValueError("Failed unscramble R.")
            unscrambled_b = unscramble_channel(pap_b, b_rp, b_cp)
            if unscrambled_b is None:
                raise ValueError("Failed unscramble B.")
            print("Estimating G channel...")
            estimated_g = (
                (unscrambled_r.astype(np.uint16) + unscrambled_b.astype(np.uint16)) // 2
            ).astype(np.uint8)
            print("Merging B, Est.G, R...")
            if not (
                unscrambled_b.shape
                == estimated_g.shape
                == unscrambled_r.shape
                == (h, w)
            ):
                raise ValueError("Shape mismatch before merge.")
            reconstructed_bgr_no_lm = cv2.merge(
                [unscrambled_b, estimated_g, unscrambled_r]
            )
            self.decoded_image_no_landmarks = reconstructed_bgr_no_lm.copy()
            print("Stored decoded (no lm) for SSIM.")
            reconstructed_bgr = reconstructed_bgr_no_lm
            if decoded_landmarks:   
                print(f"Drawing {len(decoded_landmarks)} landmarks...")
                if not reconstructed_bgr.flags["WRITEABLE"]:
                    reconstructed_bgr = reconstructed_bgr.copy()
                for x, y in decoded_landmarks:
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(reconstructed_bgr, (x, y), 2, (0, 255, 0), -1)
            self.decoded_image = reconstructed_bgr
            label_w = max(10, self.decoded_image_label.width() - 10)
            label_h = max(10, self.decoded_image_label.height() - 10)
            pixmap = cv_image_to_qpixmap(self.decoded_image, label_w, label_h)
            self.decoded_image_label.setPixmap(pixmap)
            self.save_decoded_button.setEnabled(True)
            print("Decoding Complete (RGB, Est. G).")
        except Exception as e:
            QMessageBox.critical(self, "Decoding Error", f"Error: {e}")
            print(f"Error during RGB decoding: {e}")
            self.decoded_image = None
            self.decoded_image_no_landmarks = None
            self.decoded_image_label.clear()
            self.decoded_image_label.setText("Decoding Failed")
            self.save_decoded_button.setEnabled(False)

    def save_decoded_image(self):
        if self.decoded_image is None or self.decoded_image.size == 0:
            QMessageBox.warning(self, "Save Error", "No valid decoded image to save.")
            return

        default_name = "decoded_combined_chroma.png"
        if self.pap_code_path:
            base = os.path.splitext(os.path.basename(self.pap_code_path))[0]
            base = base.replace("_pap_combo", "").replace("_pap", "")
            default_name = f"{base}_decoded_combo.png"
        elif self.input_image_path:
            base = os.path.splitext(os.path.basename(self.input_image_path))[0]
            default_name = f"{base}_decoded_combo.png"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Decoded Image (Combined Chroma)",
            default_name,
            "PNG Image (*.png)",
        )
        if file_path:
            if not file_path.lower().endswith(".png"):
                file_path += ".png"
            try:
                success = cv2.imwrite(file_path, self.decoded_image)
                if success:
                    print(f"Decoded image saved to: {file_path}")
                    QMessageBox.information(
                        self, "Success", f"Decoded image saved to:\n{file_path}"
                    )
                else:
                    raise IOError(f"cv2.imwrite returned False saving decoded image.")
            except Exception as e:
                QMessageBox.critical(
                    self, "Save Error", f"Failed to save decoded image:\n{e}"
                )
                print(f"Error saving decoded image: {e}")

