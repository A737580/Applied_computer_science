import numpy as np
import cv2
from skimage.feature import hog
from sklearn.preprocessing import normalize
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_hog(image_rgb: np.ndarray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9, block_norm='L2-Hys') -> np.ndarray:
    """Извлекает HOG признаки из одного RGB изображения."""
    # HOG требует одноканальное изображение
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    # Конвертируем в uint8, если нужно (skimage ожидает [0, 255] или [0, 1])
    if image_gray.max() <= 1.0:
         image_gray = (image_gray * 255).astype(np.uint8)

    features = hog(image_gray, orientations=orientations,
                   pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block,
                   block_norm=block_norm,
                   visualize=False, 
                   feature_vector=True) 
    return features

def extract_color_histogram(image_rgb: np.ndarray, bins=8, hist_range=(0, 1)) -> np.ndarray:
    """Извлекает L2-нормализованную RGB гистограмму из одного изображения."""
    hist_features = []
    for i in range(3): # Для каждого канала R, G, B
        channel_hist = cv2.calcHist([image_rgb], [i], None, [bins], hist_range)
        hist_features.extend(channel_hist.flatten())

    hist_features = np.array(hist_features)
    # L2 нормализация
    # Добавляем reshape(1, -1), так как normalize ожидает 2D массив
    normalized_hist = normalize(hist_features.reshape(1, -1), norm='l2').flatten()
    return normalized_hist

def extract_gabor(image_rgb: np.ndarray, frequencies=(0.1, 0.3, 0.5, 0.7), thetas_deg=(0, 45, 90, 135), ksize=(9, 9), sigma=1.0, gamma=0.5, psi=0) -> np.ndarray:
    """Извлекает признаки на основе фильтров Габора (среднее и СКО отклика)."""
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
     # Нормализуем серое изображение в [0, 1] для фильтрации
    if image_gray.max() > 1:
        image_gray = image_gray.astype(np.float32) / 255.0

    gabor_features = []
    # Конвертируем углы в радианы для OpenCV
    thetas_rad = [np.pi / 180. * t for t in thetas_deg]

    for freq in frequencies:
        ksize_adjusted = tuple(k + 1 if k % 2 == 0 else k for k in ksize)

        for theta in thetas_rad:
            kernel = cv2.getGaborKernel(ksize_adjusted, sigma, theta, 1.0/freq, gamma, psi, ktype=cv2.CV_32F)
            filtered_img = cv2.filter2D(image_gray, cv2.CV_32F, kernel)
            mean_val = np.mean(filtered_img)
            std_val = np.std(filtered_img)
            gabor_features.extend([mean_val, std_val])

    return np.array(gabor_features)


def extract_features(images: np.ndarray) -> np.ndarray:
    """
    Извлекает HOG, цветовые гистограммы и Gabor признаки из массива изображений
    и объединяет их в единый вектор признаков для каждого изображения.

    Args:
        images (np.ndarray): Массив изображений (N, H, W, C), RGB, float [0, 1].

    Returns:
        np.ndarray: Массив векторов признаков (N, total_feature_dim).
    """
    all_features = []
    logging.info(f"Начало извлечения признаков для {len(images)} изображений...")

    for img in tqdm(images, desc="Извлечение признаков"):
        # 1. HOG
        hog_feats = extract_hog(img)

        # 2. Color Histogram
        color_feats = extract_color_histogram(img, bins=8) # 8 бинов на канал = 24

        # 3. Gabor Filters
        # Частоты [0.1, 0.3, 0.5, 0.7], углы [0°, 45°, 90°, 135°]
        # 4 частоты * 4 угла = 16 фильтров
        # По каждому фильтру среднее и СКО = 16 * 2 = 32 признака
        gabor_feats = extract_gabor(img, frequencies=[0.1, 0.3, 0.5, 0.7], thetas_deg=[0, 45, 90, 135])

        # 4. Объединение признаков
        combined_features = np.concatenate((hog_feats, color_feats, gabor_feats))
        all_features.append(combined_features)

    features_array = np.array(all_features)
    logging.info(f"Извлечение признаков завершено. Размерность массива признаков: {features_array.shape}")
    if len(images) > 0:
        logging.info(f"Размерность HOG: {len(hog_feats)}")
        logging.info(f"Размерность Color Histogram: {len(color_feats)}")
        logging.info(f"Размерность Gabor: {len(gabor_feats)}")
        logging.info(f"Общая размерность признаков: {features_array.shape[1]}")


    return features_array