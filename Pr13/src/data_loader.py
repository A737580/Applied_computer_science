import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(root_dir: str, genre: str, img_size: tuple = (256, 256)):
    """
    Загружает изображения из папок <root>/<genre>/<author>/*.jpg,
    предобрабатывает их и возвращает данные для обучения.

    Args:
        root_dir (str): Корневая директория с датасетом.
        genre (str): Название жанра для загрузки.
        img_size (tuple, optional): Целевой размер изображений (ширина, высота).
                                     Defaults to (256, 256).

    Returns:
        tuple: Кортеж (X_imgs, y, class_names, class_to_idx)
            X_imgs (np.ndarray): Массив изображений (N, H, W, C), RGB, float [0, 1].
            y (np.ndarray): Массив меток классов (N,).
            class_names (list): Список имен авторов (классов).
            class_to_idx (dict): Словарь для преобразования имени автора в индекс.
    Raises:
        FileNotFoundError: Если директория жанра не найдена.
    """
    genre_path = Path(root_dir) / genre
    if not genre_path.is_dir():
        raise FileNotFoundError(f"Директория жанра не найдена: {genre_path}")

    X_imgs = []
    y = []
    class_names = []
    class_to_idx = {}
    current_idx = 0

    author_dirs = sorted([d for d in genre_path.iterdir() if d.is_dir()])
    logging.info(f"Найдено {len(author_dirs)} авторов в жанре '{genre}': {[d.name for d in author_dirs]}")

    for author_dir in author_dirs:
        author_name = author_dir.name
        if author_name not in class_to_idx:
            class_names.append(author_name)
            class_to_idx[author_name] = current_idx
            current_idx += 1

        author_idx = class_to_idx[author_name]
        logging.info(f"Загрузка изображений для автора: {author_name} (индекс {author_idx})")

        # Ищем файлы изображений с разными расширениями
        image_files = list(author_dir.glob('*.jpg')) + \
                      list(author_dir.glob('*.jpeg')) + \
                      list(author_dir.glob('*.png'))

        if not image_files:
            logging.warning(f"Не найдено изображений для автора {author_name} в {author_dir}")
            continue

        for img_path in tqdm(image_files, desc=f"Автор {author_name}"):
            try:
                # Загрузка изображения в BGR
                img = cv2.imread(str(img_path))
                if img is None:
                    logging.warning(f"Не удалось загрузить изображение: {img_path}")
                    continue

                # Ресайз
                img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)

                # Конвертация BGR -> RGB
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

                # Нормализация [0, 1]
                img_normalized = img_rgb.astype(np.float32) / 255.0

                X_imgs.append(img_normalized)
                y.append(author_idx)
            except Exception as e:
                logging.error(f"Ошибка обработки файла {img_path}: {e}")

    if not X_imgs:
        logging.error(f"Не удалось загрузить ни одного изображения для жанра '{genre}'.")
        return np.array([]), np.array([]), [], {}

    X_imgs = np.array(X_imgs)
    y = np.array(y)

    logging.info(f"Загружено {len(X_imgs)} изображений.")
    logging.info(f"Размерность массива изображений X_imgs: {X_imgs.shape}")
    logging.info(f"Размерность массива меток y: {y.shape}")
    logging.info(f"Имена классов: {class_names}")
    logging.info(f"Словарь классов: {class_to_idx}")

    return X_imgs, y, class_names, class_to_idx
