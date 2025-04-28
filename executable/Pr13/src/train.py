import argparse
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import data_loader
import feature_extractor
import model as model_utils 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_hidden_layers(layers_str):
    """Преобразует строку '128' или '128,64' в кортеж целых чисел."""
    if not layers_str:
        return (128,) 
    try:
        return tuple(map(int, layers_str.split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError("hidden_layers должен быть строкой чисел, разделенных запятыми (например, '128' или '128,64')")

def main():
    parser = argparse.ArgumentParser(description="Обучение MLP классификатора для атрибуции картин.")
    parser.add_argument('--root_dir', type=str, required=True, help="Корневая директория датасета.")
    parser.add_argument('--genre', type=str, required=True, help="Жанр для обучения (название подпапки в root_dir).")
    parser.add_argument('--output_model_path', type=str, default='./models/painter_model.joblib', help="Путь для сохранения обученной модели.")
    parser.add_argument('--img_size', type=int, default=256, help="Размер стороны изображения (будет img_size x img_size).")
    parser.add_argument('--test_size', type=float, default=0.2, help="Доля данных для тестового набора.")
    parser.add_argument('--hidden_layers', type=parse_hidden_layers, default='128', help="Размеры скрытых слоев MLP (например, '128' или '128,64').")
    parser.add_argument('--alpha', type=float, default=1e-4, help="Параметр L2 регуляризации MLP.")
    parser.add_argument('--random_state', type=int, default=42, help="Random state для воспроизводимости.")
    parser.add_argument('--max_iter', type=int, default=500, help="Макс. число итераций обучения MLP.")


    args = parser.parse_args()

    img_dim = (args.img_size, args.img_size)
    output_path = Path(args.output_model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True) 

    logging.info("--- Начало процесса обучения ---")
    logging.info(f"Параметры: {args}")

    # 1. Загрузка данных
    logging.info(f"1. Загрузка данных из: {args.root_dir}/{args.genre}")
    try:
        X_imgs, y, class_names, _ = data_loader.load_data(args.root_dir, args.genre, img_size=img_dim)
    except FileNotFoundError as e:
        logging.error(f"Ошибка загрузки данных: {e}")
        return 
    except Exception as e:
        logging.error(f"Неожиданная ошибка при загрузке данных: {e}")
        return

    if X_imgs.size == 0:
        logging.error("Нет данных для обучения. Проверьте путь к данным и содержимое папок.")
        return

    n_classes = len(class_names)
    if n_classes < 2:
        logging.error(f"Найдено менее 2 классов ({n_classes}). Обучение невозможно.")
        return

    logging.info(f"Загружено {len(X_imgs)} изображений, {n_classes} классов.")

    # 2. Извлечение признаков
    logging.info("2. Извлечение признаков...")
    try:
        X_features = feature_extractor.extract_features(X_imgs)
    except Exception as e:
        logging.error(f"Ошибка при извлечении признаков: {e}")
        return

    # 3. Разделение на обучающий и тестовый наборы
    # Мы используем test_size для разделения. MLP сам выделит валидационный набор
    # из X_train для early stopping.
    logging.info(f"3. Разделение данных (test_size={args.test_size})...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y 
        )
        logging.info(f"Размер обучающего набора: {X_train.shape[0]} ({X_train.shape[1]} признаков)")
        logging.info(f"Размер тестового набора: {X_test.shape[0]}")
    except ValueError as e:
        logging.error(f"Ошибка при разделении данных: {e}. Возможно, недостаточно данных для стратификации или указан некорректный test_size.")
        return


    # 4. Создание и обучение модели
    logging.info("4. Создание и обучение модели MLP...")
    model = model_utils.build_classifier(
        hidden_layer_sizes=args.hidden_layers,
        alpha=args.alpha,
        random_state=args.random_state,
        max_iter=args.max_iter
    )

    try:
        logging.info("Начало обучения (Pipeline)...")
        model.fit(X_train, y_train)
        logging.info("Обучение завершено.")
        if hasattr(model.named_steps['mlp'], 'loss_curve_'):
             logging.info(f"Количество итераций до остановки: {model.named_steps['mlp'].n_iter_}")
             logging.info(f"Финальное значение потерь: {model.named_steps['mlp'].loss_}")
    except Exception as e:
        logging.error(f"Ошибка во время обучения модели: {e}")
        return

    # 5. Оценка модели на тестовом наборе
    logging.info("5. Оценка модели на тестовом наборе...")
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

        logging.info(f"\n--- Отчет по классификации (на тестовых данных) ---")
        logging.info(f"Точность (Accuracy): {accuracy:.4f}")
        logging.info(f"\n{report}")
        logging.info("----------------------------------------------------")

    except Exception as e:
        logging.error(f"Ошибка при оценке модели: {e}")

    # 6. Сохранение модели
    logging.info(f"6. Сохранение модели в {output_path}...")
    try:
        model_utils.save_model(model, class_names, str(output_path))
    except Exception as e:
        logging.error(f"Не удалось сохранить модель: {e}")

    logging.info("--- Процесс обучения завершен ---")
