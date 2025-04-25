import joblib
import logging
from pathlib import Path

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- build_classifier ---
def build_classifier(hidden_layer_sizes=(128,), alpha=1e-4, random_state=42, max_iter=500) -> Pipeline:
    scaler = StandardScaler()
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha,
                        early_stopping=True, validation_fraction=0.1,
                        n_iter_no_change=10, max_iter=max_iter, solver='adam',
                        learning_rate_init=0.001, random_state=random_state, verbose=False) # verbose=False по умолчанию
    model_pipeline = Pipeline([('scaler', scaler), ('mlp', mlp)])
    logging.info("Конвейер StandardScaler + MLPClassifier создан.")
    logging.info(f"Параметры MLP: hidden_layers={hidden_layer_sizes}, alpha={alpha}, early_stopping=True, random_state={random_state}")
    return model_pipeline

# --- save_model ---
def save_model(model: Pipeline, class_names: list, filepath: str, training_params: dict = None):
    """
    Сохраняет обученную модель, имена классов и параметры обучения в файл.

    Args:
        model (Pipeline): Обученный конвейер Sklearn.
        class_names (list): Список имен классов (авторов).
        filepath (str): Путь для сохранения файла (.joblib).
        training_params (dict, optional): Словарь с параметрами, использованными
                                           при обучении (например, 'random_state',
                                           'test_size', 'img_size'). Defaults to None.
    """
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        save_data = {
            'model': model,
            'class_names': class_names,
            'training_params': training_params if training_params else {}
        }
        joblib.dump(save_data, filepath)
        logging.info(f"Модель, имена классов и параметры сохранены в: {filepath}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении модели в {filepath}: {e}")
        raise

# --- load_model ---
def load_model(filepath: str) -> tuple:
    """
    Загружает модель, имена классов и параметры обучения из файла joblib.

    Args:
        filepath (str): Путь к сохраненному файлу (.joblib).

    Returns:
        tuple: Кортеж (model, class_names, training_params).
            model (Pipeline): Загруженный конвейер Sklearn.
            class_names (list): Загруженный список имен классов.
            training_params (dict): Загруженный словарь параметров обучения.
    """
    try:
        if not Path(filepath).is_file():
             raise FileNotFoundError(f"Файл модели не найден: {filepath}")
        loaded_data = joblib.load(filepath)
        model = loaded_data['model']
        class_names = loaded_data['class_names']
        # Загружаем параметры, если они есть, иначе пустой словарь
        training_params = loaded_data.get('training_params', {})

        logging.info(f"Модель, имена классов и параметры загружены из: {filepath}")

        # Проверки типов 
        if not isinstance(model, Pipeline):
             logging.warning("Загруженный объект 'model' не является sklearn.pipeline.Pipeline")
        if not isinstance(class_names, list):
             logging.warning("Загруженный объект 'class_names' не является списком")
        if not isinstance(training_params, dict):
             logging.warning("Загруженный объект 'training_params' не является словарем")
             training_params = {} 

        return model, class_names, training_params
    except FileNotFoundError as e:
         logging.error(e)
         raise e
    except KeyError as e:
        logging.error(f"Ошибка загрузки: отсутствует ключ {e} в файле {filepath}.")
        raise Exception(f"Неверный формат файла модели: отсутствует ключ {e}")
    except Exception as e:
        logging.error(f"Ошибка при загрузке модели из {filepath}: {e}")
        raise
