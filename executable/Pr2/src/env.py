import os
import inspect
def get_caller_abs_path():
    # Получаем стек вызовов; [0] — текущая функция, [1] — её вызывающий
    caller_frame = inspect.stack()[1]
    caller_filename = caller_frame.filename  # Имя файла вызывающего кода
    return os.path.dirname(os.path.abspath(caller_filename))

src_prefix = get_caller_abs_path()
home_prefix = os.getcwd()