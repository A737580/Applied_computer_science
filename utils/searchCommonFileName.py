import os
import sys

def get_file_set(directory):
    """
    Возвращает множество с именами файлов из указанной директории.
    Если директория не существует или недоступна, возвращает пустое множество.
    """
    try:
        return set(os.listdir(directory))
    except Exception as e:
        print(f"Ошибка при чтении директории {directory}: {e}")
        return set()

def main():
    # if len(sys.argv) != 5:
    #     print("Использование: python script.py <директория1> <директория2> <директория3> <директория4>")
    #     sys.exit(1)

    # dir1, dir2, dir3, dir4 = sys.argv[1:5]

    # Создаем множества файлов для каждой директории
    set1 = get_file_set(r"C:\Users\mxm\Desktop\итог\тонко")
    set2 = get_file_set(r"C:\Users\mxm\Desktop\итог\тонко")
    set3 = get_file_set(r"C:\Users\mxm\Desktop\итог\шире")
    set4 = get_file_set(r"C:\Users\mxm\Desktop\итог\шире")

    # Вычисляем пересечение множеств
    common_files = set1 & set2 & set3 & set4

    print("Файлы, присутствующие во всех четырех директориях:")
    for filename in common_files:
        print(filename)

if __name__ == "__main__":
    main()
