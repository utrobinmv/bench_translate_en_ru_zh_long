import os
import time
import pandas as pd

def create_lock_file(lock_file_path):
    """Создает файл блокировки."""
    try:
        # Попытка создать файл блокировки
        with open(lock_file_path, 'x') as lock_file:
            lock_file.write("locked")  # Можно записать что-то в файл
        return True
    except FileExistsError:
        # Файл блокировки уже существует
        return False

def remove_lock_file(lock_file_path):
    """Удаляет файл блокировки."""
    if os.path.exists(lock_file_path):
        os.remove(lock_file_path)

def append_to_file(filename, data):
    """Записывает данные в файл с использованием блокировки."""
    lock_file_path = filename + ".lock"

    # Ожидание, пока файл блокировки не будет удален
    while os.path.exists(lock_file_path):
        time.sleep(0.1)  # Ждем 100 мс перед следующей попыткой

    # Создаем файл блокировки
    if not create_lock_file(lock_file_path):
        raise RuntimeError("Не удалось создать файл блокировки")

    try:
        # Записываем данные в файл
        with open(filename, 'a') as file:
            file.write(data + '\n')
    finally:
        # Удаляем файл блокировки
        remove_lock_file(lock_file_path)

def save_json_to_file(filename, list_filenames):
    """Записывает данные в файл с использованием блокировки."""
    lock_file_path = filename + ".lock"

    # Ожидание, пока файл блокировки не будет удален
    while os.path.exists(lock_file_path):
        time.sleep(0.1)  # Ждем 100 мс перед следующей попыткой

    # Создаем файл блокировки
    if not create_lock_file(lock_file_path):
        raise RuntimeError("Не удалось создать файл блокировки")

    try:
        # Записываем данные в файл
        df = pd.read_json(filename, lines=True, orient="records")
        df.loc[df['filename'].isin(list_filenames),"metrics"] = True
        df.to_json(filename, lines=True, orient="records")
    finally:
        # Удаляем файл блокировки
        remove_lock_file(lock_file_path)