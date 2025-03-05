import logging
import os

# Получаем уровень логирования из переменной окружения
log_level_str = os.getenv('LOG_LEVEL', 'ERROR').upper()
log_levels = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}
log_level = log_levels.get(log_level_str, logging.DEBUG)

# Создаем объект логгера
logger = logging.getLogger('benchmark_logger')
logger.setLevel(log_level)

# Создаем форматтер
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Добавляем обработчик для вывода в консоль
console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Получаем имя файла лога из переменной окружения
log_file = os.getenv('LOG_FILE')

if log_file:
    # Создаем обработчик для записи в файл
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
