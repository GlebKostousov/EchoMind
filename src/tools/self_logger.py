"""Модуль отвечает за кастомную настройку логирования"""

__all__ = ("setup_logger",)

import logging
import sys
import traceback
from logging import Logger
from textwrap import fill

from colorama import Fore, init

# Инициализируем colorama
init(autoreset=True)
COLORS = {
    "debug": Fore.BLUE,  # Синий
    "info": Fore.CYAN,  # Светло-голубой
    "warning": Fore.YELLOW,  # Пастельный жёлтый
    "error": Fore.LIGHTRED_EX,  # Светло-красный
    "critical": Fore.LIGHTMAGENTA_EX,  # Пурпурный
}


class CustomFormatter(logging.Formatter):
    """Кастомный форматер для логов с мягкими цветами и красивым выравниванием."""

    def __init__(
        self,
        name: str,
        deb: bool,
    ) -> None:
        super().__init__()
        self.deb = deb
        self.name = name

    def format(self, record: logging.LogRecord) -> str:
        # Генерируем название лога, согласно файлу его запуска
        # module_name = self.generate_log_name_for_user(record)

        ex_msg = record.exc_info
        ex_traceback = traceback.format_exc()

        # Цвет уровня логирования
        color = COLORS.get(record.levelname.lower(), Fore.WHITE)
        # Цвет msg
        color_white = Fore.WHITE

        # Форматируем сообщение для красивого переноса строк (ширина 110 символов),
        # отступ при переносе 36
        message = fill(record.getMessage(), width=110, subsequent_indent=" " * 36)

        return (
            f"{color}{self.name:<30}  {color_white}{message}\n{ex_traceback!s}"
            if ex_msg
            else f"{color}{self.name:<30}  {color_white}{message}"
        )

    # def generate_log_name_for_user(self, record: logging.LogRecord) -> str:
    #     module_name = os.path.basename(record.pathname)
    #     if self.deb:
    #         return module_name
    #
    #     if "BI" in module_name:
    #         module_name = "Загрузка данных из BI"
    #
    #     elif "Employee" in module_name:
    #         module_name = "Создание сотрудника"
    #
    #     elif "Payroll" in module_name:
    #         module_name = "Подготовка расчета сотрудника"
    #
    #     elif "Calculate" in module_name:
    #         module_name = "Расчет сотрудника"
    #
    #     elif "Division" in module_name:
    #         module_name = "Модуль подразделения"
    #
    #     else:
    #         for key, val in LAYER_NAMES.items():
    #             module_name = module_name.replace(key, val)
    #     return module_name


def setup_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """Настраивает логгер с мягкими цветами и выравниванием."""
    log: Logger = logging.getLogger(name)
    log_level = logging.DEBUG
    debug = True if log_level == logging.DEBUG else False

    if not log.hasHandlers():
        log.setLevel(log_level)

        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(CustomFormatter(deb=debug, name=name))
        log.addHandler(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelness)s - %(message)s"),
            )
            log.addHandler(file_handler)

    return log


if __name__ == "__main__":
    logger = setup_logger(__name__)
    msg = f"Log level: {logger.getEffectiveLevel()}"
    logger.info(msg)

    logger.debug("Инициализация системы завершена успешно.")
    logger.info("Загрузка BI отчетов по каждой структуре компании")
    logger.warning("Следующие файлы не соответствуют шаблонам: 20250221_132101.xlsx")
    logger.error("Ошибка при загрузке данных! Проверьте исходные файлы.")
    logger.critical("Критическая ошибка: приложение не может продолжить работу.")
