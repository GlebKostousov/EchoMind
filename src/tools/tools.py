"""Модуль для сбора функций - помощников"""

__all__ = ("timer",)
import time
from collections.abc import Callable
from functools import wraps

from src.tools.self_logger import setup_logger

logger = setup_logger(__name__)


def timer[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Декоратор замера времени выполнения"""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        time_elapsed: str = f"{end - start:.4f}"
        logger.debug("%s: %s сек", func.__name__, time_elapsed)
        return result

    return wrapper
