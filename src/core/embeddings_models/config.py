"""Модуль для настройки эмбеддинг моделей"""

__all__ = ("BaseEmbeddingConfig",)
import re
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from configs.const import BATCH_SIZE_FOR_VECTOR_ENCODER, STANDARD_PATH_TO_MODEL
from configs.custom_error import HFValidateTokenError, ModelExistsOnHFError


class BaseEmbeddingConfig(BaseModel):
    """Класс конфигуратор для валидации данных необходимых для создания эмбеддинговых моделей"""

    is_private_model: Annotated[
        bool,
        Field(
            title="Это приватная модель из HF?",
            description="Если да, то потребуется токен доступа к HF",
        ),
    ] = False

    hf_token: Annotated[
        str | None,
        Field(
            title="Токен для доступа к HF",
            description="HF токен для доступа к приватным моделям."
            "\nПолучить токен можно тут https://huggingface.co/settings/tokens",
        ),
    ] = None

    models_name_from_hf: Annotated[
        str,
        Field(
            title="HF имя модели в формате  'владелец/имя'",
            description="Модель из HF hub. Вам необходимо перенести название идентично",
        ),
    ]

    local_path_for_downloads: Annotated[
        str | Path | None,
        Field(
            title="Путь до папки, где будет храниться модель",
            description="Путь до папки с моделью."
            "Она будет загружена в папку автоматически при первом запуске.",
        ),
    ] = None

    trust_remote_code: Annotated[
        bool,
        Field(
            title="Разрешить модели выполнять код",
            description=(
                "Если True, при загрузке модели из HuggingFace будет выполнен пользовательский "
                "Python-код автора модели (удаленный код). "
                "Требуется для продвинутых или экспериментальных моделей с пользовательской логикой "
                "загрузки, обычно в `model.py` или `configuration.py`. "
                "Включение создает риски безопасности, так как автор может внедрить произвольный код, "
                "который выполнится в вашем окружении. "
                "Используйте True только для доверенных источников, никогда для непроверенных моделей. "
                "В большинстве случаев оставляйте False."
            ),
        ),
    ] = False

    batch_size: Annotated[
        int,
        Field(
            title="Размер батча",
            description="Количество образцов, обрабатываемых вместе за один прямой/обратный проход.",
        ),
    ] = BATCH_SIZE_FOR_VECTOR_ENCODER

    show_progress_bar: Annotated[
        bool,
        Field(
            title="Показывать прогресс-бар в работе",
            description="Если True, прогресс-бар будет отображаться во время "
            "длительных операций, таких как загрузки, обучение или пакетная обработка. "
            "Помогает визуализировать процесс, отслеживать оставшееся время и улучшить "
            "пользовательский опыт. Для тихих или скриптовых запусков установите False.",
        ),
    ] = True

    device: Annotated[
        Literal["CPU", "GPU", "cpu", "gpu", "cuda", "CUDA"] | None,
        Field(
            title="Тип устройства",
            description="Выберите устройство обработки: 'CPU' или 'GPU'. "
            "Если установлено None, приложение автоматически определит и выберет "
            "наиболее подходящее доступное устройство.",
        ),
    ] = None

    @classmethod
    @field_validator("device")
    def validate_device(cls, value: str) -> Literal["cpu", "cuda"] | None:
        """
        Проверяет, что значение device поддерживается SentenceTransformer.
        Корректно проходит: 'cpu', 'gpu', 'cuda', 'CPU', 'GPU', None.
        Все остальные — ValueError.

        Args:
            value (str): поле для проверки

        Returns:
            Проверенное значение

        """
        if value is None:
            return value

        allowed = {"cpu", "CPU", "gpu", "GPU", "cuda", "CUDA"}
        normalized = str(value).lower()
        if normalized not in allowed:
            msg = (
                f"device должен быть 'cpu' или 'cuda' (gpu автоматически определяется как 'cuda'). "
                f"Текущее значение: '{value}'."
            )
            raise ValueError(msg)

        if normalized in {"gpu", "cuda"}:
            return "cuda"
        return "cpu"

    @classmethod
    @field_validator("local_path_for_downloads")
    def validate_local_path(cls, value: str | Path) -> Path | None:
        """

        Args:
            value (str|Path): путь пользователя до папки с моделью

        Returns:
            Возвращает корректный Path пути

        Raises:
            ValueError если путь не существует

        """
        if value:
            path = Path(value) if isinstance(value, str) else value
            if not path.parent.exists():
                msg = "Папка для загрузки модели, не существует. "
                raise ValueError(msg)
            return path

        return value

    @classmethod
    @field_validator("models_name_from_hf")
    def validate_hf_model_name(cls, value: str) -> str:
        """
        Проверяет модель на соответствие формату организация/название.

        Args:
            value (str): Имя модели в формате organization/model-name.

        Returns:
            Валидное имя модели.

        Raises:
            ModelExistsOnHFError: Если имя модели не соответствует формату.

        """
        pattern = r"^[\w\-]+\/[\w\-\.]+$"
        if not re.match(pattern, value):
            msg = (
                f"Имя модели должно соответствовать формату 'организация/название'. "
                f"Получено: {value}"
            )
            raise ModelExistsOnHFError(msg)
        return value

    @classmethod
    @field_validator("hf_token")
    def validate_token_format(cls, value: str | None) -> str | None:
        """
        Проверяет формат токена HF (начинается с hf_)

        Args:
            value (str): полученный токен

        Returns:
            Возвращает исходное значение или сваливается в ошибку

        """
        if value is not None and not re.match(r"^hf_[A-Za-z0-9]{34,}$", value):
            msg = (
                "Токен не соответствует стандарту HF. "
                'Токен должен начинаться с "hf_" и содержать 34 символа'
            )
            raise HFValidateTokenError(msg)
        return value

    @model_validator(mode="after")
    def validate_token_for_private(self) -> Self:
        """
        Проверяет наличие токена для приватных моделей

        Returns:
            Возвращает исходную модель если условия выполнены. В случае провала сваливается в ошибку.

        """
        if self.is_private_model and not self.hf_token:
            msg = (
                "Для приватной модели требуется HF токен. "
                "Его можно получить тут: https://huggingface.co/settings/tokens"
            )
            raise HFValidateTokenError(msg)
        return self

    @model_validator(mode="after")
    def create_standard_folder(self) -> None:
        """Если папка для модели не задана, то создается и используется стандартная."""
        if self.local_path_for_downloads is None:
            Path.mkdir(STANDARD_PATH_TO_MODEL, exist_ok=True)
            self.local_path_for_downloads = STANDARD_PATH_TO_MODEL
