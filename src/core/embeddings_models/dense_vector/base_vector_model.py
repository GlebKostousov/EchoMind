"""Модуль для создания базового класса будущих ембеддинг векторов"""

import re
from pathlib import Path
from typing import Annotated, Literal, Self

from huggingface_hub import repo_exists, snapshot_download
from pydantic import BaseModel, Field, field_validator, model_validator
from src.tools.self_logger import setup_logger

from configs.custom_error import HFValidateTokenError, ModelExistsOnHFError

# TODO: Вынеси в модуль констант
# TODO: Сделай автосортировку импортов в ruff
MODEL_CONFIG_TO_EXISTS_CHECK = "config_sentence_transformers.json"
MODULES_JSON_TO_EXISTS_CHECK = "modules.json"
BATCH_SIZE_FOR_VECTOR_ENCODER = 32

logger = setup_logger(__name__)


# TODO: для стандартного пути создай функцию создания папки
class BaseVectorModel(BaseModel):
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
        str | Path,
        Field(
            title="Путь до папки, где будет храниться модель",
            description="Путь до папки с моделью."
            "Она будет загружена в папку автоматически при первом запуске.",
        ),
    ]

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
        Literal["CPU", "GPU"] | None,
        Field(
            title="Тип устройства",
            description="Выберите устройство обработки: 'CPU' или 'GPU'. "
            "Если установлено None, приложение автоматически определит и выберет "
            "наиболее подходящее доступное устройство.",
        ),
    ] = None

    @classmethod
    @field_validator("local_path_for_downloads")
    def validate_local_path(cls, value: str | Path) -> Path:
        """

        Args:
            value (str|Path): путь пользователя до папки с моделью

        Returns:
            Возвращает корректный Path пути

        Raises:
            ValueError если путь не существует

        """
        path = Path(value) if isinstance(value, str) else value
        if not path.parent.exists():
            msg = "Папка для загрузки модели, не существует. "
            raise ValueError(msg)
        return path

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


class EchoMindEmbedding:
    """Базовый класс для создания любой модели эмбеддингов"""

    def __init__(self) -> None:
        """Инициализатор класса"""


# TODO: поменяй входящие объекты на валидированные. с str на класс от BaseModel
def _check_existing_model_repo(model_name: str, token: str | None = None) -> bool:
    """

    Args:
        model_name (str): Название модели разделенное `/`.
        token (str):
                Токен для доступа к приватной модели HF
                (https://huggingface.co/docs/huggingface_hub/quick-start#authentication).
                Для неприватной модели используем None

    Returns:
        True, если модель существует

    """
    result = repo_exists(
        model_name,
        token=token,
    )
    if not result:
        msg = (
            f"Модель: {model_name} не существует в HF.\n"
            f"Убедитесь, что вы ввели корректное название"
        )
        logger.error(msg)

    return result


# TODO: поменяй входящие объекты на валидированные. с str на класс от BaseModel
def _validate_exists_local_model(
    local_path: str | Path,
    model_name: str,
) -> bool:
    """

    Args:
        local_path (str | Path): Путь, где хранится / будет храниться модель эмбеддинга
        model_name (str): Название модели для красивых логов

    Returns:
        bool: Если True, значит модель уже есть. Если False, то модель отсутствует.

    """
    model_dir = Path(local_path)
    required_files = [
        model_dir / MODULES_JSON_TO_EXISTS_CHECK,
        model_dir / MODEL_CONFIG_TO_EXISTS_CHECK,
    ]
    result = all(file.exists() for file in required_files)
    if not result:
        msg = f"Отсутствует скачанная модель {model_name} по пути: {local_path}"
        logger.warning(msg)

    return result


# TODO: поменяй входящие объекты на валидированные. с str на класс от BaseModel
def download_model_for_local_path(
    model_name_from_hf: str,
    path_to_download_model: str | Path,
    token: str | None = None,
) -> None:
    """
    Функция сохранения модели в локальную папку
    Args:
        token (Optional[str]): токен для доступа к приватной модели HF
        model_name_from_hf (str): название модели с сайта HF
        path_to_download_model (str|Path): Путь или строка пути до папки,
        где будет храниться модель

    Returns:
        None

    """
    if not _check_existing_model_repo(
        model_name=model_name_from_hf,
        token=token,
    ):
        msg = (
            f"Модель: {model_name_from_hf} не существует в HF.\n"
            'Убедитесь, что название модели имеет форму "владелец/модель", '
            'или "компания/модель"'
        )
        raise ModelExistsOnHFError(msg)

    if _validate_exists_local_model(
        local_path=path_to_download_model,
        model_name=model_name_from_hf,
    ):
        msg = f"Модель {model_name_from_hf} уже существует локально"
        logger.info(msg)

    else:
        msg = f"Начинаю загрузку модели {model_name_from_hf}"
        logger.info(msg)
        local_path = Path(path_to_download_model)

        snapshot_download(
            repo_id=model_name_from_hf,
            local_dir=local_path,
            cache_dir=local_path,
        )
        msg = f"Модель {model_name_from_hf} загружена по пути: {local_path}"
        logger.info(msg)


if __name__ == "__main__":
    frida_folder: str = "frida"
    hf_models_path = Path(
        r"D:\PythonProject\EchoMind\src\core\embeddings_models\local_file_embedding_models",
    )
    hf_models_path_openvino = Path(
        r"D:\PythonProject\EchoMind\src\core\embeddings_models\local_file_embedding_models\openvino",
    )
    model_name_frida = "ai-forever/FRIDA"
    type_of_backend_hf: Literal["torch", "onnx", "openvino"] = "openvino"

    test_file_path: str = (
        r"D:\PythonProject\EchoMind\src\storage\data_from_local_files\test.txt"
    )

    download_model_for_local_path(
        model_name_from_hf=model_name_frida,
        path_to_download_model=Path(hf_models_path / frida_folder),
    )
