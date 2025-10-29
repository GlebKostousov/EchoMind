"""Модуль для создания базового класса будущих ембеддинг векторов"""

from pathlib import Path
from typing import Literal

from huggingface_hub import repo_exists, snapshot_download
from src.tools.self_logger import setup_logger

from configs.custom_error import ModelExistsOnHFError

# TODO: Вынеси в модуль констант
# TODO: Сделай автосортировку импортов в ruff
MODEL_CONFIG_TO_EXISTS_CHECK = "config_sentence_transformers.json"
MODULES_JSON_TO_EXISTS_CHECK = "modules.json"

logger = setup_logger(__name__)


# class DenseVectorModel(BaseModel):
#     is_private_model: Annotated[
#         bool,
#         Field(
#             title="This is private model from HF?",
#             description="If attribute == True, model ask a token fot HF",
#         ),
#     ] = False
#     hf_token: Annotated[
#         Optional[str],
#         Field(
#             title="Token to get access private model from HF",
#             description="HuggingFace access token for private models."
#             "\nGet your token at https://huggingface.co/settings/tokens",
#         ),
#     ] = None
#     models_name_from_hf: Annotated[
#         str,
#         Field(
#             title="HuggingFace model name 'org/model'",
#             description="Model name from HF hub. You need insert it similarity",
#         ),
#     ]
#     local_path_for_downloads: Annotated[
#         str | Path,
#         Field(
#             title="path to folder with model",
#             description="Path to folder with model. "
#             "Will be downloaded automatically in first start",
#         ),
#     ]
#     trust_remote_code: Annotated[
#         bool,
#         Field(
#             title="Allow execution of remote code during model loading",
#             description=(
#                 "If True, custom Python code provided by the model author (remote code) will be executed "
#                 "when loading the model from HuggingFace. "
#                 "This feature is required for advanced or experimental models that include custom logic for loading, "
#                 "typically in `model.py` or `configuration.py`. "
#                 "Enabling this flag poses security risks, as the model author can inject arbitrary Python code which "
#                 "will be executed in your environment. "
#                 "Use True only for trusted sources, and never enable for unknown or unverified models. "
#                 "For most use cases, leave this field set to False."
#             ),
#         ),
#     ] = False
#     batch_size: Annotated[
#         int,
#         Field(
#             title="Batch size",
#             description="The number of samples processed together during one forward/backward pass.",
#         ),
#     ] = BATCH_SIZE_FOR_VECTOR_ENCODER
#     show_progress_bar: Annotated[
#         bool,
#         Field(
#             title="Show progress bar in work",
#             description="If True, a progress bar will be shown during
#             long-running operations such as downloads, "
#             "training, or batch processing. This helps visualize the process, "
#             "track remaining time, and improve user experience. For silent or scripted runs,
#             set to False.",
#         ),
#     ] = False
#     device: Annotated[
#         Optional[Literal["CPU", "GPU"]],
#         Field(
#             title="Device type",
#             description="Select the processing device: 'CPU' or 'GPU'.
#             If set to None, "
#             "the application will auto-detect and select the most
#             appropriate device available.",
#         ),
#     ] = None
#
#     @classmethod
#     @field_validator("local_path_for_downloads")
#     def validate_local_path(cls, value: str | Path) -> Path:
#         path = Path(value) if isinstance(value, str) else value
#         if not path.parent.exists():
#             msg = "Path for downloads must be a valid filesystem path."
#             raise ValueError(msg)
#         return path
#
#     @classmethod
#     @field_validator("models_name_from_hf")
#     def validate_hf_model_name(cls, value: str) -> str:
#         """Проверяет модель на соответствие организация/название"""
#         if not re.match(r"^[\w\-]+\/[\w\-\.]+$", value):
#
#             raise ModelExistsOnHFError(msg)
#         return value
#
#     @classmethod
#     @field_validator("hf_token")
#     def validate_token_format(cls, value: str | None) -> str | None:
#         """Проверяет формат токена HF (начинается с hf_)"""
#         if value is not None and not re.match(r"^hf_[A-Za-z0-9]{34,}$", value):
#             msg = (
#                 "Invalid HuggingFace token format. "
#                 'Token should start with "hf_" followed by at least 34 characters'
#             )
#             raise HFValidateTokenError(msg)
#         return value
#
#     @model_validator(mode="after")
#     def validate_token_for_private(self) -> Self:
#         """Проверяет наличие токена для приватных моделей"""
#         if self.is_private_model and not self.hf_token:
#             msg = (
#                 "HuggingFace token is required for private models. "
#                 "Get your token at https://huggingface.co/settings/tokens"
#             )
#             raise HFValidateTokenError(msg)
#         return self


# class EchoMindEmbedding:
#
#     def __init__(self):
#         pass


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
