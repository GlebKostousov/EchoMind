"""Модуль для создания базового класса будущих ембеддинг векторов"""

from pathlib import Path
from typing import Literal

from huggingface_hub import snapshot_download
from src.tools.self_logger import setup_logger

# TODO: Вынеси в модуль констант
MODEL_CONFIG_TO_EXISTS_CHECK = "config_sentence_transformers.json"
MODULES_JSON_TO_EXISTS_CHECK = "modules.json"

logger = setup_logger(__name__)


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
) -> None:
    """
    Функция сохранения модели в локальную папку
    Args:
        model_name_from_hf (str): название модели с сайта HF
        path_to_download_model (str|Path): Путь или строка пути до папки,
        где будет храниться модель

    Returns:
        None

    """
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
