"""Модуль для создания базового класса будущих ембеддинг векторов"""

from pathlib import Path
from typing import Literal

from huggingface_hub import snapshot_download
from src.tools.self_logger import setup_logger

logger = setup_logger(__name__)


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
    logger.info("Начинаю загрузку модели")
    local_path = Path(path_to_download_model)

    snapshot_download(
        repo_id=model_name_from_hf,
        cache_dir=local_path,
    )
    logger.info("Модель загружена")


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
        path_to_download_model=(hf_models_path / frida_folder),
    )
