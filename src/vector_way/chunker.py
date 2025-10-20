"""Модуль для разбития текста на чанки"""

__all__ = ("chunker",)

from pathlib import Path

from chonkie import SemanticChunker
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError
from loguru import logger
from src.tools.const import app_settings


def _check_model_exists(local_dir: Path | str) -> bool:
    """
    Проверяет наличие модели в локальной папке.
    Возвращает True, если папка существует и содержит config.json.
    """

    if isinstance(local_dir, str):
        local_dir = Path(local_dir)
    if not Path(local_dir).exists():
        logger.error("Папка {} не существует.", local_dir)
        return False

    config_path = local_dir / "config.json"
    if Path(config_path).exists():
        model_file = local_dir / "model.safetensors"
        if Path(model_file).exists():
            logger.debug("Модель присутствует в кеше")
            return True

    logger.info("Модель отсутствует, необходимо скачивание")
    return False


def _get_chunker(
    local_dir: str | Path,
    repo_id: str,
) -> SemanticChunker:
    if not _check_model_exists(local_dir=local_dir):
        logger.info(
            "Начинаю скачивать модель {}",
            app_settings.chunker_config.chunk_models,
        )
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
            )
        except (OSError, PermissionError) as e:
            msg = f"Ошибка доступа к файловой системе: {e}"
            raise RuntimeError(msg) from e
        except (ConnectionError, Timeout, RequestException) as e:
            msg = f"Ошибка сети при скачивании модели: {e}"
            raise RuntimeError(msg) from e
        except HfHubHTTPError as e:
            msg = f"Ошибка HTTP при скачивании с HuggingFace: {e}"
            raise RuntimeError(msg) from e
        except ValueError as e:
            msg = f"Неверные параметры для скачивания: {e}"
            raise RuntimeError(msg) from e

    return SemanticChunker(
        local_path=local_dir,
        embedding_model=app_settings.chunker_config.chunk_models,
        chunk_size=app_settings.embedding_model_config.frida_input_token,
        threshold=app_settings.chunker_config.similarity_threshold,
        min_sentences_per_chunk=app_settings.chunker_config.min_sentences_per_chunk,
        lang=app_settings.chunker_config.chunk_models_lang,
    )


chunker = _get_chunker(
    local_dir=app_settings.chunk_models_local_dir,
    repo_id=app_settings.chunker_config.chunk_models,
)

if __name__ == "__main__":
    with Path(r"D:\PythonProject\EchoMind\data\test.txt", encoding="utf-8").open() as f:
        book_text = f.read()
    chunks = chunker(book_text)

    for i, chunk in enumerate(chunks):
        logger.info("\n--- Чанк {}  ---", i + 1)
        logger.info("Текст: {}", chunk.text)
        logger.info("Токенов: {}", chunk.token_count)
