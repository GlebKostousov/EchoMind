"""Модуль запуска процесса эмбеддинга текста"""

from pathlib import Path

from chonkie import Chunk
from mypy.messages import Sequence
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from src.tools.const import app_settings

from vector_client.vector_client import vector_bd_client
from vector_way.chunker import chunker

SET_DEBUG = True
show_progress_bar = SET_DEBUG


def _create_embeddings(chunk: Chunk) -> list[float]:
    # TODO: Добавь скачивание в папку
    encoder = SentenceTransformer(
        model_name_or_path="ai-forever/FRIDA",
        cache_folder=app_settings.embedding_model_local_dir,
        trust_remote_code=False,
        model_kwargs={"torch_dtype": "auto"},
    )
    return encoder.encode(
        sentences=chunk.text,
        convert_to_tensor=True,
        prompt="search_document:",
        show_progress_bar=show_progress_bar,
    )


def _create_point(chunk: Chunk) -> PointStruct:
    return PointStruct(
        id=chunk.id,
        vector=_create_embeddings(chunk),
        payload={
            "text": chunk.text,
            "size": chunk.token_count,
            "context": chunk.context,
            "start_index": chunk.start_index,
            "end_index": chunk.end_index,
        },
    )


def fill_points_list(chunks_for_embedding: Sequence[Chunk]) -> list[PointStruct]:
    """
    Функция получает разбитый на чанки текст
    и переваривает его в список объектов для передачи в Qdrant
    """
    return [
        _create_point(chunk=chunk)
        for chunk in chunks_for_embedding
        if isinstance(chunk, Chunk)
    ]


fill_points_list()

vector_bd_client.upsert(
    collection_name=app_settings.qdrant_config.collection_name,
    points=points,
    wait=True,
)


if __name__ == "__main__":
    PATH_DATA_FOR_TEST = r"D:\PythonProject\EchoMind\data\test.txt"

    with Path(PATH_DATA_FOR_TEST).open("r", encoding="utf-8") as file:
        book_text = file.read()

    chunks = chunker(book_text)
