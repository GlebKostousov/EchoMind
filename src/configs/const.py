"""Модуль для хранения констант и конфигураций"""

__all = (
    "app_settings",
    "MODEL_CONFIG_TO_EXISTS_CHECK",
    "MODULES_JSON_TO_EXISTS_CHECK",
    "BATCH_SIZE_FOR_VECTOR_ENCODER",
    "STANDARD_PATH_TO_MODEL",
)

import os
from abc import ABC
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings
from sentence_transformers import SentenceTransformer

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 минут
os.environ["HF_HUB_ETAG_TIMEOUT"] = "120"  # 2 минуты


# -----     Sparse вектора     -----
class SparseEmbeddingAbstract(ABC):
    """Интерфейс для конфигурации эмбеддинга разряженного вектора"""

    name: str


class Bm25Encoder(BaseModel, SparseEmbeddingAbstract):
    """Конфигурация bm25"""

    name: str = "Qdrant/bm42-all-minilm-l6-v2-attentions"


# -----     Dense вектора     -----
class DenseEmbeddingAbstract(ABC):
    """Интерфейс для конфигурации эмбеддинга плотного вектора"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: SentenceTransformer
    model_input_token: int
    model_output_token: int
    show_progress: bool


class Frida(BaseModel, DenseEmbeddingAbstract):
    """Конфигурация для frida"""

    model: SentenceTransformer = SentenceTransformer("ai-forever/FRIDA")
    model_input_token: int = 512
    model_output_token: int = 1536
    show_progress: bool = True


class Qdrant(BaseModel):
    """Конфиг настройки Qdrant"""

    http_port: int = 6333
    location: str = "localhost"
    collection_name: str = "sentence_collection"


class ChunkerModel(BaseModel):
    """Настройка модели чанкирования"""

    similarity_threshold: float = 0.8
    chunk_models: str = "minishlab/potion-base-8M"
    chunk_models_lang: str = "ru"
    min_sentences_per_chunk: int = 2


class AllPath(BaseModel):
    """Все пути"""

    src_url: Path = Path(__file__).parent.parent
    chunk_models_local_dir: Path = src_url / "dense_vector" / "chunk_model"
    embedding_model_local_dir: Path = src_url / "dense_vector" / "embedding_model"
    bm25_model_local_dir: Path = src_url / "sparse_vector" / "bm25_model"


class Settings(BaseSettings):
    """Общие настройки приложения"""

    device_for_llm: Annotated[
        Literal["CPU", "GPU"],
        Field(
            title="Тип устройства",
            description="Устройство для выполнения LLM (CPU или GPU)",
        ),
    ]

    path_config: Annotated[
        AllPath,
        Field(description="Конфигурация локальных путей приложения"),
    ]

    chunker_config: Annotated[
        ChunkerModel,
        Field(description="Настройки разбиения текста на части"),
    ]

    embedding_model_config: Annotated[
        DenseEmbeddingAbstract,
        Field(description="Настройки модели для плотных эмбеддингов"),
    ]

    bm25_config: Annotated[
        SparseEmbeddingAbstract,
        Field(description="Параметры модели BM25"),
    ]

    qdrant_config: Annotated[
        Qdrant,
        Field(description="Конфигурация подключения к Qdrant"),
    ]


app_settings = Settings(
    device_for_llm="CPU",
    path_config=AllPath(),
    chunker_config=ChunkerModel(),
    embedding_model_config=Frida(),
    bm25_config=Bm25Encoder(),
    qdrant_config=Qdrant(),
)
MODEL_CONFIG_TO_EXISTS_CHECK = "config_sentence_transformers.json"
MODULES_JSON_TO_EXISTS_CHECK = "modules.json"
BATCH_SIZE_FOR_VECTOR_ENCODER = 32
# TODO: Доделай!
STANDARD_PATH_TO_MODEL = Path()
