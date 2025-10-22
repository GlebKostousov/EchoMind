"""Модуль для хранения констант и конфигураций"""

from pathlib import Path

from sentence_transformers import SentenceTransformer

__all = ("app_settings",)
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings

__all__ = ("app_settings",)
import os

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 минут
os.environ["HF_HUB_ETAG_TIMEOUT"] = "120"  # 2 минуты


class Bm25Encoder(BaseModel):
    name: str = "Qdrant/bm42-all-minilm-l6-v2-attentions"


class Frida(BaseModel):
    """Модель для эмбеддинга"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: SentenceTransformer = SentenceTransformer("ai-forever/FRIDA")
    frida_input_token: int = 512
    frida_output_token: int = 1536


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

    path_config: AllPath = AllPath()
    chunker_config: ChunkerModel = ChunkerModel()
    embedding_model_config: Frida = Frida()
    qdrant_config: Qdrant = Qdrant()
    bm25_config: Bm25Encoder = Bm25Encoder()


app_settings = Settings()
