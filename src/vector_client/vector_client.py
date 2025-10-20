"""Модуль создает клиента для Qdrant"""

__all__ = ("vector_bd_client",)

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance
from qdrant_client.models import VectorParams
from src.tools.const import app_settings


def _create_qdrant_client() -> QdrantClient:
    qdrant_client = QdrantClient(
        location=app_settings.qdrant_config.location,
        port=app_settings.qdrant_config.http_port,
    )

    collections = qdrant_client.get_collections()
    collection_name = app_settings.qdrant_config.collection_name
    if collection_name not in [col.name for col in collections.collections]:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=app_settings.embedding_model_config.frida_output_token,
                distance=Distance.COSINE,
            ),
        )
    return qdrant_client


vector_bd_client = _create_qdrant_client()
