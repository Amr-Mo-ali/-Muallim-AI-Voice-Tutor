"""
Vector store service.

Responsibility:
    Store and load document embeddings in Qdrant.

Contract:
    After a successful call, the caller receives a ready-to-use
    QdrantVectorStore instance.

Guarantees:
    - Returns a ready-to-use vector store.
    - Creates the collection if it does not already exist.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_qdrant import QdrantVectorStore
from langsmith import traceable
from qdrant_client import QdrantClient

from config import settings

# ── logging ───────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# ── configuration ─────────────────────────────────────────────────────────────

_QDRANT_URL = settings.qdrant_url
_QDRANT_API_KEY = settings.qdrant_api_key.get_secret_value()

_HF_API_KEY = settings.hf_token.get_secret_value()

_EMBEDDING_MODEL = "BAAI/bge-m3"

# ── singletons ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_qdrant_client() -> QdrantClient:
    """Return a shared Qdrant client instance."""
    return QdrantClient(
        url=_QDRANT_URL,
        api_key=_QDRANT_API_KEY,
    )


@lru_cache(maxsize=1)
def _get_embedding_model() -> HuggingFaceEndpointEmbeddings:
    """Return the shared embedding model."""
    logger.info("Loading embedding model: %s", _EMBEDDING_MODEL)

    return HuggingFaceEndpointEmbeddings(
        model=_EMBEDDING_MODEL,
        task="feature-extraction",
        huggingfacehub_api_token=_HF_API_KEY,
    )


# ── validation ────────────────────────────────────────────────────────────────

def _validate_chunks(chunks: list[Document]) -> None:
    """
    Validate the document collection before indexing.
    """
    if not chunks:
        raise ValueError(
            "Cannot create a vector store from an empty document collection."
        )


# ── public API ────────────────────────────────────────────────────────────────

@traceable
def load_or_create_vector_store(
    chunks: list[Document],
    collection_name: str,
) -> QdrantVectorStore:
    """
    Load an existing vector store or create one if it does not exist.

    Guarantees:
        - Returns a ready-to-use vector store.
        - Creates the collection when it does not already exist.
    """
    _validate_chunks(chunks)

    client = _get_qdrant_client()

    try:
        if client.collection_exists(collection_name):
            logger.info(
                "Loading existing vector store '%s'",
                collection_name,
            )

            return QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=_get_embedding_model(),
            )

        logger.info(
            "Creating vector store '%s' (%d chunks)",
            collection_name,
            len(chunks),
        )

        return QdrantVectorStore.from_documents(
            documents=chunks,
            collection_name=collection_name,
            client=client,
            embedding=_get_embedding_model(),
        )

    except Exception as exc:
        logger.exception(
            "Failed to initialize vector store '%s'",
            collection_name,
        )

        raise RuntimeError(
            f"Failed to initialize vector store '{collection_name}'."
        ) from exc


@traceable
def load_vector_store(
    collection_name: str,
) -> QdrantVectorStore:
    """
    Load an existing vector store.
    """
    client = _get_qdrant_client()

    if not client.collection_exists(collection_name):
        raise FileNotFoundError(
            f"Vector store '{collection_name}' does not exist."
        )

    logger.info(
        "Loading vector store '%s'",
        collection_name,
    )

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=_get_embedding_model(),
    )