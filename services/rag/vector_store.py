"""
Vector store service.

Responsibility:
    Manage document embeddings stored in Qdrant.

Public API:
    - create_vector_store()
    - load_vector_store()
    - get_or_create_vector_store()
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import lru_cache

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_qdrant import QdrantVectorStore
from langsmith import traceable
from qdrant_client import QdrantClient

from config import settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

_EMBEDDING_MODEL = "BAAI/bge-m3"

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

_QDRANT_URL = settings.qdrant_url
_QDRANT_API_KEY = settings.qdrant_api_key.get_secret_value()

# ──────────────────────────────────────────────────────────────
# Shared Resources
# ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_qdrant_client() -> QdrantClient:
    """Return a shared Qdrant client."""

    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key.get_secret_value(),
    )


@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEndpointEmbeddings:
    """Return the shared embedding model."""

    logger.info(
        "Loading embedding model '%s'.",
        _EMBEDDING_MODEL,
    )

    return HuggingFaceEndpointEmbeddings(
        model=_EMBEDDING_MODEL,
        task="feature-extraction",
        huggingfacehub_api_token=settings.hf_token.get_secret_value(),
    )


# ──────────────────────────────────────────────────────────────
# Private Helpers
# ──────────────────────────────────────────────────────────────

def _build_vector_store(
    collection_name: str,
) -> QdrantVectorStore:
    """Return a vector store for an existing collection."""

    return QdrantVectorStore(
        client=_get_qdrant_client(),
        collection_name=collection_name,
        embedding=_get_embeddings(),
    )


def _validate_chunks(
    chunks: Sequence[Document],
) -> None:
    """Validate documents before indexing."""

    if not chunks:
        raise ValueError(
            "Cannot create a vector store from an empty document collection."
        )


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

@traceable
def create_vector_store(
    collection_name: str,
    chunks: Sequence[Document],
) -> QdrantVectorStore:
    """
    Create a new vector store.

    Raises:
        ValueError:
            If the document collection is empty.

        RuntimeError:
            If the collection cannot be created.
    """

    _validate_chunks(chunks)

    logger.info(
        "Creating vector store '%s' (%d chunks).",
        collection_name,
        len(chunks),
    )

    try:
        return QdrantVectorStore.from_documents(
            documents=list(chunks),
            embedding=_get_embeddings(),
            url=_QDRANT_URL,
            api_key=_QDRANT_API_KEY,
            collection_name=collection_name,
        )

    except Exception as exc:
        logger.exception(
            "Failed to create vector store '%s'.",
            collection_name,
        )

        raise RuntimeError(
            f"Failed to create vector store '{collection_name}'."
        ) from exc


@traceable
def load_vector_store(
    collection_name: str,
) -> QdrantVectorStore:
    """
    Load an existing vector store.

    Raises:
        LookupError:
            If the collection does not exist.
    """

    client = _get_qdrant_client()

    if not client.collection_exists(collection_name):
        raise LookupError(
            f"Vector store '{collection_name}' does not exist."
        )

    logger.info(
        "Loading vector store '%s'.",
        collection_name,
    )

    return _build_vector_store(collection_name)


@traceable
def get_or_create_vector_store(
    collection_name: str,
    chunks: Sequence[Document],
) -> QdrantVectorStore:
    """
    Return a ready-to-use vector store.

    If the collection already exists, it is loaded.
    Otherwise, a new collection is created.
    """

    client = _get_qdrant_client()

    if client.collection_exists(collection_name):
        return load_vector_store(collection_name)

    return create_vector_store(
        collection_name=collection_name,
        chunks=chunks,
    )