"""
Retriever service.

Responsibility:
    Retrieve the most relevant document chunks from a vector store.

Contract:
    Given a user query and a vector store,
    return the most relevant document chunks for downstream generation.
"""


from __future__ import annotations

import logging
from typing import Any

from langsmith import traceable
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

# ── constants ─────────────────────────────────────────────────────────────────
_RETRIEVER_K = 5
_RETRIEVER_FETCH_K = 20
_RETRIEVER_LAMBDA = 0.75
#─────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

def _validate_query(query: str) -> None:
    """
    Validate the user query.

    Args:
        query: The user query to validate.

    Returns:
        Raises:
    ValueError:
        If the query is empty.
    """
    if not query.strip():
        raise ValueError("Query cannot be empty.")

def _create_retriever(store: QdrantVectorStore) -> Any:
    """
    Build a retriever from a vector store.

    Args:
        store: Initialized vector store.

    Returns:
        A retriever configured for semantic search.
    """
    return store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": _RETRIEVER_K,
            "fetch_k": _RETRIEVER_FETCH_K,
            "lambda_mult": _RETRIEVER_LAMBDA,
        }
    )

@traceable
def retrieve(
        store: QdrantVectorStore,
        query: str,
) -> list[Document]:
    """
    Retrieve relevant document chunks from a vector store.
    Args:
        store: Initialized vector store.
        query: User query for retrieval.
        Returns:
    Relevant document chunks.
    Raises:
        ValueError:
            If the query is empty.
    """
    _validate_query(query)

    retriever = _create_retriever(store)

    documents = retriever.invoke(query)

    pages = [
        doc.metadata.get("page")
        for doc in documents
    ]

    logger.info(
        "Retrieved %d chunks | for query=%r | Pages=%s",
        len(documents),
        query,
        pages,
    )
    return documents
