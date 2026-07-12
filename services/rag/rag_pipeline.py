"""
RAG pipeline orchestration.

Responsibility:
    Coordinate the RAG workflow.

Contract:
    Execute the RAG pipeline by delegating work
    to the appropriate services.
"""

from __future__ import annotations

import logging

from rag.context_builder import build_context
from rag.document_service import load_and_chunk
from rag.retriever import retrieve
from rag.vector_store import load_vector_store, get_or_create_vector_store
from rag.query_rewriter import rewrite_query

logger = logging.getLogger(__name__)


def index_documents(
    bytes_data: bytes,
    collection_name: str,
):
    """
    Build or load the vector store for a document collection.

    Workflow:
        Document
            ↓
        Chunking
            ↓
        Embedding
            ↓
        Vector Store

    Returns:
        Ready-to-use vector store.
    """
    logger.info(
        "Indexing document into collection '%s'",
        collection_name,
    )

    chunks = load_and_chunk(bytes_data)

    store = get_or_create_vector_store(
        chunks=chunks,
        collection_name=collection_name,
    )

    logger.info(
        "Knowledge base ready for '%s'",
        collection_name,
    )

    return store


def retrieve_context(
    query: str,
    collection_name: str,
) -> str:
    """
    Retrieve the context required to answer a user query.

    Workflow:
        Load Vector Store
              ↓
        Retrieve Chunks
              ↓
        Build Context

    Returns:
        Context ready for the LLM.
    """
    logger.info(
        "Retrieving context from '%s'",
        collection_name,
    )

    store = load_vector_store(collection_name)

    documents = retrieve(
        store=store,
        query=query,
    )

    context = build_context(documents)

    logger.info(
        "Context built successfully."
    )

    return context