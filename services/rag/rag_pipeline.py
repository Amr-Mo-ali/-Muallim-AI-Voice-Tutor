"""
RAG pipeline orchestration.

Responsibility:
    Coordinate the end-to-end Retrieval-Augmented Generation workflow.

Contract:
    Orchestrate the pipeline by delegating each step
    to the appropriate service.

Workflow:

    Indexing
    --------
    PDF
      ↓
    Document Service
      ↓
    Vector Store


    Question Answering
    ------------------
    User Question
          ↓
    Query Rewriter
          ↓
    Retriever
          ↓
    Context Builder
          ↓
    Generator
          ↓
       Final Answer
"""

from __future__ import annotations

import logging

from langchain_core.messages import BaseMessage
from langchain_qdrant import QdrantVectorStore

from services.rag.document_service import load_and_chunk
from services.rag.vector_store import (
    get_or_create_vector_store,
    load_vector_store,
)
from services.rag.query_rewriter import rewrite_query
from services.rag.retriever import retrieve
from services.rag.context_builder import build_context
from services.rag.generator import generate_answer

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Indexing Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def index_documents(
    bytes_data: bytes,
    collection_name: str,
) -> QdrantVectorStore:
    """
    Build (or load) the knowledge base for a document.

    Workflow:
        PDF
            ↓
        Chunking
            ↓
        Embedding
            ↓
        Vector Store
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
        "Knowledge base ready."
    )

    return store


# ──────────────────────────────────────────────────────────────────────────────
# Inference Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def answer_question(
    *,
    query: str,
    history: list[BaseMessage],
    language: str,
    collection_name: str,
) -> str:
    """
    Execute the complete RAG inference pipeline.

    Workflow:

        Original Question
                ↓
         Rewrite Query
                ↓
          Retrieve Chunks
                ↓
          Build Context
                ↓
          Generate Answer
    """

    logger.info("Starting RAG pipeline.")

    store = load_vector_store(collection_name)

    rewritten_query = rewrite_query(
        query=query,
        history=history,
    )

    retrieved_chunks = retrieve(
        store=store,
        query=rewritten_query,
    )

    context = build_context(
        retrieved_chunks,
    )
    if not context:
        logger.info("No relevant chunkd retrieved")
        return (
            "I couldn`t find relevant information in the upload documents"
        )
    answer = generate_answer(
        query=query,
        context=context,
        language=language,
    )

    logger.info("RAG pipeline completed successfully.")

    return answer