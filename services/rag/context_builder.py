"""
Context builder.

Responsibility:
    Transform retrieved document chunks into a prompt-ready context.

Contract:
    Given retrieved document chunks,
    return a formatted context string for the LLM.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _validate_document(chunk: Document) -> bool:
    """
    Check whether a retrieved chunk contains usable content.
    """
    return bool(chunk.page_content.strip())


def _format_chunk(chunk: Document) -> str:
    """
    Format a single document chunk.
    """
    page = chunk.metadata.get("page", "Unknown")
    source = chunk.metadata.get("source", "Unknown")

    return (
        f"[Source: {source} | Page: {page}]\n"
        f"{chunk.page_content.strip()}"
    )


def _join_chunks(formatted_chunks: list[str]) -> str:
    """
    Combine formatted chunks into a single context string.
    """
    return "\n\n--------------------\n\n".join(formatted_chunks)


def build_context(chunks: Sequence[Document]) -> str:
    """
    Build an LLM-ready context from retrieved document chunks.
    """
    valid_chunks = [
        chunk
        for chunk in chunks
        if _validate_document(chunk)
    ]

    if not valid_chunks:
        logger.warning("No valid chunks available to build context.")
        return ""

    formatted_chunks = [
        _format_chunk(chunk)
        for chunk in valid_chunks
    ]

    context = _join_chunks(formatted_chunks)

    logger.info(
        "Built context from %d chunks.",
        len(valid_chunks),
    )

    return context