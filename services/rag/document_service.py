"""
Document processing service.

Responsibility:
    Load PDF documents and prepare them for indexing.

Contract:
    Convert raw PDF bytes into enriched LangChain document chunks.

Workflow:
    PDF Bytes
        ↓
    Load Pages
        ↓
    Validate
        ↓
    Filter Empty Pages
        ↓
    Split into Chunks
        ↓
    Enrich Metadata
"""

from __future__ import annotations

import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from collections.abc import Sequence

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

_CHUNK_SIZE = 512
_CHUNK_OVERLAP = 100

_SEPARATORS = (
    "\n\n",
    "\n",
    ".",
    "؟",
    "!",
    "،",
    "؛",
    " ",
)


# ──────────────────────────────────────────────────────────────────────────────
# Splitter
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_splitter() -> RecursiveCharacterTextSplitter:
    """Return a shared text splitter."""

    return RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        separators=_SEPARATORS,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Document Loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_document(bytes_data: bytes) -> list[Document]:
    """
    Load a PDF document from raw bytes.

    Guarantees:
        - Returns LangChain Document objects.
        - Temporary files are always cleaned up.
    """

    if not bytes_data:
        raise ValueError("Document cannot be empty.")

    temp_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as temp_file:
            temp_file.write(bytes_data)
            temp_file.flush()
            temp_path = temp_file.name

        pages = PyMuPDFLoader(temp_path).load()

        logger.info(
            "Loaded %d page(s).",
            len(pages),
        )

        return pages

    except Exception as exc:
        logger.exception("Failed to load PDF document.")
        raise RuntimeError("Failed to load document.") from exc

    finally:
        if temp_path and Path(temp_path).exists():
            try:
                os.unlink(temp_path)
            except OSError:
                logger.warning(
                    "Failed to delete temporary file '%s'.",
                    temp_path,
                )


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

def _is_valid_page(page: Document) -> bool:
    """Return True if the page contains readable content."""

    return bool(page.page_content.strip())


def _filter_valid_pages(
    pages: Sequence[Document],
) -> list[Document]:
    """
    Remove empty pages from the document.
    """

    valid_pages = [
        page
        for page in pages
        if _is_valid_page(page)
    ]

    if not valid_pages:
        raise ValueError(
            "Document contains no readable content."
        )

    logger.info(
        "Validated pages: %d/%d",
        len(valid_pages),
        len(pages),
    )

    return valid_pages


# ──────────────────────────────────────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────────────────────────────────────

def _split_documents(
    pages: Sequence[Document],
) -> list[Document]:
    """
    Split pages into overlapping chunks.
    """

    chunks = _get_splitter().split_documents(list(pages))

    logger.info(
        "Created %d chunk(s).",
        len(chunks),
    )

    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Metadata
# ──────────────────────────────────────────────────────────────────────────────

def _enrich_chunk_metadata(
    chunks: list[Document],
) -> list[Document]:
    """
    Add default metadata required by downstream services.
    """

    total_chunks = len(chunks)

    for index, chunk in enumerate(chunks):

        metadata = chunk.metadata or {}

        metadata.setdefault(
            "source",
            "uploaded_pdf",
        )

        metadata.setdefault(
            "page",
            0,
        )

        metadata["chunk_index"] = index
        metadata["chunk_count"] = total_chunks

        chunk.metadata = metadata

    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_and_chunk(
    bytes_data: bytes,
) -> list[Document]:
    """
    Load a PDF document and prepare chunks for indexing.

    Workflow:
        PDF Bytes
            ↓
        Load Pages
            ↓
        Filter Empty Pages
            ↓
        Split into Chunks
            ↓
        Enrich Metadata

    Returns:
        Ready-to-index document chunks.
    """

    pages = _load_document(bytes_data)

    valid_pages = _filter_valid_pages(pages)

    chunks = _split_documents(valid_pages)

    return _enrich_chunk_metadata(chunks)