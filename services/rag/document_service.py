from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ── Chunking configuration ────────────────────────────────────────────────────
_CHUNK_SIZE = 512
_CHUNK_OVERLAP = 100

_SEPARATORS = [
    "\n\n",
    "\n",
    ".",
    "؟",
    "!",
    "،",
    "؛",
    " ",
]


def _load_document(bytes_data: bytes) -> list[Document]:
    """
    Load a document from raw bytes.

    Guarantees:
        - Returns LangChain Document objects.
        - Cleans up temporary files.
    """
    if not bytes_data:
        raise ValueError("Document cannot be empty.")

    logger.info("Loading document")

    tmp_path: str | None = None

    try:
        # Use delete=False + manual cleanup (best practice on Windows)
        # delete=False mean don`t delete the file automaticly
        # suffix=".pdf" -> save the temp file as .pdf
        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as tmp:
            tmp.write(bytes_data)
            tmp.flush()
            tmp_path = tmp.name

        loader = PyMuPDFLoader(tmp_path)

        return loader.load()

    except Exception as exc:
        raise RuntimeError("Failed to load document.") from exc

    finally:
        if tmp_path and Path(tmp_path).exists():
            try:
                os.unlink(tmp_path)
            except OSError:
                logger.warning(
                    "Failed to delete temporary file: %s",
                    tmp_path,
                )


def _split_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into overlapping chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        separators=_SEPARATORS,
    )

    return splitter.split_documents(documents)


def _enrich_chunk_metadata(chunks: list[Document]) -> None:
    """
    Add default metadata required by the pipeline.
    """
    for index, chunk in enumerate(chunks):
        chunk.metadata = chunk.metadata or {}

        chunk.metadata.setdefault("source", "uploaded_pdf")
        chunk.metadata.setdefault("page", 0)
        chunk.metadata["chunk_index"] = index


def load_and_chunk(bytes_data: bytes) -> list[Document]:
    """
    Load a document and prepare chunks for downstream processing.
    """
    documents = _load_document(bytes_data)

    chunks = _split_documents(documents)

    _enrich_chunk_metadata(chunks)

    return chunks