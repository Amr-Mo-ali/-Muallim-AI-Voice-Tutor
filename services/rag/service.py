"""
service.py — Muallim RAG pipeline
Responsibilities:
  - PDF loading & chunking
  - Embedding + Chroma vector store (idempotent with content hash)
  - Semantic retrieval via MMR
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langsmith import traceable
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── env ───────────────────────────────────────────────────────────────────────
load_dotenv()

_CHROMA_BASE_DIR = os.getenv("CHROMA_DB_DIR", "data/chroma_db")

# ── constants ─────────────────────────────────────────────────────────────────
_CHUNK_SIZE         = 512
_CHUNK_OVERLAP      = 100
_RETRIEVER_K        = 3
_RETRIEVER_FETCH_K  = 10

_EMBEDDING_MODEL    = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_HASH_FILENAME      = "pdf_hash.json"

# ── singletons ────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    logger.info("Loading embedding model: %s", _EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(
        model_name=_EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 32},
    )


# ── hash helpers ──────────────────────────────────────────────────────────────
def _chunks_sha256(chunks: list[Document]) -> str:
    """Deterministic SHA-256 of all chunk contents."""
    h = hashlib.sha256()
    for doc in chunks:
        h.update(doc.page_content.encode("utf-8"))
    return h.hexdigest()


def _read_stored_hash(persist_dir: str) -> str | None:
    hash_file = Path(persist_dir) / _HASH_FILENAME
    if hash_file.exists():
        try:
            return json.loads(hash_file.read_text(encoding="utf-8"))["sha256"]
        except Exception:
            return None
    return None


def _write_stored_hash(persist_dir: str, sha: str) -> None:
    hash_file = Path(persist_dir) / _HASH_FILENAME
    hash_file.parent.mkdir(parents=True, exist_ok=True)
    hash_file.write_text(json.dumps({"sha256": sha}), encoding="utf-8")


# ── public API ────────────────────────────────────────────────────────────────
@traceable
def load_and_chunk(bytes_data: bytes) -> list[Document]:
    """
    Load a PDF from bytes and split it into overlapping chunks.
    Fixed for Windows permission issues.
    """
    if not bytes_data:
        raise ValueError("PDF cannot be empty.")
    
    logger.info("Loading PDF from memory")
    
    # Use delete=False + manual cleanup (best practice on Windows)
    # delete=False mean don`t delete the file automaticly
    # suffix=".pdf" -> save the temp file as .pdf
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(bytes_data)
            tmp.flush()
            tmp_path = tmp.name
        
        # Now the file is closed → PyMuPDF can read it safely
        loader = PyMuPDFLoader(file_path=tmp_path)
        documents = loader.load()
        
    except Exception as exc:
        raise RuntimeError("Failed to parse PDF from bytes.") from exc
    finally:
        # Clean up temp file
        if tmp_path and Path(tmp_path).exists():
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning("Failed to delete temp file %s: %s", tmp_path, e)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)
    logger.info("PDF split into %d chunks", len(chunks))
    return chunks

@traceable
def get_or_create_vector_store(
    chunks: list[Document],
    persist_dir: str | None = None,
) -> Chroma:
    """
    Idempotent vector store creator/loader.
    """
    persist_dir = persist_dir or _CHROMA_BASE_DIR
    current_sha = _chunks_sha256(chunks)
    stored_sha = _read_stored_hash(persist_dir)

    if stored_sha == current_sha and Path(persist_dir).exists():
        logger.info("Hash match → loading existing vector store")
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=_get_embeddings(),
        )

    logger.info("Creating new vector store (hash mismatch or first time)")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=_get_embeddings(),
        persist_directory=persist_dir,
    )

    _write_stored_hash(persist_dir, current_sha)
    logger.info("Vector store created successfully with %d chunks", len(chunks))
    return vector_store


def load_vector_db(persist_dir: str) -> Chroma:
    """Load existing Chroma for session recovery."""
    if not Path(persist_dir).exists():
        raise FileNotFoundError(f"Vector store not found: {persist_dir}")
    
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=_get_embeddings(),
    )


@traceable
def search_vector_db(vector_store: Chroma, query: str) -> list[Document]:
    """MMR retrieval for better diversity."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    try:
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": _RETRIEVER_K, "fetch_k": _RETRIEVER_FETCH_K},
        )
        return retriever.invoke(query)
    except Exception as exc:
        raise RuntimeError("Vector store retrieval failed.") from exc