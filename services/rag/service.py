"""
service.py — Muallim RAG pipeline
Responsibilities:
  - PDF loading & chunking
  - Embedding + QdrantVectorStore (idempotent)
  - Semantic retrieval via MMR
"""

from __future__ import annotations


import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from langchain_qdrant import QdrantVectorStore
from langsmith import traceable
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

# ── logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
# ── env ───────────────────────────────────────────────────────────────────────
load_dotenv()
_QDRANT_URL = os.getenv("QDRANT_URL")
_QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
_HF_API_KEY = os.getenv("HF_TOKEN")
# ── constants ─────────────────────────────────────────────────────────────────
_CHUNK_SIZE         = 512
_CHUNK_OVERLAP      = 100
_RETRIEVER_K        = 3
_RETRIEVER_FETCH_K  = 10
_EMBEDDING_MODEL    = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ── singletons ────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEndpointEmbeddings:
    logger.info("Loading embedding model: %s", _EMBEDDING_MODEL)
    return HuggingFaceEndpointEmbeddings(
        model=_EMBEDDING_MODEL,
        task="feature-extraction",
        huggingfacehub_api_token=_HF_API_KEY,
    )
@lru_cache(maxsize=1)
def _get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=_QDRANT_URL ,api_key=_QDRANT_API_KEY)
# ── public API ────────────────────────────────────────────────────────────────
@traceable
def load_and_chunk(bytes_data: bytes) -> list[Document]:
    """
    Load a PDF from bytes and split it into overlapping chunks.
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

def count_chunks(collection_name: str) -> int:
    client = _get_qdrant_client()
    return client.count(collection_name).count

@traceable
def get_or_create_vector_store(
        chunks: list[Document],
        collection_name: str
        ) ->  QdrantVectorStore:
    """
    Idempotent vector store creator 
    """
    client = _get_qdrant_client()
    exists = client.collection_exists(collection_name) 
    try:
        if exists:
            return QdrantVectorStore(
                client=client,          
                collection_name=collection_name,
                embedding=_get_embeddings(),
            )
        else:
            vector_store = QdrantVectorStore.from_documents(
                documents=chunks,       
                embedding=_get_embeddings(),
                url=_QDRANT_URL,
                api_key=_QDRANT_API_KEY,
                collection_name=collection_name,
            )
            return vector_store
    except Exception as e:
         raise RuntimeError(f"Failed to Find or create Vector Database: {e}") from e

def load_vector_db(collection_name: str) -> QdrantVectorStore:
    """Load existing QdrantVectorStore for session recovery."""
    client = _get_qdrant_client()
    if not client.collection_exists(collection_name):
        raise FileNotFoundError(f"Vector store not found: {collection_name}")
    
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=_get_embeddings(),
    )


@traceable
def search_vector_db(vector_store: QdrantVectorStore, query: str) -> list[Document]:
    """Retrieve relevant documents using MMR search."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    try:
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": _RETRIEVER_K, "fetch_k": _RETRIEVER_FETCH_K},
        )
        return retriever.invoke(query)
    except Exception as exc:
        raise RuntimeError(f"Vector store retrieval failed:{exc}") from exc