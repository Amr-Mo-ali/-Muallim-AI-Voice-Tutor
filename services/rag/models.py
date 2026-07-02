"""
This module defines the data model for the rewritten query result in the RAG pipeline. It uses Pydantic's BaseModel to ensure data validation and type checking.
The `RewriteResult` class encapsulates the original query, the rewritten query, 
and metadata about the rewriting process, including whether a fallback was used and the lengths of the original and rewritten queries.
"""

from enum import Enum
from pydantic import BaseModel


class RewriteStatus(str, Enum):
    SUCCESS = "success"
    FALLBACK = "fallback"
    FAILED = "failed"


class RewriteResult(BaseModel):

    original_query: str

    retrieval_query: str

    status: RewriteStatus

    fallback_reason: str | None = None

    original_length: int

    rewritten_length: int