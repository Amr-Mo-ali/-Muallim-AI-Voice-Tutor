"""
This module defines the data models used in the RAG (Retrieval-Augmented Generation) service.
"""

from pydantic import BaseModel

class RewriteResult(BaseModel):
    """
    Represents the result of a rewrite operation.

    Attributes:
        original_query (str): The original query.
        retrieval_query (str): The query used for retrieval.
        metadata (dict): Additional metadata related to the rewrite operation.
    """
    original_query: str
    retrieval_query: str
    metadata: dict

    used_fallback: bool = False
    fallback_reason: str | None =  None

    original_length: int
    rewritten_length: int