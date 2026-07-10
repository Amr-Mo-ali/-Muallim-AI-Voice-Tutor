"""
Context builder.

Responsibility:
    Transform retrieved documents into a prompt-ready context.

Contract:
    Given retrieved document chunks,
    return a formatted context string for the LLM.
"""
from __future__ import annotations

import logging 

from rag.document_service import load_and_chunk, load