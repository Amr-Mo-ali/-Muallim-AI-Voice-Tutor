"""
Query rewriter service.

Responsibility:
    Rewrite the user's query using the conversation history.

Contract:
    Given the current user query and recent conversation history,
    return a standalone query optimized for retrieval.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langsmith import traceable
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langfuse import get_client

from config import settings

logger = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────

_MAX_HISTORY = 4

_MODEL_NAME = "llama-3.1-8b-instant"

_GROQ_API_KEY = settings.groq_api_key.get_secret_value()

_langfuse = get_client()


@lru_cache(maxsize=1)
def _get_llm() -> ChatGroq:
    """
    Return the singleton LLM used for query rewriting.
    """
    return ChatGroq(
        model=_MODEL_NAME,
        api_key=_GROQ_API_KEY,
    )


@lru_cache(maxsize=1)
def _get_prompt():
    """
    Load the rewrite prompt from Langfuse.
    """
    return _langfuse.get_prompt(
        "muallim-rewrite_query-prompt2",
        type="chat",
    )


def _validate_query(query: str) -> None:
    """
    Validate the incoming query.
    """
    if not query.strip():
        raise ValueError("Query cannot be empty.")


def _format_history(
    history: list[BaseMessage],
) -> str:
    """
    Convert conversation history into prompt text.
    """
    lines: list[str] = []

    for message in history:

        if message.type == "human":
            role = "Student"

        elif message.type == "ai":
            role = "Tutor"

        else:
            continue

        lines.append(
            f"{role}: {message.content}"
        )

    return "\n\n".join(lines)

@traceable
def rewrite_query(
    query: str,
    history: list[BaseMessage],
) -> str:
    """
    Rewrite the user query into a standalone query.
    """
    _validate_query(query)

    prompt = _get_prompt()

    compiled_prompt = prompt.compile(
        history=_format_history(
            history[-_MAX_HISTORY:]
        ),
        query=query,
    )

    llm = _get_llm()

    try:

        response = llm.invoke(compiled_prompt)

        rewritten_query = response.content.strip()

        logger.info(
            "Query successfully rewritten."
        )

        return rewritten_query

    except Exception:

        logger.exception(
            "Query rewriting failed."
        )

        # Graceful degradation
        return query