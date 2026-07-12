"""
Generator service.

Responsibility:
    Generate an answer from the user query
    and the retrieved context.

Contract:
    Given a user query and a formatted context,
    return an LLM-generated answer.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langfuse import get_client

from config import settings

# ── logging ───────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

_MODEL_NAME = "llama-3.3-70b-versatile"

# ── env ───────────────────────────────────────────────────────────────────────

_GROQ_API_KEY = settings.groq_api_key.get_secret_value()

_langfuse = get_client()


# ── helpers ───────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_llm() -> ChatGroq:
    """
    Return a singleton LLM instance.
    """
    return ChatGroq(
        model=_MODEL_NAME,
        api_key=_GROQ_API_KEY,
    )


@lru_cache(maxsize=1)
def _get_prompt():
    """
    Load the system prompt from Langfuse.
    """
    return _langfuse.get_prompt(
        "muallim-system-prompt",
        type="chat",
    )


def _validate_query(query: str) -> None:
    """
    Validate the user query.
    """
    if not query.strip():
        raise ValueError("Query cannot be empty.")


def _validate_context(context: str) -> None:
    """
    Validate retrieved context.
    """
    if not context.strip():
        raise ValueError("Context cannot be empty.")


def _build_prompt_variables(
    *,
    query: str,
    context: str,
    language: str,
) -> dict[str, str]:
    """
    Prepare prompt variables.
    """
    return {
        "question": query,
        "context": context,
        "language": language,
    }


def _build_prompt(**variables):
    """
    Compile the Langfuse prompt.
    """
    prompt = _get_prompt()
    return prompt.compile(**variables)

def _normalize_language(lang_code: str) -> str:
    """
    Normalize language names to a consistent format.

    Args:
        lang_code: The language name as returned by the STT service (e.g., "Arabic", "English").

    Returns:
        A normalized language name (e.g., "Arabic" or "English").
    """
    lang = lang_code.lower()
    if "ar" in lang:
        return "Arabic"
    elif "en" in lang:
        return "English"
    else:
        return "English"  # default to English if unrecognized


# ── public API ────────────────────────────────────────────────────────────────

def generate_answer(
    query: str,
    context: str,
    language: str,
) -> str:
    """
    Generate an answer using the retrieved context.
    """

    _validate_query(query)
    _validate_context(context)

    variables = _build_prompt_variables(
        query=query,
        context=context,
        language=_normalize_language(language),
    )

    messages = _build_prompt(**variables)

    llm = _get_llm()

    response = llm.invoke(messages)

    logger.info("Answer generated successfully.")

    return response.content.strip()