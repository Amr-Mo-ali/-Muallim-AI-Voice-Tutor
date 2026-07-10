"""
    Query Rewriter Service
    Rewrites user queries to the best format the model can understand, based on the conversation history.
"""

from __future__ import annotations

import logging

from langchain_groq import ChatGroq
from functools import lru_cache
from config import settings
from langfuse import get_client


MAX_HISTORY = 4
_MODEL_NAME_FOR_QUERY_REWRITER = "llama-3.1-8b-instant"
_GROQ_API_KEY = settings.groq_api_key.get_secret_value()


langfuse = get_client()

query_rewriter_logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def _get_llm_for_query_rewriter() -> ChatGroq:
    """
    Get a ChatGroq instance for generating responses.

    Returns:
        A ChatGroq instance initialized with the appropriate model and API key.
    """
    return ChatGroq(
        model=_MODEL_NAME_FOR_QUERY_REWRITER,
        api_key=_GROQ_API_KEY,
    )


def format_history(history):
    lines = []

    for msg in history:
        if msg.type == "human":
            role = "Student"
        elif msg.type == "ai":
            role = "Tutor"
        else:
            continue
        lines.append(f"{role}: {msg.content}")
    return "\n\n".join(lines)

def rewrite_query(query: str, history: list) -> str:
    """
    Rewrite the user query based on the conversation history.

    Args:
        query (str): The original user query.
        history (list): The conversation history.

    Returns:
        str: The rewritten query.
    """
    prompt = langfuse.get_prompt(
        "muallim-rewrite_query-prompt2",
        type="chat",
    )
    recent_history = history[-MAX_HISTORY:]

    compiled_prompt = prompt.compile(
        history=format_history(recent_history),
        query=query,
    )

    llm = _get_llm_for_query_rewriter()

    try:
        response = llm.invoke(compiled_prompt)
        return response.content.strip()
    except Exception:
        query_rewriter_logger.exception("Query rewriting failed.")