"""
Audio RAG Orchestrator.

Responsibility:
    Coordinate the complete audio question-answering workflow.

Workflow:

    Audio
      ↓
    Speech-to-Text
      ↓
    RAG Pipeline
      ↓
    Text-to-Speech
      ↓
    Update Conversation History
      ↓
    Return Result
"""

from __future__ import annotations

import logging

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)
from langchain_qdrant import QdrantVectorStore

from services.rag.rag_pipeline import answer_question
from services.stt.service import transcribe
from services.tts.service import synthesize

logger = logging.getLogger(__name__)


def ask(
    *,
    audio_bytes: bytes,
    history: list[BaseMessage],
    store: QdrantVectorStore,
) -> tuple[str, bytes, list[BaseMessage]]:
    """
    Execute the complete audio question-answering workflow.

    Steps:
        1. Speech-to-Text
        2. RAG Pipeline
        3. Text-to-Speech
        4. Update conversation history
    """

    logger.info("Starting audio pipeline.")

    query, language = transcribe(audio_bytes)

    language = _normalize_language(language)

    logger.info(
        "User query transcribed successfully."
    )

    answer = answer_question(
        query=query,
        history=history,
        language=language,
        store=store,
    )

    try:
        audio_response = synthesize(answer)

    except Exception:
        logger.exception(
            "TTS failed. Returning text only."
        )

        audio_response = b""

    updated_history = _append_history(
        history,
        query,
        answer,
    )

    logger.info("Audio pipeline completed.")

    return (
        answer,
        audio_response,
        updated_history,
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _append_history(
    history: list[BaseMessage],
    query: str,
    answer: str,
) -> list[BaseMessage]:
    """
    Append the latest interaction to the conversation history.
    """

    return [
        *history,
        HumanMessage(content=query),
        AIMessage(content=answer),
    ]


def _normalize_language(language: str) -> str:
    """
    Normalize language names returned by the STT service.
    """

    language = language.lower()

    if "ar" in language:
        return "Arabic"

    if "en" in language:
        return "English"

    logger.warning(
        "Unknown language '%s'. Falling back to English.",
        language,
    )

    return "English"