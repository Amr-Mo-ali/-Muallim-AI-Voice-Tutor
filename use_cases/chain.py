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

from langfuse import get_client

from services.rag.rag_pipeline import answer_question
from services.stt.service import transcribe
from services.tts.service import synthesize


# ── env ───────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

_langfuse = get_client
# ───────────────────────────────────────────────────────────────────────
with _langfuse._start_as_current_otel_span_with_processed_media(as_type="trace", name="ask-request") as trace :
        logger.info("Processing ask for user session")
def ask(
    *,
    audio_bytes: bytes,
    history: list[BaseMessage],
    collection_name: str,
) -> tuple[str, bytes, list[BaseMessage]]:
    """
    Execute the complete audio question-answering workflow.

    Steps:
        1. Speech-to-Text
        2. RAG Pipeline
        3. Text-to-Speech
        4. Update conversation history

    Returns:
        answer,
        synthesized audio,
        updated conversation history.
    """


    # ------------------------------------------------------------
    # Speech-to-Text
    # ------------------------------------------------------------

    with _langfuse.start_as_current_observation(as_type="span", name="stt-request") as span:
        logger.info("Starting audio pipeline.")

        query, language = transcribe(audio_bytes)

        language = _normalize_language(language)

        logger.info("Speech successfully transcribed.")
        span.update(output={"query":query ,"language": language})
    # ------------------------------------------------------------
    # RAG
    # ------------------------------------------------------------
    with _langfuse.start_as_current_observation(as_type="span", name="rag_pipeline") as span:
        
        answer = answer_question(
            query=query,
            history=history,
            language=language,
            collection_name=collection_name,
        )
        span.update(
            input={
            "query":query,
            "history":history,
            "language":language,
            "collection_name":collection_name,
        },
        output={"answer":answer},
        )

    # ------------------------------------------------------------
    # Text-to-Speech
    # ------------------------------------------------------------
    with _langfuse.start_as_current_observation(as_type="spen", name="tts_response") as spen:
        try:
            audio_response = synthesize(answer)

        except Exception:
            logger.exception(
                "TTS failed. Returning text only."
            )

            audio_response = b""
        span.update(output={
                    "audio_length": len(audio_response),
                })

    # ------------------------------------------------------------
    # Conversation History
    # ------------------------------------------------------------

    updated_history = _append_history(
        history=history,
        query=query,
        answer=answer,
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
    Return a new conversation history with the latest interaction appended.
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