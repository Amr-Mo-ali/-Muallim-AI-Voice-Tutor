"""
Orchestrates the RAG pipeline for handling audio queries.
Responsibilities:
  - Transcribing audio to text using the STT service.
  - Retrieving relevant chunks from the RAG vector store.
    - Generating a response using the LLM based on the retrieved context.
This module serves as the central coordinator for processing audio queries and generating responses.
"""
from __future__ import annotations

from services.rag import service as rag_service
from services.stt import service as stt_service
from services.tts import service as tts_service


from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langfuse import get_client

from functools import lru_cache
import logging
from config import settings
from pprint import pformat


# ── logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
# ── env ───────────────────────────────────────────────────────────────────────
_GROQ_API_KEY = settings.groq_api_key.get_secret_value()
# ── constants ─────────────────────────────────────────────────────────────────
_MODEL_NAME = "llama-3.3-70b-versatile"  # choose the appropriate model for your use case
_MODEL_NAME_FOR_QUERY_REWRITER = "llama-3.1-8b-instant"
MAX_HISTORY = 4
langfuse = get_client()

#── helpers ───────────────────────────────────────────────────────────────
def build_context(chunks):

    context = []

    for chunk in chunks:

        page = chunk.metadata.get("page", "?")

        context.append(
            f"[Page {page}]\n{chunk.page_content}"
        )

    return "\n\n".join(context)
# ── public API ────────────────────────────────────────────────────────────────
def ask(
    audio_bytes: bytes,
    history: list,
    vector_store
    ) -> tuple[str, bytes, list, str]:
    """
        Orchestrates the RAG pipeline:
        1. Transcribes audio to text.
        2. Retrieves relevant chunks from the vector store.
        3. Generates a response using the LLM.
    """
    with langfuse._start_as_current_otel_span_with_processed_media(as_type="trace", name="ask-request") as trace :
        logger.info("Processing ask for user session")
        # Create a span using a context manager
        with langfuse.start_as_current_observation(as_type="span", name="stt-request") as span:
            # Step 1: Transcribe audio
            # Note: In a real implementation, we would also want to handle language detection and possibly translation here.
            # Transcript returns both the transcribed text and language
            query , language = stt_service.transcribe(audio_bytes)
            language = _normalize_language(language)
            logger.info("Transcribed query: %s (Language: %s)", query, language)
            span.update(output={"query":query ,"language": language})
            # stor the rewrite_query in a new variable 
        # Create a span using a context manager
        # Step 2: Rewrite the query
        with langfuse.start_as_current_observation(
            as_type="generation",
            name="query-rewrite",
            model=_MODEL_NAME_FOR_QUERY_REWRITER,
            ) as generation:
            rewrite = rewrite_query(query, history)
            generation.update(
                input={
                    "original_query": query,
                    "history": history,
                },
                output={
                    "retrieval_query": rewrite,
                },
            )
        # Step 3: Retrieve relevant chunks
        with langfuse.start_as_current_observation(
            as_type="span",
            name="retrieve",
            ) as span:
            relevant_chunks = rag_service.search_vector_db(
                vector_store,
                rewrite,
            )
            context = build_context(relevant_chunks)
            span.update(
                output={
                    "chunks_count": len(relevant_chunks),
                    "pages": [
                        c.metadata.get("page")
                        for c in relevant_chunks
                    ],
                    "chunk_indices": [
                        c.metadata.get("chunk_index")
                        for c in relevant_chunks
                    ],
                    "sources": list({
                        c.metadata.get("source")
                        for c in relevant_chunks
                    }),
                    "context_length": len(context),
                }
                )
            chat_prompt = langfuse.get_prompt(
                "muallim-system-prompt",
                type="chat",
            )

            compiled_prompt = chat_prompt.compile(
                context=context,
                language=language,
            )

            logger.info(
                pformat(compiled_prompt, width=120)
            )
            messages = [
                *compiled_prompt,
                *history,
                HumanMessage(content=query)
                ]
                
            # Create a nested generation for an LLM call
        with langfuse.start_as_current_observation(
            as_type="generation", 
            name="llm-response",
            model=_MODEL_NAME) as generation:
                try:
                    # Step 4: Generate response using LLM

                    logger.info(
                        pformat(messages, width=120)
                    )
                    llm = _get_llm()
                    response = llm.invoke(messages)
                    generation.update(
                    input={"messages": messages},
                    output={"response": response.content},
                    usage_details={
                        # to calculate Cost Per User, Cost Per Session, Token Consumption
                        "input": response.usage_metadata["input_tokens"],
                        "output": response.usage_metadata["output_tokens"]
                    }
                )
                except Exception as e:
                    logger.warning("LLM failed: %s", e)
                    generation.update(output={"error": str(e)})
                    raise RuntimeError("LLM failed") from e 
                    
        with langfuse.start_as_current_observation(as_type="span", name="tts-request") as span:
            try:
                audio_file = tts_service.synthesize(response.content)
                span.update(output={
                    "audio_length": len(audio_file),
                    "characters_sent": len(response.content)
                })
            except Exception as e:
                logger.warning("TTS failed, returning text only: %s", e)
                span.update(output={"error": str(e), "fallback": "text-only"})
                audio_file = b"" 
        updated_history = [
        *history,
        HumanMessage(content=query),
        AIMessage(content=response.content)
        ]
        langfuse.flush()
        return response.content, audio_file, updated_history, rewrite

# ── private API ────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_llm() -> ChatGroq:
    """
    Get a ChatGroq instance for generating responses.

    Returns:
        A ChatGroq instance initialized with the appropriate model and API key.
    """
    return ChatGroq(
        model=_MODEL_NAME,
        api_key=_GROQ_API_KEY,
    )

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
    prompt = langfuse.get_prompt(
        "muallim-rewrite_query-prompt",
        type="text",
    )

    recent_history = history[-4:]

    compiled_prompt = prompt.compile(
        history=format_history(recent_history),
        query=query,
    )

    llm = _get_llm_for_query_rewriter()

    try:
        response = llm.invoke(compiled_prompt)
        return response.content.strip()

    except Exception:
        logger.exception("Query rewriting failed.")
        return query
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