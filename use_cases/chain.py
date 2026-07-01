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


# ── logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
# ── env ───────────────────────────────────────────────────────────────────────
_GROQ_API_KEY = settings.groq_api_key.get_secret_value()
# ── constants ─────────────────────────────────────────────────────────────────
_MODEL_NAME = "llama-3.3-70b-versatile"  # choose the appropriate model for your use case
_MODEL_NAME_FOR_QUERY_REWRITER = "llama-3.1-8b-instant"
langfuse = get_client()
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
    with langfuse.start_as_current_observation(as_type="trace", name="ask-request") as trace :
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
        with langfuse.start_as_current_observation(
            input=("rewrite", rewrite),
            as_type="span", 
            name="relevant_chunks") as span:
            # Step 2: Retrieve relevant chunks from vector store
            rewrite = rewrite_query(query, history)
            relevant_chunks = rag_service.search_vector_db(vector_store, rewrite)
            logger.info("Retrieved %d relevant chunks from vector store", len(relevant_chunks))
            # Step 3: build context string
            context = "\n\n".join(chunk.page_content for chunk in relevant_chunks)
            span.update(
            output={
                "chunks_count": len(relevant_chunks),
                "context_length": len(context)
            }
        )
        chat_prompt = langfuse.get_prompt(
            "muallim-system-prompt",
            type="chat"
        )

        compiled_prompt = chat_prompt.compile(
            context=context,
            language=language
        )

        messages = [
            *compiled_prompt,
            *history,
            HumanMessage(content=query),
        ]
         # Create a nested generation for an LLM call
        with langfuse.start_as_current_observation(
            as_type="generation", 
            name="llm-response",
            input=context,
            model=_MODEL_NAME) as generation:
                try:
                    # Step 4: Generate response using LLM
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

def rewrite_query(query: str, history: list) -> str:
    """
    Rewrite the user's query into a retrieval-optimized query.

    Args:
        query: Original user query.
        history: Conversation history.

    Returns:
        A rewritten query optimized for retrieval.
        Falls back to the original query if rewriting fails.
    """

    prompt = langfuse.get_prompt(
        "muallim-rewrite_query-prompt",
        type="chat",
    )

    llm = _get_llm_for_query_rewriter()

    compiled_prompt = prompt.compile(
        query=query,
        history=history,
    )

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