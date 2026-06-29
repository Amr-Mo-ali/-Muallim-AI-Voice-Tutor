"""
Service for speech-to-text (STT) functionality using the Groq API.
This module provides a function to transcribe audio bytes into text and identify the language of the transcription. 
It uses the "whisper-large-v3" model from Groq for transcription.
The `transcribe` function takes audio data as bytes, validates the input, and returns the transcribed text along with the detected language. 
It raises exceptions for invalid input or transcription failures.
"""
from __future__ import annotations
import logging
from config import settings
from groq import Groq 
from functools import lru_cache

# ── logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
# ── env ──────────────────────────────────────────────────────────────────────
_GROQ_API_KEY = settings.groq_api_key.get_secret_value()
# ── constants ─────────────────────────────────────────────────────────────────
_MODEL_NAME = "whisper-large-v3"  # choose the appropriate model for your use case
# ── private API ────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_client() -> Groq:
    """
    Get a Groq client instance.

    Returns:
        A Groq client instance initialized with the API key.

    Exceptions:
        Raises:
            ValueError: If the GROQ_API_KEY is not set in the environment variables.
    """
    if not _GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set in the environment variables.")
    return Groq(api_key=_GROQ_API_KEY)
# ── public API ────────────────────────────────────────────────────────────────

def transcribe(audio_bytes: bytes) -> tuple[str, str]:
    """
    Transcribe the given audio bytes to text.

    Args:
        audio_bytes: The audio data as bytes.

    Returns:
        The transcribed text.
        language: The language of the transcribed text.

    Exceptions:
        Raises:
            ValueError: If audio_bytes is not a valid audio format or is empty.
            RuntimeError: If the whisper model fails to transcribe the audio.
    """
    logger.info("Transcribing %d bytes of audio", len(audio_bytes))
    # Validate input
    if not audio_bytes:
        raise ValueError("Audio bytes cannot be empty.")

    client = _get_client()
    try:
        result = client.audio.transcriptions.create(
            model=_MODEL_NAME,
            file=("audio.wav", audio_bytes),
            response_format="verbose_json"  # We want the language information, so we use verbose_json format
        )
    except Exception as e:
        logger.exception()
        raise RuntimeError("Failed to transcribe audio.") from e

    text = result.text
    language = result.language
    return text, language  # Return the actual transcribed text and language