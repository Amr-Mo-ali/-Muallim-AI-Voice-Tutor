

from __future__ import annotations

import logging
from config import settings
from elevenlabs import ElevenLabs
from functools import lru_cache

# ── logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
# ── env ──────────────────────────────────────────────────────────────────────
_ELEVENLABS_API_KEY = settings.elevenlabs_api_key.get_secret_value()
_ELEVENLABS_VOICE_ID = settings.elevenlabs_voice_id
# ── constants ─────────────────────────────────────────────────────────────────

_MODEL_NAME = "eleven_multilingual_v2"  # choose the appropriate model for your use case

# ── private API ────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_client() -> ElevenLabs:
    """
    Get a ELEVENLABS_API_KEY client instance.

    Returns:
        A ELEVENLABS_API_KEY client instance initialized with the API key.

    Exceptions:
        Raises:
            ValueError: If the ELEVENLABS_API_KEY is not set in the environment variables.
    """
    if not _ELEVENLABS_API_KEY:
        raise ValueError("ELEVENLABS_API_KEY is not set in the environment variables.")
    return ElevenLabs(api_key=_ELEVENLABS_API_KEY)

# ── public API ────────────────────────────────────────────────────────────────

def synthesize(text: str) -> bytes:
    """
    Convert text to speech using ElevenLabs.

    Returns:
        Audio data as bytes (MP3).

    Raises:
        ValueError: If text is empty.
        RuntimeError: If ElevenLabs API fails.
    """
    logger.info("Synthesizing %d characters", len(text))
    # Validate input
    if not text:
        raise ValueError("Text file cannot be empty")
    
    clint = _get_client()
    try:
        audio_generator = clint.text_to_speech.convert(
            voice_id=_ELEVENLABS_VOICE_ID,
            text=text,
            model_id=_MODEL_NAME
        )
    except Exception as e:
        raise RuntimeError("Failed to synthesize text.") from e
    
    return b"".join(audio_generator)  # generator → bytes