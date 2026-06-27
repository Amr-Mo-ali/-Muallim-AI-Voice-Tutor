"""
What contract does this module promise?
If Settings() succeeds,
every consumer can trust the configuration. 
"""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    app_name: str = "Muallim"
    debug: bool = False

    # Qdrant
    qdrant_url: str
    qdrant_api_key: SecretStr

    # Hugging Face
    hf_token: SecretStr

    # Redis
    redis_url: str

    # Groq
    groq_api_key: SecretStr

    # Langfuse
    langfuse_public_key: SecretStr
    langfuse_secret_key: SecretStr
    langfuse_host: str

    # ELEVENLABS
    elevenlabs_api_key: SecretStr
    elevenlabs_voice_id: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()