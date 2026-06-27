class MuallimError(Exception):
    """Base exception for the application."""

class ValidationError(MemoryError):
    "Invalid user input."

class VectorStoreError(MemoryError):
    """Qdrant operation failed."""

class EmbeddingError(MemoryError):
    """Embedding generation failed."""

class STTError(MemoryError):
    """Speech-to-text failed."""

class LLMError(MuallimError):
    """LLM request failed."""


class PromptError(MuallimError):
    """Prompt compilation failed."""


class CollectionError(VectorStoreError):
    """Collection management failed."""