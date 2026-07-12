"""
FastAPI entrypoint.

Responsibilities:
    - Manage HTTP requests.
    - Manage user sessions.
    - Delegate work to application services.

This module intentionally contains no RAG business logic.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import redis
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)
from opentelemetry.instrumentation.threading import (
    ThreadingInstrumentor,
)

from config import settings
from services.rag.rag_pipeline import (
    answer_question,
    index_documents,
)
from services.stt import service as stt_service
from services.tts import service as tts_service

# ──────────────────────────────────────────────────────────────────────────────
# logging
# ──────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# app
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Muallim",
    version="1.0.0",
)

ThreadingInstrumentor().instrument()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# constants
# ──────────────────────────────────────────────────────────────────────────────

_MAX_HISTORY = 10

# ──────────────────────────────────────────────────────────────────────────────
# infrastructure
# ──────────────────────────────────────────────────────────────────────────────

executor = ThreadPoolExecutor(max_workers=4)

_sessions: dict[str, dict] = {}

# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────


def _serialize_history(history: list) -> str:
    return json.dumps(
        [
            {
                "role": "human" if isinstance(msg, HumanMessage) else "ai",
                "content": msg.content,
            }
            for msg in history
        ]
    )


def _deserialize_history(history: str) -> list:
    return [
        HumanMessage(content=item["content"])
        if item["role"] == "human"
        else AIMessage(content=item["content"])
        for item in json.loads(history)
    ]


async def _run_blocking(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, fn, *args)


@lru_cache(maxsize=1)
def _get_redis() -> redis.Redis:
    return redis.Redis.from_url(
        settings.redis_url,
        decode_responses=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# routes
# ──────────────────────────────────────────────────────────────────────────────


@app.get("/")
def root():
    return {"message": "Welcome to Muallim"}


@app.get("/ui")
def ui():
    return FileResponse("muallim.html")


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
):
    session_id = str(uuid.uuid4())

    bytes_data = await file.read()

    collection_name = f"session_{session_id}"

    await _run_blocking(
        index_documents,
        bytes_data,
        collection_name,
    )

    redis_client = _get_redis()

    redis_client.hset(
        f"session:{session_id}",
        mapping={
            "collection_name": collection_name,
            "history": "[]",
        },
    )

    redis_client.expire(
        f"session:{session_id}",
        60 * 60 * 24,
    )

    logger.info(
        "Session %s created",
        session_id,
    )

    return {
        "session_id": session_id,
    }


@app.post("/chat/text")
async def chat_text(
    session_id: str = Form(...),
    question: str = Form(...),
    language: str = Form(default="Arabic"),
):
    redis_client = _get_redis()

    session = redis_client.hgetall(
        f"session:{session_id}"
    )

    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found.",
        )

    history = _deserialize_history(
        session["history"]
    )

    answer, updated_history = await _run_blocking(
        answer_question,
        query=question,
        history=history,
        language=language,
        collection_name=session["collection_name"],
    )

    redis_client.hset(
        f"session:{session_id}",
        "history",
        _serialize_history(
            updated_history[-_MAX_HISTORY:]
        ),
    )

    return {
        "answer": answer,
    }


@app.post("/chat/audio")
async def chat_audio(
    session_id: str = Form(...),
    audio_file: UploadFile = File(...),
):
    session = _get_redis().hgetall(
        f"session:{session_id}"
    )

    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found.",
        )

    audio = await audio_file.read()

    query, language = await _run_blocking(
        stt_service.transcribe,
        audio,
    )

    history = _deserialize_history(
        session["history"]
    )

    answer, updated_history = await _run_blocking(
        answer_question,
        query=query,
        history=history,
        language=language,
        collection_name=session["collection_name"],
    )

    speech = await _run_blocking(
        tts_service.synthesize,
        answer,
    )

    _get_redis().hset(
        f"session:{session_id}",
        "history",
        _serialize_history(
            updated_history[-_MAX_HISTORY:]
        ),
    )

    return {
        "answer": answer,
        "query": query,
        "audio": base64.b64encode(speech).decode(),
        "audio_format": "mp3",
    }