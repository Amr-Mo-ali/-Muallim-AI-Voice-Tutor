"""
FastAPI entrypoint.

Responsibilities
----------------
- Expose HTTP endpoints.
- Manage user sessions.
- Delegate work to the application layer.

No business logic lives here.
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

from config import settings

from services.rag.rag_pipeline import (
    index_documents,
)

from use_cases.chain import ask

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Muallim",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=4)

_MAX_HISTORY = 10


# ---------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_redis() -> redis.Redis:
    return redis.Redis.from_url(
        settings.redis_url,
        decode_responses=True,
    )


# ---------------------------------------------------------------------
# Async helper
# ---------------------------------------------------------------------

async def run_blocking(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()

    return await loop.run_in_executor(
        executor,
        lambda: fn(*args, **kwargs),
    )


# ---------------------------------------------------------------------
# History
# ---------------------------------------------------------------------

def serialize_history(history):
    return json.dumps(
        [
            {
                "role": "human" if isinstance(m, HumanMessage) else "ai",
                "content": m.content,
            }
            for m in history
        ]
    )


def deserialize_history(data):
    return [
        HumanMessage(content=x["content"])
        if x["role"] == "human"
        else AIMessage(content=x["content"])
        for x in json.loads(data)
    ]


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Welcome to Muallim"
    }


@app.get("/ui")
def ui():
    return FileResponse("muallim.html")


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
):
    session_id = str(uuid.uuid4())

    collection = f"session_{session_id}"

    pdf = await file.read()

    await run_blocking(
        index_documents,
        pdf,
        collection,
    )

    redis = get_redis()

    redis.hset(
        f"session:{session_id}",
        mapping={
            "collection_name": collection,
            "history": "[]",
        },
    )

    redis.expire(
        f"session:{session_id}",
        86400,
    )

    logger.info(
        "Session created %s",
        session_id,
    )

    return {
        "session_id": session_id,
    }


@app.post("/ask")
async def ask_audio(
    session_id: str = Form(...),
    audio_file: UploadFile = File(...),
):

    redis = get_redis()

    session = redis.hgetall(
        f"session:{session_id}"
    )

    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found.",
        )

    audio = await audio_file.read()

    history = deserialize_history(
        session["history"]
    )

    answer, audio_response, history = await run_blocking(
        ask,
        audio_bytes=audio,
        history=history,
        collection_name=session["collection_name"],
    )

    redis.hset(
        f"session:{session_id}",
        "history",
        serialize_history(
            history[-_MAX_HISTORY:]
        ),
    )

    return {
        "answer": answer,
        "audio": base64.b64encode(audio_response).decode(),
        "audio_format": "mp3",
    }