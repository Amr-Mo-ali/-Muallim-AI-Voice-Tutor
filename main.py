"""
main.py - FastAPI server for STUDYFLOW AI
Responsibilities:
  - /upload  → accept PDF, build/load vector DB (idempotent)
  - /ask     → answer questions using the session's vector DB
  - /ui      → serve the frontend
Session recovery: если server restarts, sessions are reloaded from disk on demand.
"""
from __future__ import annotations
from functools import lru_cache
import json
import os

from fastapi import FastAPI ,File ,UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import redis
from services.rag.service import load_and_chunk, load_vector_db ,get_or_create_vector_store ,count_chunks
from orchestrator.chain import ask as process_ask 
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage 

import base64
import uuid
import asyncio
import logging


from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
# ── logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
# ── env ───────────────────────────────────────────────────────────────────────
load_dotenv()
# ── app setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="STUDYFLOW AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://192.168.1.7:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# ── constants ─────────────────────────────────────────────────────────────────
_MAX_HISTORY = 10
# ── infrastructure ────────────────────────────────────────────────────────────
# ThreadPoolExecutor: blocking ops (embedding, LLM)
executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=4)
# In-memory session store: {session_id: {"vector_db": Chroma, "persist_dir": str}}
# NOTE: بيتمسح على server restart لكن بنعمل recovery من الديسك تلقائياً
_sessions: dict[str, dict] = {}
# ── helpers ───────────────────────────────────────────────────────────────────
def _serialize_history(history: list) -> str:
    return json.dumps([
        {"role": "human" if isinstance(msg, HumanMessage) else "ai",
         "content": msg.content}
        for msg in history
    ])


def _deserialize_history(history_str: str) -> list:
    return [
        HumanMessage(content=msg["content"]) if msg["role"] == "human"
        else AIMessage(content=msg["content"])
        for msg in json.loads(history_str)
    ]

async def _run_blocking(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, fn, *args)

@lru_cache(maxsize=1)
def _get_redis() -> redis.Redis:
    return redis.Redis.from_url(os.getenv("REDIS_URL"), decode_responses=True)

# ── routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return{"Message": "Wellcome to STUDYFLOW AI"}

@app.get("/ui")
def ui():
    return FileResponse("muallim.html")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), ):
    session_id = str(uuid.uuid4())
    bytes_data = await file.read() 
    chunks = await _run_blocking(load_and_chunk, bytes_data)
    collection_name = f"session_{session_id}"

    r = _get_redis()

    r.hset(f"session:{session_id}", mapping={
    "collection_name": collection_name,
    "history": "[]"
    })
    r.expire(f"session:{session_id}", 60 * 60 * 24)  # 24 ساعة
    # indexing أو load من cache — كله في executor لأنه blocking
    vector_db = await _run_blocking(get_or_create_vector_store, chunks, collection_name)

    _sessions[session_id] = {
    "vector_db":   vector_db,
    "collection_name": collection_name,
    "history":     [],          
    }

    chunk_count = await _run_blocking(count_chunks, collection_name)
    return {"session_id": session_id, "chunks": chunk_count}

@app.post("/ask")
async def ask(
    session_id: str = Form(...),
    audio_file: UploadFile = File(...),
):
    r = _get_redis()

    # 1. جيب الـ session من Redis
    session_data = r.hgetall(f"session:{session_id}")
    if not session_data:
        raise HTTPException(404, "Session not found — please re-upload the PDF.")

    # 2. الـ audio
    audio_bytes = await audio_file.read()
    if not audio_bytes:
        raise HTTPException(400, "Audio file is empty.")

    # 3. الـ vector_db من cache أو Qdrant
    vector_db = _sessions.get(session_id, {}).get("vector_db")
    if not vector_db:
        vector_db = await _run_blocking(load_vector_db, session_data["collection_name"])
        _sessions[session_id] = {"vector_db": vector_db}

    # 4. الـ history من Redis
    history = _deserialize_history(session_data["history"])
    # 5. process
    answer, response_audio, updated_history, query = await _run_blocking(
        process_ask, audio_bytes, history, vector_db,
    )

    # 6. حفظ الـ history
    trimmed = updated_history[-_MAX_HISTORY:]
    r.hset(f"session:{session_id}", "history", _serialize_history(trimmed))

    audio_b64 = base64.b64encode(response_audio).decode("utf-8")
    return {"answer": answer, "audio": audio_b64, "audio_format": "mp3", "query": query}