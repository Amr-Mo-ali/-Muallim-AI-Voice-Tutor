"""
main.py - FastAPI server for STUDYFLOW AI
Responsibilities:
  - /upload  → accept PDF, build/load vector DB (idempotent)
  - /ask     → answer questions using the session's vector DB
  - /ui      → serve the frontend
Session recovery: если server restarts, sessions are reloaded from disk on demand.
"""
from __future__ import annotations

from fastapi import FastAPI ,File ,UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from services.rag.service import load_and_chunk, load_vector_db ,get_or_create_vector_store ,search_vector_db
from orchestrator.chain import ask as process_ask  

import base64
import uuid
import asyncio
import logging


from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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
def _data_dir() -> Path:
    p = Path("data")
    p.mkdir(exist_ok=True)
    return p

async def _run_blocking(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, fn, *args)

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
    persist_dir = str(_data_dir() / f"chroma_{session_id}")
    # indexing أو load من cache — كله في executor لأنه blocking
    vector_db = await _run_blocking(get_or_create_vector_store, chunks, persist_dir)

    _sessions[session_id] = {
    "vector_db":   vector_db,
    "persist_dir": persist_dir,
    "history":     [],          
    }

    # نحسب عدد الـ chunks من الـ collection مش من الـ PDF تاني
    chunk_count = vector_db._collection.count()

    return {"session_id": session_id, "chunks": chunk_count}

@app.post("/ask")
async def ask(
    session_id: str = Form(...),
    audio_file: UploadFile = File(...),

):
    """Answer a question using the session's vector DB."""
    session = _sessions.get(session_id)
    logger.info("Processing ask request for session %s", session_id)
    # ── session recovery after server restart ─────────────────────────────
    audio_bytes = await audio_file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty.")
    if session is None:
        persist_dir = str(_data_dir() / f"chroma_{session_id}")
        if not Path(persist_dir).exists():
            raise HTTPException(
                status_code=404,
                detail="Session not found — please re-upload the PDF.",
            )
        vector_db = await _run_blocking(load_vector_db, persist_dir)
        session = {"vector_db": vector_db, "persist_dir": persist_dir, "history": []}
        _sessions[session_id] = session

    answer, response_audio, updated_history, query= await _run_blocking(
        process_ask,
        audio_bytes,
        session["history"],
        session["vector_db"],
    )
    session["history"] = updated_history[-_MAX_HISTORY:] 
    audio_b64 = base64.b64encode(response_audio).decode("utf-8")

    return {
    "answer": answer,
    "audio": audio_b64,
    "audio_format": "mp3",
    "query": query        
}