# مُعلِّم — Muallim AI Voice Tutor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green?style=flat-square&logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-1.3-orange?style=flat-square&logo=langchain)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

**Production-grade Arabic AI voice tutor — speak your question, get a spoken answer.**

[🚀 Live Demo](https://enthusiastic-simplicity-production.up.railway.app/ui) · [📖 API Docs](https://enthusiastic-simplicity-production.up.railway.app/docs) · [🐛 Report Bug](https://github.com/Amr-Mo-ali/muallim/issues)

</div>

---

## Overview

**Muallim** (مُعلِّم — Arabic for "teacher") is an Arabic-first AI voice tutor that transforms any PDF document into an interactive voice-based learning experience.

Upload a study material → Ask a question in Arabic → Get a spoken answer grounded in the document.

Built for **Egyptian Arabic dialect** support, with a full RAG pipeline, real-time TTS, and session memory.

---

## Demo

> **Try it live:** [enthusiastic-simplicity-production.up.railway.app/ui](https://enthusiastic-simplicity-production.up.railway.app/ui)

![مُعلِّم UI](https://github.com/Amr-Mo-ali/-Muallim-AI-Voice-Tutor/raw/main/docs/WhatsApp%20Image%202026-06-14%20at%206.29.16%20PM.jpeg)

---

## Pipeline

```
Voice Input
    ↓
Whisper large-v3 (Groq)      — Arabic speech-to-text
    ↓
ChromaDB + MMR Search         — semantic retrieval from PDF
    ↓
LLaMA 3.3 70B (Groq)         — context-aware answer generation
    ↓
ElevenLabs Multilingual v2    — text-to-speech
    ↓
Voice + Text Output
```

---

## Tech Stack

| Layer | Technology |
|:---|:---|
| **STT** | Whisper large-v3 via Groq API |
| **LLM** | LLaMA 3.3 70B via Groq API |
| **TTS** | ElevenLabs `eleven_multilingual_v2` |
| **Embeddings** | `paraphrase-multilingual-MiniLM-L12-v2` (HuggingFace) |
| **Vector Store** | ChromaDB with MMR retrieval |
| **Backend** | FastAPI + asyncio + ThreadPoolExecutor |
| **Orchestration** | LangChain + LangSmith tracing |
| **Deployment** | Docker + Docker Compose + Railway |

---

## Key Features

- **Arabic-first** — optimized for Egyptian Arabic dialect via Whisper large-v3
- **RAG pipeline** — answers grounded strictly in uploaded PDF content
- **Idempotent indexing** — SHA-256 hash prevents re-embedding the same PDF
- **Session memory** — conversation history maintained per session
- **Session recovery** — vector store reloaded from disk on server restart
- **Async architecture** — blocking AI calls run in ThreadPoolExecutor
- **MMR retrieval** — diverse chunk selection for better context coverage

---

## Project Structure

```
muallim/
├── services/
│   ├── stt/service.py        # Whisper STT via Groq
│   ├── rag/service.py        # ChromaDB + MMR retrieval
│   └── tts/service.py        # ElevenLabs TTS
├── orchestrator/
│   └── chain.py              # Pipeline coordinator
├── muallim.html              # Arabic UI (single-file frontend)
├── main.py                   # FastAPI server
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Quick Start

### Using Docker (Recommended)

```bash
git clone https://github.com/Amr-Mo-ali/muallim.git
cd muallim
cp .env.example .env        # fill in your API keys
docker-compose up --build
```

Open `http://localhost:8000/ui`

### Local Development

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn main:app --reload
```

---

## Environment Variables

```env
GROQ_API_KEY=gsk_...
ELEVENLABS_API_KEY=sk_...
ELEVENLABS_VOICE_ID=...
LANGCHAIN_API_KEY=...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=muallim
CHROMA_DB_DIR=data/chroma_db
```

---

## API Reference

| Endpoint | Method | Description |
|:---|:---|:---|
| `/` | GET | Health check |
| `/ui` | GET | Arabic frontend |
| `/upload` | POST | Upload PDF → returns `session_id` |
| `/ask` | POST | Send audio → returns text + base64 audio |
| `/docs` | GET | Interactive API docs |

### `/upload` Response
```json
{
  "session_id": "uuid",
  "chunks": 42
}
```

### `/ask` Response
```json
{
  "answer": "الإجابة بالعربي...",
  "audio": "base64_mp3...",
  "audio_format": "mp3",
  "query": "السؤال المنطوق"
}
```

---

## Architecture Decisions

| Decision | Choice | Reason |
|:---|:---|:---|
| STT model | Whisper large-v3 | Best Arabic dialect support |
| LLM | LLaMA 3.3 70B | Strong Arabic + free via Groq |
| Embeddings | multilingual-MiniLM | Covers Arabic + English PDFs |
| Retrieval | MMR over similarity | Diversity prevents redundant chunks |
| Temp files | `delete=False` + manual cleanup | Windows file lock compatibility |
| Async | ThreadPoolExecutor | Keeps FastAPI non-blocking |

---

## License

MIT License — see [LICENSE](LICENSE)

---

<div align="center">

Built by **[Amr Mohamed Ali](https://github.com/Amr-Mo-ali)** · ML/AI Engineer · Arabic NLP Specialist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-amr--mohamed21-0077B5?style=flat-square&logo=linkedin)](https://linkedin.com/in/amr-mohamed21)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-AmrMohamed21-FFD21E?style=flat-square&logo=huggingface)](https://huggingface.co/AmrMohamed21)

</div>
