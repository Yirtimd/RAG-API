# RAG API

A production-ready REST API for Retrieval-Augmented Generation (RAG). Upload PDF documents, ask questions, get answers grounded in your documents.

**Live:** https://rag-api-production-e10c.up.railway.app

## Features

- **POST /ingest** — upload a PDF and index it in a vector database
- **POST /query** — ask a question, get an answer with source references
- **GET /health** — service status and number of indexed documents

## Tech Stack

- **FastAPI** — REST API
- **ChromaDB** — vector database for semantic search
- **sentence-transformers** — text embeddings (`all-MiniLM-L6-v2`)
- **OpenRouter** — LLM inference (free models)
- **Docker** — containerization
- **Railway** — cloud deployment

## Installation

```bash
git clone https://github.com/Yirtimd/RAG-API.git
cd RAG-API
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:

## Usage

```bash
uvicorn main:app --reload
```

**Index a PDF:**
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@document.pdf"
```

**Ask a question:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "what is RAG?"}'
```

**Response:**
```json
{
  "answer": "RAG stands for Retrieval-Augmented Generation...",
  "sources": [
    {"filename": "document.pdf", "distance": 0.266}
  ]
}
```

## Docker

```bash
docker build -t rag-api .
docker run -p 8000:8000 \
  -e OPENROUTER_API_KEY=your_key \
  -e BASE_URL=https://openrouter.ai/api/v1 \
  rag-api
```

## How It Works

1. **Ingest:** PDF → extract text → split into chunks → embed → store in ChromaDB
2. **Query:** question → embed → find top-3 similar chunks → pass to LLM → answer

Documents persist across restarts via ChromaDB's local storage.