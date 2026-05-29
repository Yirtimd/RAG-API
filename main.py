import os
from pathlib import Path
from contextlib import asynccontextmanager

import io
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from fastapi import FastAPI, UploadFile, File, HTTPException
from openai import OpenAI
from pydantic import BaseModel
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

# --- Set ChromaDB and LLM via start --- 

DB_PATH = str(Path(__file__).parent / 'chroma_db')
embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name='paraphrase-multilingual-MiniLM-L12-v2'
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = chromadb.PersistentClient(path=DB_PATH)
    app.state.collection = app.state.db.get_or_create_collection(
        name='documents', embedding_function=embedding_fn
    )
    app.state.llm = OpenAI(
        api_key=os.environ['OPENROUTER_API_KEY'],
        base_url=os.environ['BASE_URL'],
    )
    yield

app = FastAPI(lifespan=lifespan)


# --- Pydantic model ---

class IngestResponse(BaseModel):
    filename: str
    chunks: int
    total_in_collection: int

class QueryRequest(BaseModel):
    question: str
    n_results: int = 3

class Source(BaseModel):
    filename: str
    distance: float

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]

# --- Auxiliary functions ---

def extract_text(contents: bytes) -> str:
    reader = PdfReader(io.BytesIO(contents))
    return '\n'.join(page.extract_text() or '' for page in reader.pages)

def chunks_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start: start + chunk_size])
        start += chunk_size - overlap
    return chunks


# --- Endpoint ---

@app.get('/health')
def health():
    return {'status': 'ok', 'docs_indexed': app.state.collection.count()}

@app.post('/ingest', response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail='Только PDF файлы')

    contents = await file.read()
    text = extract_text(contents)

    if not text.strip():
        raise HTTPException(status_code=422, detail='Не удалось извлечь текст из PDF файла')
    
    collection = app.state.collection

    # remove olds chunk this file

    existing = collection.get(where={'source': file.filename})
    if existing['ids']:
        collection.delete(ids=existing['ids'])

    chunks = chunks_text(text)
    collection.upsert(
        documents=chunks,
        ids=[f'{Path(file.filename).stem}_chunk_{i}' for i in range(len(chunks))],
        metadatas=[{'source': file.filename, 'chunk': i } for i in range(len(chunks))],
    )

    return IngestResponse(filename=file.filename, chunks=len(chunks), total_in_collection=collection.count())


@app.post('/query', response_model=QueryResponse)
async def query(req: QueryRequest):

    collection = app.state.collection

    if collection.count() == 0:
        raise HTTPException(status_code=400, detail='Коллекция пуста — сначала загрузите документы через /ingest')
    
    results = collection.query(
        query_texts=[req.question],
        n_results=req.n_results,
    )

    chunks = results['documents'][0]
    metadatas = results['metadatas'][0]
    distance = results['distances'][0]

    context = '\n\n --- \n\n'.join(chunks)
    prompt = f'''Ответь на вопрос используя только информацию из контекста ниже.
    Если ответа нет в контексте - скажи об этом честно. Не выдумывая сама.
    
    Контекст:
    {context}
    
    Вопрос:
    {req.question} '''

    response = app.state.llm.chat.completions.create(
        model='google/gemma-4-26b-a4b-it:free',
        messages=[{'role': 'user', 'content': prompt}],
    )

    return QueryResponse(
        answer=response.choices[0].message.content,
        sources=[
            Source(filename=m['source'], distance=round(d, 3))
            for m, d in zip(metadatas, distance)
        ],
    )

