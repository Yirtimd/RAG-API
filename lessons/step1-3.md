# Проект 1.3 — RAG API: шаги 1–3

## Архитектура

```
POST /ingest  ←  PDF файл  →  ChromaDB (индексация)
POST /query   ←  вопрос    →  ChromaDB (поиск) → LLM → ответ + источники
GET  /health  →  статус + кол-во проиндексированных документов
```

## Ключевая концепция: lifespan

В 0.4/0.5 мы создавали клиентов на уровне модуля. В большом приложении лучше использовать `lifespan` — он инициализирует ресурсы при старте и закрывает при остановке.

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Код до yield — выполняется при старте
    app.state.collection = chromadb.PersistentClient(...).get_or_create_collection(...)
    app.state.llm = OpenAI(...)
    yield
    # Код после yield — при остановке (закрыть соединения, etc.)

app = FastAPI(lifespan=lifespan)
```

`app.state` — специальный объект FastAPI для хранения глобального состояния приложения. Доступен в любом эндпоинте через `app.state.collection`.

## Pydantic модели запроса и ответа

```python
class QueryRequest(BaseModel):
    question: str
    n_results: int = 3          # поле с default = необязательное в JSON

class Source(BaseModel):
    filename: str
    distance: float

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]       # вложенная модель
```

## /ingest — загрузка и индексация PDF

```python
@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Только PDF файлы")

    contents = await file.read()
    text = extract_text(contents)                     # bytes → текст через PdfReader

    existing = collection.get(where={"source": file.filename})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])        # удалить старые чанки

    chunks = chunk_text(text)
    collection.upsert(documents=chunks, ids=[...], metadatas=[...])

    return IngestResponse(filename=file.filename, chunks=len(chunks), ...)
```

## /query — поиск + генерация

```python
@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    results = collection.query(query_texts=[req.question], n_results=req.n_results)

    chunks    = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Собираем контекст из найденных чанков
    context = "\n\n---\n\n".join(chunks)

    response = llm.chat.completions.create(
        model="google/gemma-3-4b-it:free",
        messages=[{"role": "user", "content": prompt}],
    )

    return QueryResponse(
        answer=response.choices[0].message.content,
        sources=[Source(filename=m["source"], distance=round(d, 3))
                 for m, d in zip(metadatas, distances)],
    )
```

## Тестирование через curl

```bash
# Индексация
curl -X POST http://localhost:8000/ingest \
  -F "file=@doc.pdf"
# → {"filename":"doc.pdf","chunks":12,"total_in_collection":12}

# Вопрос по документу
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "что такое эмбеддинг?"}'
# → {"answer":"...","sources":[{"filename":"doc.pdf","distance":0.671},...]}

# Вопрос вне документа — честный ответ + высокие дистанции (0.96+)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "какая столица Франции?"}'
# → {"answer":"В контексте нет ответа...","sources":[...distance:0.958...]}

# Не-PDF — 400
curl -X POST http://localhost:8000/ingest -F "file=@main.py"
# → {"detail":"Только PDF файлы"}
```

## Что узнали

- `lifespan` — правильный способ инициализировать ресурсы в FastAPI (ChromaDB, LLM клиент)
- `app.state` — хранилище глобального состояния приложения
- Вложенные Pydantic-модели: `list[Source]` внутри `QueryResponse`
- Высокие дистанции (> 0.9) = сигнал что вопрос вне документа — модель должна это признать
- Разделение ответственности: `/ingest` — только индексация, `/query` — только поиск+генерация
