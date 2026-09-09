# ClinicalLens 🔬

RAG-powered Q&A over pharmaceutical PDF documents (clinical study reports,
protocols, prescribing information, etc.).

Upload PDFs → they are parsed, chunked, and embedded into a local **ChromaDB**
collection → ask natural-language questions → **Gemini** answers *only* from
the retrieved passages, with `[Source N]` citations mapping to filename + page.

## Architecture

```
frontend/app.py          Streamlit UI (upload, document list, chat)
        │  HTTP
        ▼
backend/main.py          FastAPI: /upload  /query  /documents  /documents/{id}
        │
        ▼
backend/rag.py           RAGPipeline: embed query → retrieve top-k across ALL
                         documents → grounded prompt → Gemini → cited answer
        │
        ├── backend/pdf_parser.py     PyMuPDF text extraction (page-by-page)
        ├── backend/utils.py          TextChunker (RecursiveCharacterTextSplitter)
        ├── backend/embeddings.py     sentence-transformers all-MiniLM-L6-v2
        ├── backend/database.py       ChromaDBManager (persistent local store)
        └── backend/prompts.py        grounded prompt templates
```

## Setup

**1. Clone and create a virtual environment**

```bash
git clone https://github.com/ShantanuBorhade-bit/ClinicalLens.git
cd ClinicalLens
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure environment variables**

```bash
cp .env.example .env            # Windows: copy .env.example .env
```

Edit `.env` and set your Gemini API key (get one free at
https://aistudio.google.com/apikey):

```
GEMINI_API_KEY=your_key_here
```

The first run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB).

## Run

**Terminal 1 — backend:**

```bash
uvicorn backend.main:app --reload
# → http://127.0.0.1:8000  (interactive docs at /docs)
```

**Terminal 2 — frontend:**

```bash
streamlit run frontend/app.py
# → http://localhost:8501
```

> Run both from the project root (the folder containing `backend/`), so
> relative paths like `chroma_db/` resolve correctly.

## Usage

1. Upload one or more PDFs in the sidebar → they are parsed, chunked,
   embedded, and stored in ChromaDB.
2. Ask questions in the chat box. Retrieval runs across **all** uploaded
   documents, and every answer cites `[Source N]` → filename + page (expand
   **📚 Sources** under the answer).
3. Delete any document from the sidebar; its chunks are removed from the
   collection immediately.
4. If nothing is uploaded, the app shows a clear warning and Gemini will
   answer "Not found in the documents."

## API reference

| Method   | Endpoint                  | Description                          |
|----------|---------------------------|--------------------------------------|
| `GET`    | `/health`                 | Status, key check, total chunk count |
| `POST`   | `/upload`                 | Multipart PDF → parse → embed → store |
| `POST`   | `/query`                  | `{"question": "...", "top_k": 4}` → grounded answer + sources |
| `GET`    | `/documents`              | List uploaded documents              |
| `DELETE` | `/documents/{document_id}`| Delete a document and all its chunks |

## Configuration

All settings are env vars (see `.env.example`): `GEMINI_API_KEY`,
`GEMINI_MODEL`, `GEMINI_TEMPERATURE`, `TOP_K`, `CHUNK_SIZE`/`CHUNK_OVERLAP`
(constants in `backend/config.py`), `UPLOAD_FOLDER`, `CHROMA_DB_PATH`,
`API_HOST`/`API_PORT`, `BACKEND_URL`, `MAX_UPLOAD_SIZE_MB`.

## Notes & troubleshooting

- **`chroma_db/` and `uploads/` are local data**, not code — keep them out of
  git (see `.gitignore`). If `chroma_db/` was ever committed, remove it with
  `git rm -r --cached chroma_db && git commit` (local files are kept).
- **Scanned PDFs** without a text layer can't be parsed (no OCR support) —
  the upload endpoint returns a clear 422 in that case.
- **502 mentioning daily free-tier quota / 429 per-day** → the free tier
  allows only ~20 requests/day **per model**. The pipeline now defaults to
  the lightest model (`gemini-flash-lite-latest`) and automatically falls
  back through `GEMINI_FALLBACK_MODELS` (each has its own quota bucket).
  Quotas reset at midnight Pacific Time; enable billing on your Google AI
  Studio project for higher limits.
- **Query returns 503** → `GEMINI_API_KEY` is missing on the backend (or is
  still the placeholder from `.env.example`).
- **Query returns 502 mentioning 504 / Deadline Exceeded after several
  attempts** → Gemini was unreachable or overloaded; the pipeline already
  retried with backoff (`GEMINI_MAX_RETRIES`, `GEMINI_RETRY_BASE_DELAY` in
  `.env`). Usually resolves itself; increase retries for flaky networks.
- **502 "API key not valid"** → the key is wrong — create a fresh one at
  https://aistudio.google.com/apikey (run `python check_key.py` to verify).
- **502 mentioning a retired model** → set `GEMINI_MODEL=gemini-flash-latest`
  in `.env` (the default) so you always track the current stable model.
- **Deleting/renaming a collection by hand** — use the UI; editing the
  `chroma_db/` folder directly can corrupt the store.
