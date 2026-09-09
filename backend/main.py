# backend/main.py

"""
FastAPI backend for ClinicalLens.

Endpoints:
    GET    /health                   -> service status
    POST   /upload                   -> parse -> chunk -> embed -> store a PDF
    POST   /query                    -> RAG answer with citations
    GET    /documents                -> list uploaded documents
    DELETE /documents/{document_id}  -> delete one document and its chunks

Run with:
    uvicorn backend.main:app --reload
"""

import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import (
    ALLOWED_EXTENSIONS,
    GEMINI_API_KEY,
    MAX_UPLOAD_SIZE_MB,
    UPLOAD_FOLDER,
)
from backend.database import ChromaDBManager
from backend.embeddings import EmbeddingGenerator
from backend.pdf_parser import PDFParser
from backend.rag import get_pipeline
from backend.utils import TextChunker


# ---------------------------------------------------------------------- #
# Shared singletons (created once at startup, reused by every request)   #
# ---------------------------------------------------------------------- #

parser: PDFParser | None = None
chunker: TextChunker | None = None
embedder: EmbeddingGenerator | None = None
db: ChromaDBManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global parser, chunker, embedder, db
    Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

    parser = PDFParser()
    chunker = TextChunker()
    embedder = EmbeddingGenerator()  # loads all-MiniLM-L6-v2 once
    db = ChromaDBManager()

    if not GEMINI_API_KEY:
        print(
            "WARNING: GEMINI_API_KEY is not set — /query will return 503. "
            "Copy .env.example to .env and add your key."
        )
    yield


app = FastAPI(
    title="ClinicalLens API",
    description="RAG Q&A over pharmaceutical PDF documents",
    version="1.0.0",
    lifespan=lifespan,
)

# The Streamlit frontend runs on a different port, so allow its origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------- #
# Schemas                                                                #
# ---------------------------------------------------------------------- #


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class SourceOut(BaseModel):
    source: str
    page: int
    document_id: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceOut]


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    num_pages: int
    num_chunks: int


class DocumentOut(BaseModel):
    document_id: str
    filename: str


def _require_ready() -> None:
    if parser is None or chunker is None or embedder is None or db is None:
        raise HTTPException(status_code=503, detail="Backend is still starting up.")


# The .env.example template ships with this placeholder value; treat it as
# "no key" so /health and /query don't report a configured key that isn't real.
_GEMINI_PLACEHOLDER = "your_gemini_api_key_here"


def _gemini_ready() -> bool:
    return bool(GEMINI_API_KEY) and _GEMINI_PLACEHOLDER not in GEMINI_API_KEY


# ---------------------------------------------------------------------- #
# Endpoints                                                              #
# ---------------------------------------------------------------------- #


@app.get("/health")
def health():
    return {
        "status": "ok",
        "gemini_key_set": _gemini_ready(),
        "total_chunks": db.collection.count() if db else 0,
    }


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    """Upload a PDF, then parse -> chunk -> embed -> store in ChromaDB."""
    _require_ready()

    filename = Path(file.filename or "").name  # strips any path components
    if not filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Only {', '.join(sorted(ALLOWED_EXTENSIONS))} files are supported.",
        )

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {MAX_UPLOAD_SIZE_MB} MB).",
        )
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file.")

    # Save inside a unique per-upload folder, keeping the ORIGINAL filename.
    # PDFParser uses the filename as the citation source, so it must stay
    # clean; the unique folder name prevents "report (1).pdf" style
    # collisions and odd characters in filenames.
    upload_dir = Path(UPLOAD_FOLDER) / uuid.uuid4().hex[:8]
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_path = upload_dir / filename
    saved_path.write_bytes(contents)

    try:
        pages = parser.extract_text(str(saved_path))
        if not pages:
            raise HTTPException(
                status_code=422,
                detail="No extractable text found in this PDF "
                       "(it may be a scanned image — OCR is not supported).",
            )

        chunks = chunker.chunk_pages(pages)

        # Re-upload of the same filename replaces the old version's chunks
        # so retrieval never mixes stale and current content.
        db.delete_document_by_filename(filename)

        embeddings = embedder.embed_chunks(chunks)
        embeddings = [[float(v) for v in vector] for vector in embeddings]

        document_id = db.add_chunks(chunks, embeddings)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to process PDF: {exc}"
        ) from exc
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)  # parsed; keep no copy

    return UploadResponse(
        document_id=document_id,
        filename=filename,
        num_pages=len(pages),
        num_chunks=len(chunks),
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Ask a question; get a grounded answer plus source citations."""
    _require_ready()

    if not _gemini_ready():
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured (or still the placeholder "
                   "from .env.example). Paste your real key into .env and "
                   "restart the backend.",
        )

    question = (request.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        result = get_pipeline().ask(question, top_k=request.top_k or None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return QueryResponse(
        question=question,
        answer=result.answer,
        sources=result.sources,
    )


@app.get("/documents", response_model=list[DocumentOut])
def list_documents():
    """List all uploaded documents."""
    _require_ready()
    documents = db.list_documents()
    return [DocumentOut(document_id=doc_id, filename=name)
            for doc_id, name in documents.items()]


@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    """Delete one uploaded document and all of its chunks."""
    _require_ready()

    # Validate that it's a UUID so invalid ids get a clean 400/404 instead of
    # silently deleting nothing.
    try:
        uuid.UUID(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid document ID.") from exc

    documents = db.list_documents()
    if document_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found.")

    db.delete_document(document_id)
    return {"detail": f"Deleted '{documents[document_id]}'",
            "document_id": document_id}
