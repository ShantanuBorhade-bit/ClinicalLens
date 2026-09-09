# frontend/app.py

"""
ClinicalLens — Streamlit UI.

Run with:
    streamlit run frontend/app.py

Requires the FastAPI backend to be running:
    uvicorn backend.main:app --reload
"""

import os

import requests
import streamlit as st
from dotenv import load_dotenv

# Read BACKEND_URL (and friends) from <project root>/.env so both servers
# can be configured from one file.
load_dotenv()

# ---------------------------------------------------------------------- #
# Page config + backend URL                                              #
# ---------------------------------------------------------------------- #

st.set_page_config(
    page_title="ClinicalLens",
    page_icon="🔬",
    layout="wide",
)

# BACKEND_URL comes from <project root>/.env (loaded above), or defaults to
# the local FastAPI server. NOTE: don't use st.secrets here — it raises
# StreamlitSecretNotFoundError when no secrets.toml exists at all.
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")


def api(path: str) -> str:
    return f"{BACKEND_URL}{path}"


# ---------------------------------------------------------------------- #
# Session state                                                          #
# ---------------------------------------------------------------------- #

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, sources}

if "documents" not in st.session_state:
    st.session_state.documents = []  # list of {document_id, filename}


def refresh_documents() -> None:
    try:
        response = requests.get(api("/documents"), timeout=10)
        response.raise_for_status()
        st.session_state.documents = response.json()
    except requests.RequestException as exc:
        st.sidebar.error(f"Cannot reach backend: {exc}")
        st.session_state.documents = []


# ---------------------------------------------------------------------- #
# Sidebar: health, upload, document list                                 #
# ---------------------------------------------------------------------- #

st.sidebar.title("🔬 ClinicalLens")
st.sidebar.caption("RAG Q&A over pharmaceutical PDFs")

try:
    health = requests.get(api("/health"), timeout=5).json()
    if health.get("gemini_key_set"):
        st.sidebar.success("Backend online")
    else:
        st.sidebar.warning(
            "Backend online, but GEMINI_API_KEY is not set — "
            "questions will fail until you add it to .env."
        )
except requests.RequestException:
    st.sidebar.error(
        f"Backend not reachable at `{BACKEND_URL}`.\n\n"
        "Start it with:\n"
        "`uvicorn backend.main:app --reload`"
    )

st.sidebar.header("Upload documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload pharmaceutical PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    help="Parsed, chunked, embedded, and stored in ChromaDB.",
)

if uploaded_files and st.sidebar.button("Process uploads", type="primary"):
    progress = st.sidebar.progress(0)
    for i, uploaded_file in enumerate(uploaded_files, start=1):
        try:
            response = requests.post(
                api("/upload"),
                files={"file": (uploaded_file.name, uploaded_file.getvalue(),
                                "application/pdf")},
                timeout=300,
            )
            if response.status_code == 200:
                data = response.json()
                st.sidebar.success(
                    f"✅ {data['filename']}: {data['num_pages']} pages → "
                    f"{data['num_chunks']} chunks"
                )
            else:
                detail = response.json().get("detail", response.text)
                st.sidebar.error(f"❌ {uploaded_file.name}: {detail}")
        except requests.RequestException as exc:
            st.sidebar.error(f"❌ {uploaded_file.name}: {exc}")
        progress.progress(i / len(uploaded_files))
    refresh_documents()

st.sidebar.header("Uploaded documents")
if st.sidebar.button("🔄 Refresh list"):
    refresh_documents()

refresh_documents()

if not st.session_state.documents:
    st.sidebar.info("No documents uploaded yet.")
else:
    for document in st.session_state.documents:
        col1, col2 = st.sidebar.columns([4, 1])
        col1.markdown(f"📄 **{document['filename']}**")
        if col2.button("🗑️", key=f"del_{document['document_id']}",
                       help=f"Delete {document['filename']}"):
            try:
                response = requests.delete(
                    api(f"/documents/{document['document_id']}"), timeout=30
                )
                if response.status_code == 200:
                    st.sidebar.success(f"Deleted {document['filename']}")
                    refresh_documents()
                    st.rerun()
                else:
                    detail = response.json().get("detail", response.text)
                    st.sidebar.error(f"Delete failed: {detail}")
            except requests.RequestException as exc:
                st.sidebar.error(f"Delete failed: {exc}")

# ---------------------------------------------------------------------- #
# Main area: chat                                                        #
# ---------------------------------------------------------------------- #

st.title("Ask your documents")

if not st.session_state.documents:
    st.warning(
        "⚠️ **No documents uploaded yet.** Upload one or more PDFs in the "
        "sidebar — until then, questions can't be answered from your library."
    )

# Replay conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"- `{source['source']}` — page {source['page']}"
                    )

if prompt := st.chat_input(
    "Ask a question about your documents..."
    if st.session_state.documents
    else "Upload a document first, then ask a question..."
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating an answer..."):
            try:
                response = requests.post(
                    api("/query"),
                    json={"question": prompt},
                    # Long enough to cover backend retries with backoff
                    # (GEMINI_MAX_RETRIES x GEMINI_TIMEOUT_SECONDS + delays).
                    timeout=300,
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                else:
                    answer = None
                    error_detail = response.json().get("detail", response.text)
                    st.error(f"Backend error: {error_detail}")
                    sources = []
            except requests.RequestException as exc:
                answer = None
                st.error(f"Could not reach the backend: {exc}")
                sources = []

        if answer is not None:
            st.markdown(answer)
            if sources:
                with st.expander("📚 Sources"):
                    for source in sources:
                        st.markdown(f"- `{source['source']}` — page {source['page']}")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
