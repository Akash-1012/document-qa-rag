import os
from pathlib import Path

import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

APP_DIR = Path(__file__).parent
DOCS_DIR = APP_DIR / "documents"
SUPPORTED_EXTENSIONS = {".txt", ".md"}

st.set_page_config(page_title="Document Q&A — RAG", page_icon="📚", layout="wide")
st.title("📚 Document Q&A — Retrieval-Augmented Answers")
st.write("Ask questions about your own documents. Answers are grounded in retrieved passages.")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def read_documents(uploaded_files):
    docs = []
    for path in DOCS_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                docs.append((path.name, path.read_text(encoding="utf-8")))
            except UnicodeDecodeError:
                st.warning(f"Could not read {path.name}; please save it as UTF-8 text.")
    for uploaded in uploaded_files:
        if Path(uploaded.name).suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.append((uploaded.name, uploaded.getvalue().decode("utf-8", errors="replace")))
    # Keep non-empty documents only.
    return [(name, text.strip()) for name, text in docs if text.strip()]

def chunk_text(text, chunk_size=900, overlap=150):
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks

def build_passages(documents):
    passages = []
    for filename, text in documents:
        for index, chunk in enumerate(chunk_text(text), start=1):
            passages.append({"source": filename, "chunk": index, "text": chunk})
    return passages

def retrieve(question, passages, model, top_k=3):
    passage_texts = [p["text"] for p in passages]
    vectors = model.encode([question] + passage_texts, normalize_embeddings=True)
    scores = cosine_similarity(vectors[0:1], vectors[1:]).ravel()
    ranked = np.argsort(scores)[::-1][:top_k]
    return [(passages[i], float(scores[i])) for i in ranked]

with st.sidebar:
    st.header("Add documents")
    uploaded_files = st.file_uploader(
        "Upload .txt or .md files",
        type=["txt", "md"],
        accept_multiple_files=True,
        help="You can also place .txt or .md files in the documents/ folder."
    )
    st.caption("For privacy, uploaded files are used in this session and are not saved to disk.")

documents = read_documents(uploaded_files or [])
if not documents:
    st.info("Upload a .txt or .md file, or add a sample document to the documents/ folder, to get started.")
    st.stop()

passages = build_passages(documents)
if not passages:
    st.warning("No readable text was found in the selected documents.")
    st.stop()

model = load_model()
st.success(f"Loaded {len(documents)} document(s) and created {len(passages)} text chunk(s).")

question = st.text_input("Ask a question about your documents", placeholder="What is the main topic discussed?")
top_k = st.slider("Number of source passages to show", min_value=1, max_value=5, value=3)

if st.button("Find answer", type="primary", disabled=not question.strip()):
    results = retrieve(question.strip(), passages, model, top_k=top_k)
    # A conservative relevance threshold avoids pretending that unrelated text answers the question.
    best_score = results[0][1] if results else 0.0
    if not results or best_score < 0.25:
        st.subheader("Answer")
        st.warning("Not found in the provided documents. Try asking a different question or adding a more relevant document.")
    else:
        st.subheader("Answer from your documents")
        st.write(results[0][0]["text"])
        st.caption(
            f"Source: {results[0][0]['source']} · Chunk: {results[0][0]['chunk']} · "
            f"Similarity: {results[0][1]:.3f}"
        )
        st.caption(
            "This is an extractive answer: it quotes the most relevant passage rather than "
            "inventing details beyond the documents."
        )
    st.subheader("Retrieved passages")
    for rank, (passage, score) in enumerate(results, start=1):
        with st.expander(f"[{rank}] {passage['source']} — chunk {passage['chunk']} (similarity: {score:.3f})", expanded=(rank == 1)):
            st.write(passage["text"])
            st.caption(f"Source: {passage['source']} · Chunk: {passage['chunk']}")

st.divider()
st.caption("RAG pipeline: document loading → chunking → embedding → similarity search → cited evidence.")
