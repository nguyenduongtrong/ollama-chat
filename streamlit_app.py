import os
import io
import pdfplumber
import numpy as np
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
import uuid
from tqdm import tqdm

# ============================================
# CONFIG
# ============================================
CHROMA_DIR = "data/vector_store"
UPLOAD_DIR = "data/uploaded_docs"
MODEL_EMBED = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
# Default model to call on the LLM server; can be overridden via env var or UI
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gemma:2b")
TOP_K = 6
RERANK_TOP_N = 3
MAX_HISTORY = 5
# ============================================

# ---------- INIT ----------
@st.cache_resource
def get_embedder():
    return SentenceTransformer(MODEL_EMBED)

@st.cache_resource
def get_chroma():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    if "docs" not in [c.name for c in client.list_collections()]:
        return client.create_collection("docs")
    return client.get_collection("docs")

# ---------- PDF PARSER ----------
def extract_text_from_pdf(file_bytes):
    texts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            texts.append({"page": i+1, "text": text})
    return texts

def chunk_text(text, chunk_size=800, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks

# ---------- INGEST ----------
def ingest_pdf(file_bytes, file_name):
    embedder = get_embedder()
    collection = get_chroma()

    pages = extract_text_from_pdf(file_bytes)
    docs, metadatas, ids = [], [], []
    for p in pages:
        for chunk in chunk_text(p["text"]):
            if chunk.strip():
                docs.append(chunk)
                metadatas.append({"page": p["page"], "source": file_name})
                ids.append(str(uuid.uuid4()))

    if not docs:
        return 0

    embeddings = embedder.encode(docs, convert_to_numpy=True)
    collection.add(documents=docs, metadatas=metadatas, ids=ids, embeddings=embeddings.tolist())
    return len(docs)

# ---------- RETRIEVE ----------
def retrieve(query, top_k=TOP_K):
    embedder = get_embedder()
    collection = get_chroma()
    # Encode the query
    q_emb = embedder.encode([query], convert_to_numpy=True)[0]

    # Query the collection for candidate documents
    try:
        res = collection.query(query_embeddings=[q_emb.tolist()], n_results=top_k, include=["documents", "metadatas"])
    except Exception:
        return []

    # Handle empty results
    docs_list = []
    docs_texts = None
    metas = None
    # Normalize results shape safely
    docs_texts = []
    metas = []
    try:
        docs_container = res.get("documents")
        metas_container = res.get("metadatas")
        if docs_container and len(docs_container) > 0 and docs_container[0]:
            docs_texts = docs_container[0]
        if metas_container and len(metas_container) > 0 and metas_container[0]:
            metas = metas_container[0]
    except Exception:
        docs_texts = []
        metas = []

    if not docs_texts:
        return []

    # Compute embeddings for returned docs in one batch, then compute cosine similarity scores
    try:
        doc_embs = embedder.encode(docs_texts, convert_to_numpy=True)
        scores = cosine_similarity(q_emb.reshape(1, -1), doc_embs)[0]
    except Exception:
        # Fallback: give equal low score
        scores = [0.0] * len(docs_texts)

    for d, m, s in zip(docs_texts, metas, scores):
        docs_list.append({"text": d, "meta": m, "score": float(s)})

    docs_sorted = sorted(docs_list, key=lambda x: x["score"], reverse=True)
    return docs_sorted[:RERANK_TOP_N]

# ---------- LLM CALL ----------
def ask_llm(prompt):
    # Backwards-compatible wrapper that accepts a model argument.
    # Keep original simple signature for calls that don't pass a model.
    return ask_llm_with_model(prompt, model=DEFAULT_MODEL)


def ask_llm_with_model(prompt, model=DEFAULT_MODEL, headers=None, timeout=120):
    """Call the LLM HTTP endpoint with a configurable model name.

    Returns a string (answer) or an informative error message.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        # Request a non-streaming final response when supported by the server.
        # Some LLM servers return incremental/streamed chunks with {"response":..., "done": false}.
        # Asking for stream=False (and setting max_new_tokens) will usually return a finished answer.
        "stream": False,
        "max_new_tokens": 512,
        "temperature": 0.1,
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, headers=(headers or {}), timeout=timeout)
    except requests.RequestException as e:
        return f"Error calling LLM (network): {e}"

    if not (200 <= r.status_code < 300):
        body = ""
        try:
            body = r.json()
        except Exception:
            body = r.text or ""
        return f"Error calling LLM: status {r.status_code} - {body}"

    # Parse JSON or return raw text
    try:
        data = r.json()
    except ValueError:
        return r.text or ""

    # Handle common response shapes
    if isinstance(data, dict):
        if data.get("text"):
            return data.get("text")
        # Some servers return {'response': '...', 'done': bool}
        if data.get("response") is not None:
            resp = data.get("response")
            # If generation isn't finished, include a clear debug note so the UI shows it's partial
            if data.get("done") is False:
                try:
                    raw = json.dumps(data, ensure_ascii=False)
                except Exception:
                    raw = str(data)
                return f"{resp}\n\n[Note: generation incomplete (done=false). Raw response: {raw}]"
            return resp

        out = data.get("output")
        if out:
            if isinstance(out, list):
                parts = []
                for o in out:
                    if isinstance(o, dict):
                        parts.append(o.get("content") or o.get("text") or "")
                    else:
                        parts.append(str(o))
                joined = "\n".join([p for p in parts if p])
                if joined:
                    return joined
            elif isinstance(out, dict):
                return out.get("content") or out.get("text") or str(out)
            else:
                return str(out)

        choices = data.get("choices")
        if choices and isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                return first.get("text") or first.get("message") or first.get("content") or ""
            return str(first)

    if isinstance(data, str):
        return data

    return ""

# ---------- PROMPT BUILDER ----------
def build_prompt(query, retrieved, history):
    history_str = ""
    if history:
        for h in history[-MAX_HISTORY:]:
            history_str += f"User: {h['user']}\nAssistant: {h['bot']}\n"
    context = "\n\n".join([f"[Nguồn {i+1}] Trang {d['meta'].get('page')} ({d['meta'].get('source')}):\n{d['text']}"
                           for i, d in enumerate(retrieved)])
    # Strong Vietnamese instruction + optional few-shot examples to force direct answers in Vietnamese
    instruction = (
        "LUÔN LUÔN trả lời trực tiếp câu hỏi dưới đây bằng tiếng Việt. "
        "KHÔNG hỏi lại để làm rõ. Nếu thông tin không có trong nguồn, hãy trả lời rõ ràng 'Tôi không biết'."
    )

    # A couple of short examples in Vietnamese to bias the model toward the desired behavior
    examples = """
Ví dụ:
Q: Nước được tạo thành từ gì?
A: Nước (H2O) được tạo từ hai nguyên tử hydro và một nguyên tử oxy. [Nguồn 1]

Q: Tài liệu có đề cập đến lịch sử của công ty X không?
A: Tôi không biết. (Không tìm thấy thông tin liên quan trong các nguồn đã cung cấp.)
"""

    prompt = f"""
{instruction}

{examples}

Lịch sử hội thoại:
{history_str}

Nguồn thông tin:
{context}

Câu hỏi của người dùng: {query}

Vui lòng viết câu trả lời ngắn gọn, chính xác bằng tiếng Việt và trích dẫn các nguồn theo định dạng [Nguồn 1, Nguồn 2].
"""
    return prompt

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Local RAG Chat", layout="wide")
st.title("📚 Local RAG Chat with Session Memory & Auto Ingest")

# INIT session state
if "history" not in st.session_state:
    st.session_state.history = []

uploaded = st.file_uploader("📄 Upload a PDF to use as knowledge base", type=["pdf"])
if uploaded:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    bytes_data = uploaded.getvalue()
    file_path = os.path.join(UPLOAD_DIR, uploaded.name)
    with open(file_path, "wb") as f:
        f.write(bytes_data)
    with st.spinner("Indexing document..."):
        chunks_added = ingest_pdf(bytes_data, uploaded.name)
    st.success(f"✅ Indexed {chunks_added} text chunks from {uploaded.name}")

# Sidebar controls for model and endpoint
with st.sidebar:
    st.header("LLM settings")
    selected_model = st.text_input("Model (e.g. gemma:2b)", value=DEFAULT_MODEL)
    ollama_url = st.text_input("LLM endpoint URL", value=OLLAMA_URL)
    # Allow quick endpoint probe
    if st.button("Test LLM endpoint"):
        probe_payload = {"model": selected_model, "prompt": "Kiểm tra kết nối. Hãy trả lời: Xin chào"}
        try:
            r = requests.post(ollama_url, json=probe_payload, timeout=10)
            try:
                body = r.json()
            except Exception:
                body = r.text
            st.write({"status": r.status_code, "body": body})
        except Exception as e:
            st.write({"error": str(e)})

st.markdown("---")

question = st.text_input("💬 Ask a question:")
if st.button("Ask") and question.strip():
    with st.spinner("🔍 Retrieving relevant info..."):
        retrieved = retrieve(question)

    prompt = build_prompt(question, retrieved, st.session_state.history)

    with st.spinner("🤖 Thinking..."):
        # Call LLM with the selected model and endpoint
        # Update the OLLAMA_URL used by the LLM call in memory
        OLLAMA_URL = ollama_url
        answer = ask_llm_with_model(prompt, model=selected_model)

    st.session_state.history.append({"user": question, "bot": answer})

    st.markdown("### 🧠 Answer:")
    st.write(answer)

    st.markdown("### 🔗 Sources used:")
    for i, d in enumerate(retrieved):
        st.write(f"**Source {i+1}:** {d['meta'].get('source')} (page {d['meta'].get('page')}) — score {d['score']:.3f}")

st.markdown("---")
st.markdown("### 🕒 Conversation History")
for h in st.session_state.history[-MAX_HISTORY:]:
    st.markdown(f"**You:** {h['user']}")
    st.markdown(f"**Bot:** {h['bot']}")
    st.markdown("---")
