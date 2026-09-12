import os
import re
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from config import MODEL_NAME
from utils.loaders import load_and_split_pdfs, load_and_split_single_pdf

# ── Paths ──────────────────────────────────────────────────────────
VECTOR_STORE_BASE = "data/vector_store"          # per-PDF stores go here
LEGACY_VECTOR_STORE_PATH = "data/vector_store"   # kept for backward compat

# FAISS cosine similarity: 1.0 = identical, 0.0 = unrelated
RELEVANCE_THRESHOLD = 0.25


def get_embeddings():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from config import GOOGLE_API_KEY
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)


# ── Helpers ────────────────────────────────────────────────────────
def _pdf_stem(filename: str) -> str:
    """Convert 'Module 1.pdf' → 'Module_1' (safe directory name)."""
    stem = os.path.splitext(filename)[0]
    return re.sub(r'[^\w\-]', '_', stem)


def _pdf_store_path(filename: str) -> str:
    """Return the FAISS directory for a specific PDF filename."""
    return os.path.join(VECTOR_STORE_BASE, _pdf_stem(filename))


# ── Per-PDF vector store ────────────────────────────────────────────
def build_vector_store_for_pdf(pdf_path: str) -> None:
    """Build (or rebuild) a FAISS index for a single PDF file.
    Saves to data/vector_store/<pdf_stem>/.
    """
    filename = os.path.basename(pdf_path)
    store_path = _pdf_store_path(filename)
    os.makedirs(store_path, exist_ok=True)

    chunks = load_and_split_single_pdf(pdf_path)
    if not chunks:
        raise ValueError(f"No content extracted from '{filename}'.")

    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(store_path)


def load_vector_store_for_pdf(filename: str) -> FAISS:
    """Load the FAISS index for a specific PDF filename."""
    store_path = _pdf_store_path(filename)
    if not os.path.exists(store_path):
        raise FileNotFoundError(
            f"No vector store found for '{filename}'. Upload it first."
        )
    embeddings = get_embeddings()
    return FAISS.load_local(
        store_path, embeddings, allow_dangerous_deserialization=True
    )


def get_academic_chain_for_pdfs(llm, pdf_filenames: list):
    """Return a RetrievalQA chain that searches across ALL given PDF filenames.

    If multiple PDFs are active for a session, their FAISS indices are merged
    in-memory so retrieval spans all of them — but never leaks into other PDFs
    not in this session.
    """
    if not pdf_filenames:
        raise ValueError("No PDFs specified.")

    embeddings = get_embeddings()
    merged_store = None

    for fname in pdf_filenames:
        store_path = _pdf_store_path(fname)
        if not os.path.exists(os.path.join(store_path, "index.faiss")):
            continue  # skip PDFs whose index hasn't been built yet
        store = FAISS.load_local(
            store_path, embeddings, allow_dangerous_deserialization=True
        )
        if merged_store is None:
            merged_store = store
        else:
            merged_store.merge_from(store)

    if merged_store is None:
        raise FileNotFoundError("None of the specified PDFs have an index yet.")

    retriever = merged_store.as_retriever(search_kwargs={"k": 4})
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )


def is_query_relevant_to_pdfs(question: str, pdf_filenames: list,
                               threshold: float = RELEVANCE_THRESHOLD) -> bool:
    """Check if the question is relevant to any of the session's active PDFs."""
    try:
        embeddings = get_embeddings()
        merged_store = None

        for fname in pdf_filenames:
            store_path = _pdf_store_path(fname)
            if not os.path.exists(os.path.join(store_path, "index.faiss")):
                continue
            store = FAISS.load_local(
                store_path, embeddings, allow_dangerous_deserialization=True
            )
            if merged_store is None:
                merged_store = store
            else:
                merged_store.merge_from(store)

        if merged_store is None:
            return False

        results = merged_store.similarity_search_with_relevance_scores(question, k=3)
        if not results:
            return False
        best_score = max(score for _, score in results)
        return best_score >= threshold
    except Exception:
        return False


# ── Legacy (kept for backward compatibility) ────────────────────────
def build_vector_store(dir_path):
    chunks = load_and_split_pdfs(dir_path)
    if not chunks:
        return None
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(LEGACY_VECTOR_STORE_PATH)
    return vector_store


def load_vector_store():
    embeddings = get_embeddings()
    return FAISS.load_local(
        LEGACY_VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


def get_academic_chain(llm):
    if os.path.exists(LEGACY_VECTOR_STORE_PATH):
        vector_store = load_vector_store()
    else:
        raise FileNotFoundError("No vector store found.")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=False
    )


def is_query_relevant_to_docs(question: str, threshold: float = RELEVANCE_THRESHOLD) -> bool:
    try:
        if not os.path.exists(LEGACY_VECTOR_STORE_PATH):
            return False
        vector_store = load_vector_store()
        results = vector_store.similarity_search_with_relevance_scores(question, k=3)
        if not results:
            return False
        best_score = max(score for _, score in results)
        return best_score >= threshold
    except Exception:
        return False


def academic_query(chain, question):
    result = chain.invoke({"query": question})
    return result["result"]

