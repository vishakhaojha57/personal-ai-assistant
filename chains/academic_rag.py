import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from config import MODEL_NAME
from utils.loaders import load_and_split_pdfs

VECTOR_STORE_PATH = "data/vector_store"

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_vector_store(dir_path):
    chunks = load_and_split_pdfs(dir_path)
    if not chunks:
        return None
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    return vector_store


def load_vector_store():
    embeddings = get_embeddings()
    return FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


def get_academic_chain(llm):
    if os.path.exists(VECTOR_STORE_PATH):
        vector_store = load_vector_store()
    else:
        raise FileNotFoundError("No vector store found. Add a PDF first using build_vector_store().")

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )


def academic_query(chain, question):
    result = chain.invoke({"query": question})
    return result["result"]
