from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_pdfs(dir_path, chunk_size=1000, chunk_overlap=200):
    loader = PyPDFDirectoryLoader(dir_path)
    documents = loader.load()
    
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
    )

    chunks = splitter.split_documents(documents)
    return chunks