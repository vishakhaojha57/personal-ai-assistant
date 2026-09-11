from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_pdfs(dir_path, chunk_size=1000, chunk_overlap=200):
    """Load and split all PDFs in a directory (legacy / bulk use)."""
    loader = PyPDFDirectoryLoader(dir_path)
    documents = loader.load()

    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def load_and_split_single_pdf(file_path: str, chunk_size=1000, chunk_overlap=200):
    """Load and split a single PDF file.
    Each chunk's metadata will contain 'source' = the original file path,
    which lets us filter by document later.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)