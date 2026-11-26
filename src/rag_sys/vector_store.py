
from langchain_chroma import Chroma

from rag_sys.config import settings
from rag_sys.embedder import get_embedding_model


def get_vector_store() -> Chroma:
    """
    Initialize or connect to the persistent ChromaDB collection.
    """

    embeddings = get_embedding_model()

    vector_store = Chroma(
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_directory),
    )

    return vector_store


def store_documents(documents) -> Chroma:
    """
    Store document chunks and their embeddings in ChromaDB.
    """

    vector_store = get_vector_store()

    vector_store.add_documents(documents)

    return vector_store


if __name__ == "__main__":
    from rag_sys.ingestion import load_documents
    from rag_sys.chunker import split_documents

    documents = load_documents()
    chunks = split_documents(documents)

    vector_store = store_documents(chunks)

    print("Documents stored successfully!")
    print("Total chunks:", len(chunks))
    print("Collection:", settings.CHROMA_COLLECTION_NAME)
    print("Database:", settings.chroma_directory)
