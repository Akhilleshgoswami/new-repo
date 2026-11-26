
from langchain_core.documents import Document

from rag_sys.vector_store import get_vector_store


def retrieve_documents(
    query: str,
    k: int = 3,
) -> list[Document]:
    """
    Retrieve the most relevant documents from ChromaDB.
    """

    vector_store = get_vector_store()

    results = vector_store.similarity_search(
        query=query,
        k=k,
    )

    return results


if __name__ == "__main__":
    query = "My payment was deducted but my subscription isn't active."

    documents = retrieve_documents(query)

    print("User question:", query)
    print("\nRetrieved documents:", len(documents))

    for index, document in enumerate(documents):
        print("\n" + "=" * 50)
        print(f"Result {index + 1}")
        print("Source:", document.metadata.get("source"))
        print("Content:", document.page_content)
