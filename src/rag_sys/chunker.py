
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[Document],
) -> list[Document]:
    """
    Split documents into smaller chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":
    from rag_sys.ingestion import load_documents

    documents = load_documents()

    chunks = split_documents(documents)

    print(f"Total documents: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks):
        print("\n" + "=" * 50)
        print(f"Chunk: {index + 1}")
        print("Source:", chunk.metadata["source"])
        print("Content:", chunk.page_content)
