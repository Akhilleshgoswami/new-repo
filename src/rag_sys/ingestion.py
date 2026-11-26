
from pathlib import Path

from langchain_core.documents import Document


from rag_sys.config import PROJECT_ROOT


KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "data" / "knowledge_base"


def load_documents() -> list[Document]:
    """
    Load all TXT documents from the knowledge base.
    """

    documents = []

    if not KNOWLEDGE_BASE_DIR.exists():
        raise FileNotFoundError(
            f"Knowledge base directory not found: {KNOWLEDGE_BASE_DIR}"
        )

    for file_path in KNOWLEDGE_BASE_DIR.glob("*.txt"):

        content = file_path.read_text(encoding="utf-8")

        if not content.strip():
            continue

        document = Document(
            page_content=content,
            metadata={
                "source": file_path.name,
                "file_path": str(file_path),
                "file_type": file_path.suffix,
            },
        )

        documents.append(document)

    if not documents:
        raise ValueError("No documents found in the knowledge base.")

    return documents


if __name__ == "__main__":
    loaded_documents = load_documents()

    print(f"Total documents loaded: {len(loaded_documents)}")

    for document in loaded_documents:
        print("\n------------------------------")
        print("Source:", document.metadata["source"])
        print("Content:")
        print(document.page_content)
