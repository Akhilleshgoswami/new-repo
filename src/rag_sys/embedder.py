
from langchain_openai import OpenAIEmbeddings

from rag_sys.config import settings


def get_embedding_model() -> OpenAIEmbeddings:
    """
    Initialize OpenAI embedding model.
    """

    embeddings = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )

    return embeddings


if __name__ == "__main__":
    embeddings = get_embedding_model()

    text = "Customers can request a refund within 7 days."

    vector = embeddings.embed_query(text)

    print("Original text:", text)
    print("Vector dimensions:", len(vector))
    print("First 10 values:", vector[:10])
