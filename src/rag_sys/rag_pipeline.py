
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from rag_sys.config import settings
from rag_sys.retriever import retrieve_documents


SYSTEM_PROMPT = """
You are a helpful customer support assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not make up information.
2. If the answer is not available in the context,
   clearly say that you don't have enough information.
3. Keep answers simple, clear and professional.
4. Do not assume policies that are not mentioned.
5. Include relevant policy details when available.

Context:
{context}
"""


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


llm = ChatOpenAI(
    model=settings.LLM_MODEL,
    temperature=0,
    api_key=settings.OPENAI_API_KEY,
)


def answer_question(question: str) -> dict:
    """
    Complete RAG pipeline:
    1. Retrieve relevant documents.
    2. Prepare context.
    3. Send context and question to LLM.
    4. Return generated answer and sources.
    """

    documents = retrieve_documents(question, k=3)

    if not documents:
        return {
            "question": question,
            "answer": "I couldn't find relevant information in the knowledge base.",
            "sources": [],
        }

    context_parts = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown")

        context_parts.append(
            f"""
Document {index}
Source: {source}
Content:
{document.page_content}
"""
        )

    context = "\n\n".join(context_parts)

    chain = prompt | llm

    response = chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    sources = list(
        dict.fromkeys(
            document.metadata.get("source", "Unknown")
            for document in documents
        )
    )

    return {
        "question": question,
        "answer": response.content,
        "sources": sources,
    }


if __name__ == "__main__":

    question = "Can I get a refund if I purchased my subscription 3 days ago?"

    result = answer_question(question)

    print("\nQUESTION:")
    print(result["question"])

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    for source in result["sources"]:
        print("-", source)
