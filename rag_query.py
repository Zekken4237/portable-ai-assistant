import json
from urllib import error, request

from rag_index import DOCS_DIR, EMBED_MODEL_NAME, INDEX_DIR, LLM_MODEL_NAME, build_index

OLLAMA_URL = "http://localhost:11434/api/generate"
INDEX_FILE = INDEX_DIR / "index.faiss"


def load_vector_store():
    try:
        from langchain_community.vectorstores import FAISS
    except ModuleNotFoundError:
        print(
            "Missing dependencies. Install `langchain-community` in the Python "
            "environment running this script."
        )
        return None

    try:
        from rag_index import get_ollama_embeddings
    except ImportError:
        print("Could not import the embedding configuration from rag_index.py.")
        return None

    embeddings = get_ollama_embeddings()

    if not INDEX_FILE.exists():
        print(f"Index not found at {INDEX_FILE}. Building it from PDFs in {DOCS_DIR}...")
        try:
            build_index()
        except (FileNotFoundError, RuntimeError) as exc:
            print(exc)
            return None

    return FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def main() -> None:
    query = input("Ask a question: ").strip()
    if not query:
        print("Please enter a question.")
        return

    db = load_vector_store()
    if db is None:
        return
    docs = db.similarity_search(query, k=4)
    context = "\n".join(doc.page_content for doc in docs)

    prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question:
{query}
"""

    payload = json.dumps(
        {
            "model": LLM_MODEL_NAME,
            "prompt": prompt,
            "stream": False,
        }
    ).encode("utf-8")

    http_request = request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"Failed to contact Ollama at {OLLAMA_URL}: {exc}")
        return

    answer = data.get("response")
    if not answer:
        print("Ollama returned no response text.")
        return

    print("\nAI Answer:\n")
    print(answer)


if __name__ == "__main__":
    main()
