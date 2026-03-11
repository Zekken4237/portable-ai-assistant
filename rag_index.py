import contextlib
import io
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
INDEX_DIR = BASE_DIR / "vector_db"
SOURCE_PDF = DOCS_DIR / "the-art-of-seduction-robert-greene.pdf"
EMBED_MODEL_NAME = "nomic-embed-text"
LLM_MODEL_NAME = "llama2-uncensored:7b"


def load_pdf_documents(pdf_path: Path):
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_core.documents import Document
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependencies. Install langchain-community and "
            "langchain-text-splitters in the Python environment running this script."
        ) from exc

    extraction_errors = []

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            documents = PyPDFLoader(str(pdf_path)).load()
        documents = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
        if documents:
            print(f"Loaded {len(documents)} pages with PyPDFLoader.")
            return documents
        extraction_errors.append("PyPDFLoader returned no readable text.")
    except Exception as exc:
        extraction_errors.append(f"PyPDFLoader failed: {exc}")

    try:
        from pypdf import PdfReader

        with contextlib.redirect_stderr(io.StringIO()):
            reader = PdfReader(str(pdf_path), strict=False)

        documents = []
        skipped_pages = 0
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    text = page.extract_text() or ""
            except Exception:
                skipped_pages += 1
                continue

            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": str(pdf_path), "page": page_number - 1},
                    )
                )

        if documents:
            print(
                f"Loaded {len(documents)} pages with pypdf"
                + (f"; skipped {skipped_pages} unreadable pages." if skipped_pages else ".")
            )
            return documents

        extraction_errors.append(
            "pypdf returned no readable text."
            + (f" Skipped {skipped_pages} unreadable pages." if skipped_pages else "")
        )
    except ModuleNotFoundError:
        extraction_errors.append("pypdf is not installed.")
    except Exception as exc:
        extraction_errors.append(f"pypdf failed: {exc}")

    try:
        import fitz

        documents = []
        with fitz.open(str(pdf_path)) as pdf_document:
            for page_number, page in enumerate(pdf_document):
                text = page.get_text("text") or ""
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"source": str(pdf_path), "page": page_number},
                        )
                    )

        if documents:
            print(f"Loaded {len(documents)} pages with PyMuPDF.")
            return documents

        extraction_errors.append("PyMuPDF returned no readable text.")
    except ModuleNotFoundError:
        extraction_errors.append("PyMuPDF is not installed.")
    except Exception as exc:
        extraction_errors.append(f"PyMuPDF failed: {exc}")

    details = " ".join(extraction_errors)
    raise RuntimeError(
        f"Could not extract readable text from {pdf_path.name}. "
        "The PDF may be malformed, image-only, or compressed in a way the available readers cannot decode. "
        f"Details: {details}"
    )


def build_index(pdf_path: Path = SOURCE_PDF, index_dir: Path = INDEX_DIR) -> int:
    try:
        from langchain_community.embeddings import OllamaEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependencies. Install langchain-community and "
            "langchain-text-splitters in the Python environment running this script."
        ) from exc

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    documents = load_pdf_documents(pdf_path)
    if not documents:
        raise RuntimeError(f"No pages were loaded from {pdf_path.name}.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise RuntimeError(f"No text chunks were created from {pdf_path.name}.")

    texts = []
    metadatas = []
    for chunk in chunks:
        text = chunk.page_content.strip()
        if text:
            texts.append(text)
            metadatas.append(chunk.metadata)

    if not texts:
        raise RuntimeError(
            f"All extracted chunks from {pdf_path.name} were empty. "
            "The PDF may be image-only or unreadable to PyPDFLoader."
        )

    embeddings = OllamaEmbeddings(model=EMBED_MODEL_NAME)

    try:
        vectors = embeddings.embed_documents(texts)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create embeddings with Ollama model '{EMBED_MODEL_NAME}'. "
            "Make sure Ollama is running and the model is installed with "
            f"`ollama pull {EMBED_MODEL_NAME}`."
        ) from exc

    if not vectors or not vectors[0]:
        raise RuntimeError(
            f"Ollama model '{EMBED_MODEL_NAME}' returned no embeddings. "
            "Use an embedding model and verify it is installed."
        )

    db = FAISS.from_embeddings(list(zip(texts, vectors)), embeddings, metadatas=metadatas)

    index_dir.mkdir(parents=True, exist_ok=True)
    db.save_local(str(index_dir))

    return len(texts)


def main() -> None:
    try:
        chunk_count = build_index()
    except (FileNotFoundError, RuntimeError) as exc:
        print(exc)
        return

    print(
        f"Indexed {chunk_count} chunks from {SOURCE_PDF.name} into {INDEX_DIR} "
        f"using embedding model {EMBED_MODEL_NAME}."
    )


if __name__ == "__main__":
    main()
