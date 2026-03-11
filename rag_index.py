import contextlib
import io
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path
from langchain_core.documents import Document

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
INDEX_DIR = BASE_DIR / "vector_db"
POPPLER_BIN_DIR = Path(r"C:\Users\PC\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin")
EMBED_MODEL_NAME = "nomic-embed-text"
LLM_MODEL_NAME = "llama2-uncensored:7b"

def get_ollama_embeddings():
    try:
        from langchain_ollama import OllamaEmbeddings
    except ModuleNotFoundError:
        try:
            from langchain_community.embeddings import OllamaEmbeddings
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing embeddings dependency. Install `langchain-ollama` or "
                "`langchain-community` in the Python environment running this script."
            ) from exc

    return OllamaEmbeddings(model=EMBED_MODEL_NAME)


def list_pdf_files(docs_dir: Path = DOCS_DIR) -> list[Path]:
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    pdf_files = sorted(path for path in docs_dir.iterdir() if path.suffix.lower() == ".pdf")
    if not pdf_files:
        raise RuntimeError(f"No PDF files found in {docs_dir}")

    return pdf_files


def load_pdf_with_ocr(pdf_path: Path):
    print(f"OCR scanning {pdf_path.name}")

    try:
        images = convert_from_path(
            str(pdf_path),
            poppler_path=str(POPPLER_BIN_DIR),
            dpi=300,
        )
    except Exception as exc:
        raise RuntimeError(
            f"OCR rasterization failed for {pdf_path.name}. "
            f"Check Poppler at {POPPLER_BIN_DIR}."
        ) from exc

    documents = []

    for i, img in enumerate(images):
        try:
            text = pytesseract.image_to_string(
                img,
                lang="eng",
                config="--oem 3 --psm 6",
            )
        except pytesseract.TesseractNotFoundError as exc:
            raise RuntimeError(
                "Tesseract executable was not found. "
                "Check `pytesseract.pytesseract.tesseract_cmd`."
            ) from exc

        if text and text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": str(pdf_path), "page": i},
                )
            )

    print(f"OCR extracted {len(documents)} pages")
    return documents


def load_pdf_documents(pdf_path: Path):
    documents = []

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
            suffix = f"; skipped {skipped_pages} unreadable pages." if skipped_pages else "."
            print(f"Loaded {len(documents)} pages from {pdf_path.name} with pypdf{suffix}")
            return documents

    except ModuleNotFoundError:
        pass
    except Exception as exc:
        print(f"pypdf failed for {pdf_path.name}: {exc}")

    try:
        import fitz

        documents = []
        skipped_pages = 0
        with contextlib.redirect_stderr(io.StringIO()):
            pdf_document = fitz.open(str(pdf_path))

        with pdf_document:
            for page_number, page in enumerate(pdf_document):
                try:
                    with contextlib.redirect_stderr(io.StringIO()):
                        text = page.get_text("text") or ""
                except Exception:
                    skipped_pages += 1
                    continue

                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"source": str(pdf_path), "page": page_number},
                        )
                    )

        if documents:
            suffix = f"; skipped {skipped_pages} unreadable pages." if skipped_pages else "."
            print(f"Loaded {len(documents)} pages from {pdf_path.name} with PyMuPDF{suffix}")
            return documents

    except ModuleNotFoundError:
        pass
    except Exception as exc:
        print(f"PyMuPDF failed for {pdf_path.name}: {exc}")

    if not documents:
        print("Standard extraction failed, attempting OCR...")
        documents = load_pdf_with_ocr(pdf_path)

    if not documents:
        raise RuntimeError(f"Could not extract readable text from {pdf_path.name}")

    return documents


def build_index(docs_dir: Path = DOCS_DIR, index_dir: Path = INDEX_DIR) -> int:
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependencies. Install `langchain-community` and "
            "`langchain-text-splitters` in the Python environment running this script."
        ) from exc

    pdf_files = list_pdf_files(docs_dir)

    all_documents = []
    failed_files = []
    for pdf_path in pdf_files:
        try:
            all_documents.extend(load_pdf_documents(pdf_path))
        except RuntimeError as exc:
            failed_files.append(f"{pdf_path.name}: {exc}")

    if not all_documents:
        joined_errors = " | ".join(failed_files) if failed_files else "No readable documents were loaded."
        raise RuntimeError(f"No readable PDF content found in {docs_dir}. {joined_errors}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_documents)
    if not chunks:
        raise RuntimeError("No text chunks were created from the readable PDF content.")

    texts = []
    metadatas = []
    for chunk in chunks:
        text = chunk.page_content.strip()
        if text:
            texts.append(text)
            metadatas.append(chunk.metadata)

    if not texts:
        raise RuntimeError("All extracted chunks were empty after cleaning.")

    embeddings = get_ollama_embeddings()

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

    if failed_files:
        print("Skipped unreadable PDFs:")
        for item in failed_files:
            print(f"- {item}")

    return len(texts)


def main() -> None:
    try:
        chunk_count = build_index()
    except (FileNotFoundError, RuntimeError) as exc:
        print(exc)
        return

    print(
        f"Indexed {chunk_count} chunks from PDFs in {DOCS_DIR} into {INDEX_DIR} "
        f"using embedding model {EMBED_MODEL_NAME}."
    )


if __name__ == "__main__":
    main()
