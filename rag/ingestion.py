"""Read uploaded files and break their text into smaller pieces."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints
from pypdf import PdfReader


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Document(BaseModel):
    """Text extracted from one page or one plain-text file."""

    text: NonEmptyText
    source: NonEmptyText
    page: int = Field(ge=1)


class Chunk(BaseModel):
    """A retrievable piece of a document with citation metadata."""

    id: NonEmptyText
    document_id: NonEmptyText
    text: NonEmptyText
    source: NonEmptyText
    page: int = Field(ge=1)
    position: int = Field(ge=0)

supported_extensions = {".pdf", ".txt", ".md"}
chunk_size = 220
chunk_overlap = 40


def load_upload(name: str, content: bytes) -> list[Document]:
    """Read one uploaded file and return its text page by page."""
    source = Path(name).name
    extension = Path(source).suffix.lower()

    if not source or not content:
        raise ValueError("The uploaded file is empty.")
    if extension not in supported_extensions:
        supported = ", ".join(sorted(supported_extensions))
        raise ValueError(f"Unsupported file type '{extension or 'unknown'}'. Use {supported}.")

    if extension == ".pdf":
        documents = _load_pdf(source, content)
    else:
        documents = _load_text_file(source, content)

    if not documents:
        raise ValueError(f"No readable text was found in {source}.")

    return documents


def load_uploads(files: list[tuple[str, bytes]]) -> list[Document]:
    """Read several uploads and put all their pages in one list."""
    if not files:
        raise ValueError("Choose at least one document to index.")

    documents = []
    for name, content in files:
        documents.extend(load_upload(name, content))

    return documents


def _load_text_file(source, content):
    """Read a plain-text or Markdown file."""
    try:
        text = content.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise ValueError(f"{source} is not valid UTF-8 text.") from error

    if not text:
        return []

    return [Document(text=text, source=source, page=1)]


def _load_pdf(source, content):
    """Read each non-empty page of a PDF."""
    try:
        reader = PdfReader(BytesIO(content))
    except Exception as error:
        raise ValueError(f"Could not read PDF file {source}.") from error

    documents = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception as error:
            raise ValueError(f"Could not extract page {page_number} from {source}.") from error

        if text:
            documents.append(Document(text=text, source=source, page=page_number))

    return documents


def chunk_documents(documents, size=chunk_size, overlap=chunk_overlap):
    """Split documents into word groups, keeping some overlap between them."""
    if size <= 0:
        raise ValueError("size must be greater than zero.")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be at least zero and smaller than size.")

    document_ids = _create_document_ids(documents)
    chunks = []
    step = size - overlap

    for document in documents:
        words = document.text.split()
        document_id = document_ids[document.source]
        start = 0
        position = 0

        while start < len(words):
            text = " ".join(words[start : start + size])
            chunk_id = _make_id(document_id, document.page, position, text)

            chunk = Chunk(
                id=chunk_id,
                document_id=document_id,
                text=text,
                source=document.source,
                page=document.page,
                position=position,
            )
            chunks.append(chunk)

            if start + size >= len(words):
                break

            start += step
            position += 1

    return chunks


def _create_document_ids(documents):
    """Give every uploaded document a repeatable ID based on its contents."""
    pages_by_source = {}

    for document in documents:
        if document.source not in pages_by_source:
            pages_by_source[document.source] = []

        page_text = f"{document.page}:{document.text}"
        pages_by_source[document.source].append(page_text)

    document_ids = {}
    for source, pages in pages_by_source.items():
        document_ids[source] = _make_id(source, "|".join(pages))

    return document_ids


def _make_id(*parts):
    """Create a short, repeatable ID from one or more values."""
    text = ":".join(str(part) for part in parts)
    return sha256(text.encode("utf-8")).hexdigest()[:16]
