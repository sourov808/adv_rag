# Load PDF/TXT/MD files into page records: {'text', 'source', 'page'}.

import logging
from pathlib import Path
from pypdf import PdfReader

logger = logging.getLogger(__name__)


def load_pdf(path: Path) -> list[dict]:
    logger.info(f'Loading PDF: {path.name}')
    try:
        reader = PdfReader(path)
    except Exception as e:
        logger.error(f'Failed to open PDF {path}: {e}')
        return []

    records = []
    for page_num, page in enumerate(reader.pages):
        try:
            text = (page.extract_text() or '').strip()
            if not text:  # skip blank pages
                continue
            records.append({'text': text, 'source': path.name, 'page': page_num})
        except Exception as e:
            logger.warning(f'Failed to extract page {page_num} of {path.name}: {e}')
            continue

    logger.info(f'Successfully loaded {len(records)} pages from {path.name}')
    return records


def load_text_file(path: Path) -> list[dict]:
    # Used for both .txt and .md, which we treat as one block of text.
    logger.info(f'Loading text file: {path.name}')
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read().strip()
    except Exception as e:
        logger.error(f'Failed to read {path}: {e}')
        return []
    if not text:
        return []
    return [{'text': text, 'source': path.name, 'page': 0}]


def load_document(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == '.pdf':
        return load_pdf(path)
    if suffix in {'.txt', '.md'}:
        return load_text_file(path)
    logger.warning(f'Unsupported file type: {suffix} for {path.name}')
    return []


def load_directory(directory: Path, supported_extensions: set[str]) -> list[dict]:
    if not directory.exists():
        logger.error(f'Directory does not exist: {directory.resolve()}')
        return []

    all_records = []
    logger.info(f'Scanning directory: {directory.resolve()}')
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in supported_extensions:
            all_records.extend(load_document(path))
    logger.info(f'Total records ingested: {len(all_records)}')
    return all_records
