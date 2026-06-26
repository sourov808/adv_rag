# Split page records into smaller overlapping chunks, keeping their metadata.

import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def split_records(records: list[dict], chunk_size: int = 600, chunk_overlap: int = 100) -> list[dict]:
    logger.info(f'Splitting {len(records)} records (chunk_size={chunk_size}, overlap={chunk_overlap})')

    # Splits on paragraphs, then sentences, then words, to keep meaning together.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=['\n\n', '\n', ' ', ''],
    )

    chunked_records = []
    chunk_counter = 0
    for record in records:
        for chunk in splitter.split_text(record['text']):
            chunk = chunk.strip()
            if not chunk:
                continue
            chunked_records.append({
                'text': chunk,
                'source': record['source'],
                'page': record['page'],
                'chunk_id': chunk_counter,
            })
            chunk_counter += 1

    logger.info(f'Generated {len(chunked_records)} chunks from {len(records)} records')
    return chunked_records
