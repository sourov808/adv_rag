# Turn text into vectors. Chunks and the query must use the same model.

import logging
from sentence_transformers import SentenceTransformer

import config

logger = logging.getLogger(__name__)


def load_embedding_model(model_name: str = config.EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    logger.info(f'Loading embedding model: {model_name}')
    return SentenceTransformer(model_name)


def embed_chunks(model: SentenceTransformer, chunks: list[dict]) -> list[dict]:
    # Adds an 'embedding' to each chunk in place.
    texts = [chunk['text'] for chunk in chunks]
    vectors = model.encode(texts, convert_to_numpy=True)
    for chunk, vector in zip(chunks, vectors):
        chunk['embedding'] = vector
    logger.info(f'Embedded {len(chunks)} chunks into shape {vectors.shape}')
    return chunks


def embed_query(model: SentenceTransformer, query: str):
    return model.encode(query, convert_to_numpy=True)
