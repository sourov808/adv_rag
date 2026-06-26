# Qdrant: connect, create the collection, upload chunks, and search by vector.

import os
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

import config

logger = logging.getLogger(__name__)


def connect_qdrant() -> QdrantClient:
    endpoint = os.getenv('QDRANT_ENDPOINT')
    api_key = os.getenv('QDRANT_API_KEY')
    if not endpoint or not api_key:
        raise RuntimeError('QDRANT_ENDPOINT and QDRANT_API_KEY must be set (check your .env file).')
    client = QdrantClient(url=endpoint, api_key=api_key)
    logger.info('Connected to Qdrant Cloud')
    return client


def create_collection(client: QdrantClient, vector_size: int, name: str = config.COLLECTION_NAME) -> None:
    # Recreate wipes any existing collection of this name. Fine for a demo, not production.
    client.recreate_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    logger.info(f'Collection "{name}" ready (size={vector_size}, cosine)')


def upsert_chunks(client: QdrantClient, chunks: list[dict], name: str = config.COLLECTION_NAME) -> int:
    points = []
    for idx, chunk in enumerate(chunks):
        points.append(
            PointStruct(
                id=idx,
                vector=chunk['embedding'].tolist(),  # numpy -> list for the JSON API
                payload={
                    'text': chunk['text'],
                    'source': chunk['source'],
                    'page': chunk['page'],
                    'chunk_id': chunk['chunk_id'],
                },
            )
        )
    # wait=True so the count below reflects a finished upload.
    client.upsert(collection_name=name, points=points, wait=True)
    count = client.count(collection_name=name).count
    logger.info(f'Upserted {len(points)} points (collection now holds {count})')
    return count


def dense_search(client: QdrantClient, query_vector, top_k: int, name: str = config.COLLECTION_NAME) -> list[dict]:
    response = client.query_points(
        collection_name=name,
        query=query_vector.tolist(),
        limit=top_k,
        with_payload=True,
    )
    hits = []
    for point in response.points:
        payload = point.payload
        hits.append({
            'chunk_id': payload['chunk_id'],
            'text': payload['text'],
            'source': payload['source'],
            'page': payload['page'],
            'score': point.score,
        })
    return hits
