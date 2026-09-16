"""Save document chunks in a local Qdrant collection and search them later."""

from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

database_path = Path(".rag_cache/qdrant")
collection_name = "documents_openai"
vector_size = 1536
vector_client = None


def open_collection():
    """Open the Qdrant collection or create it when it is missing."""
    global vector_client

    if vector_client is None:
        vector_client = QdrantClient(path=str(database_path))

    if not vector_client.collection_exists(collection_name):
        vector_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    return vector_client


def save_chunks(client, chunks, embeddings):
    """Save chunks and their OpenAI embeddings in Qdrant."""
    if len(chunks) != len(embeddings):
        raise ValueError("Every chunk needs one embedding.")
    
    if not chunks:
        return
    
    points = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        points.append(
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, chunk.id)),
                vector=embedding,
                payload={
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "source": chunk.source,
                    "page": chunk.page,
                    "position": chunk.position,
                },
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True,
    )


def search_collection(client, question_embedding, number_of_results=4):
    """Find the chunks with the highest cosine similarity to a question."""
    if number_of_results <= 0:
        raise ValueError("number_of_results must be greater than zero.")

    if collection_count(client) == 0:
        return []

    result = client.query_points(
        collection_name=collection_name,
        query=question_embedding,
        limit=number_of_results,
        with_payload=True,
    )

    matches = []
    for point in result.points:
        matches.append(
            {
                "id": point.payload["chunk_id"],
                "text": point.payload["text"],
                "metadata": {
                    "document_id": point.payload["document_id"],
                    "source": point.payload["source"],
                    "page": point.payload["page"],
                    "position": point.payload["position"],
                },
                "score": point.score,
            }
        )

    return matches


def collection_count(client):
    """Return how many chunks are stored in the Qdrant collection."""
    return client.count(collection_name=collection_name, exact=True).count
