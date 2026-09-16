"""Create OpenAI embeddings for document chunks and questions."""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

embedding_model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
embedding_client = None


def get_embedding_client():
    """Return the OpenAI client used for embeddings."""
    global embedding_client

    if embedding_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Add OPENAI_API_KEY to your .env file.")

        embedding_client = OpenAI(api_key=api_key)

    return embedding_client


def embed_chunks(client, chunks):
    """Create one embedding for every chunk."""
    return embed_texts(client, [chunk.text for chunk in chunks])


def embed_question(client, question):
    """Create an embedding for a user's question."""
    if not question or not question.strip():
        raise ValueError("The question cannot be empty.")

    return embed_texts(client, [question])[0]


def embed_texts(client, texts):
    """Create embeddings and return them as regular lists."""
    if not texts:
        return []

    response = client.embeddings.create(
        model=embedding_model_name,
        input=texts,
    )
    return [item.embedding for item in response.data]
