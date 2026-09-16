"""Find document chunks that can help answer a question."""

from rag.embeddings import embed_question
from rag.vector_store import search_collection

number_of_results = 3


def retrieve_chunks(question, model, collection, limit=number_of_results):
    """Embed a question and return its closest document chunks."""
    if not question or not question.strip():
        raise ValueError("The question cannot be empty.")

    question_embedding = embed_question(model, question)
    matches = search_collection(
        collection,
        question_embedding,
        limit,
    )

    return matches


def build_context(matches):
    """Combine retrieved chunks into context for the language model."""
    if not matches:
        return ""

    context_parts = []

    for number, match in enumerate(matches, start=1):
        source = match["metadata"]["source"]
        page = match["metadata"]["page"]
        text = match["text"]

        context_parts.append(
            f"Source {number}: {source}, page {page}\n{text}"
        )

    return "\n\n".join(context_parts)


def get_source_labels(matches):
    """Return short source labels for showing citations in the interface."""
    labels = []

    for match in matches:
        source = match["metadata"]["source"]
        page = match["metadata"]["page"]
        label = f"{source}, page {page}"

        if label not in labels:
            labels.append(label)

    return labels
