"""Streamlit interface for the RAG application."""

import json
from pathlib import Path

import streamlit as st

from rag.embeddings import embed_chunks, get_embedding_client
from rag.generation import create_openai_model, generate_answer
from rag.ingestion import chunk_documents, load_uploads
from rag.retrieval import build_context, get_source_labels, retrieve_chunks
from rag.vector_store import collection_count, open_collection, save_chunks

results_file = Path("data/evaluation_results.json")


def main():
    """Show the document upload and question-answer interface."""
    st.set_page_config(page_title="RAG", page_icon="📚")
    st.title("RAG")
    st.write("Upload documents, index them, and ask questions about their content.")

    embedding_client = get_embedding_client()
    collection = open_collection()

    st.subheader("1. Add documents")
    uploaded_files = st.file_uploader(
        "Choose PDF, text, or Markdown files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("Index documents", type="primary"):
        index_documents(uploaded_files, embedding_client, collection)

    st.divider()
    st.subheader("2. Ask a question")

    with st.form("question_form"):
        question = st.text_input("Question")
        submitted = st.form_submit_button("Get answer")

    if submitted:
        answer_question(question, embedding_client, collection)

    st.divider()
    show_evaluation_results()


def index_documents(uploaded_files, embedding_client, collection):
    """Read uploaded files and save their chunks in ChromaDB."""
    if not uploaded_files:
        st.warning("Choose at least one document first.")
        return

    files = []
    for uploaded_file in uploaded_files:
        files.append((uploaded_file.name, uploaded_file.getvalue()))

    try:
        with st.spinner("Reading and indexing documents..."):
            documents = load_uploads(files)
            chunks = chunk_documents(documents)
            embeddings = embed_chunks(embedding_client, chunks)
            save_chunks(collection, chunks, embeddings)

        st.success(f"Indexed {len(chunks)} chunks from {len(files)} files.")
    except Exception as error:
        st.error(str(error))


def answer_question(question, embedding_client, collection):
    """Retrieve context, generate an answer, and show its sources."""
    if not question or not question.strip():
        st.warning("Enter a question first.")
        return

    if collection_count(collection) == 0:
        st.warning("Index at least one document before asking a question.")
        return

    try:
        with st.spinner("Searching the documents..."):
            matches = retrieve_chunks(question, embedding_client, collection)
            context = build_context(matches)
            llm = create_openai_model()
            answer, usage = generate_answer(
                llm,
                question,
                context,
            )

        st.subheader("Answer")
        st.write(answer)
        st.caption(
            f"Input: {usage['prompt_tokens']} tokens | "
            f"Output: {usage['completion_tokens']} tokens | "
            f"Total: {usage['total_tokens']} tokens"
        )

        source_labels = get_source_labels(matches)
        if source_labels:
            st.subheader("Sources")
            for source in source_labels:
                st.write(f"- {source}")
    except Exception as error:
        st.error(str(error))


def show_evaluation_results():
    """Display completed evaluation results without running another evaluation."""
    st.subheader("3. Evaluation results")

    if not results_file.exists():
        st.info("Run the offline evaluation to see results here.")
        return

    with results_file.open(encoding="utf-8") as file:
        report = json.load(file)

    averages = report["averages"]
    metrics = st.columns(4)
    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    for column, name in zip(metrics, metric_names, strict=True):
        score = averages[name]
        label = name.replace("_", " ").title()
        column.metric(label, f"{score:.3f}" if score is not None else "N/A")

    rows = []
    for result in report["results"]:
        rows.append(
            {
                "ID": result["id"],
                "Question": result["question"],
                "Faithfulness": result["scores"]["faithfulness"],
                "Answer relevancy": result["scores"]["answer_relevancy"],
                "Context precision": result["scores"]["context_precision"],
                "Context recall": result["scores"]["context_recall"],
            }
        )

    st.dataframe(rows, hide_index=True, use_container_width=True)
    st.caption("Context precision and recall are not scored for unanswerable questions.")


if __name__ == "__main__":
    main()
