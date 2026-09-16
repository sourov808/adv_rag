# RAG

A document question-answering application built with OpenAI, Qdrant, and Ragas.
Upload a PDF, text, or Markdown document, then ask grounded questions about its content.

## Features

- Upload PDF, TXT, and Markdown files.
- Split documents into overlapping chunks.
- Create embeddings with OpenAI `text-embedding-3-small`.
- Store and retrieve vectors with local Qdrant.
- Generate grounded answers with OpenAI Agents SDK and `gpt-4o-mini`.
- Show source pages and token usage in Streamlit.
- Evaluate retrieval and answer quality with Ragas.

## How it works

```text
Document upload
  → text extraction
  → chunking
  → OpenAI embeddings
  → Qdrant vector storage

Question
  → OpenAI embedding
  → Qdrant retrieval
  → OpenAI Agents SDK
  → grounded answer with sources
```

## Tech stack

- Python
- Streamlit
- OpenAI API
- OpenAI Agents SDK
- `gpt-4o-mini`
- `text-embedding-3-small`
- Qdrant
- Ragas
- PyPDF
- Pydantic

## Evaluation

The project was evaluated with 15 questions from *The Wonderful Wizard of Oz*, including
two unanswerable questions to check hallucination behavior.

| Metric | Score | What it measures |
| --- | ---: | --- |
| Faithfulness | 0.878 | Whether the answer is supported by retrieved context. |
| Answer relevancy | 0.643 | Whether the answer directly addresses the question. |
| Context precision | 0.808 | Whether relevant chunks are ranked near the top. |
| Context recall | 0.769 | Whether retrieval finds enough information for the answer. |

These results show strong groundedness and retrieval precision. The next focus is improving
context recall and making answers more direct.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EVALUATION_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

## Run the app

```bash
uv run streamlit run main.py
```

Upload `data/wizard_of_oz.txt`, click **Index documents**, then ask a question.

Example questions:

```text
Who did Dorothy live with in Kansas?

How did Dorothy defeat the Wicked Witch of the West?

What did Dorothy, the Scarecrow, the Tin Woodman, and the Lion each want from Oz?
```

## Run evaluation

After indexing the document, stop Streamlit and run:

```bash
uv run python -m rag.evaluation
```

The evaluation saves:

- `data/evaluation_results.json` — full per-question scores and answers
- `data/evaluation_report.md` — shareable Markdown summary

## Project structure

```text
main.py                Streamlit application
rag/embeddings.py      OpenAI embedding functions
rag/ingestion.py       File reading and document chunking
rag/vector_store.py    Local Qdrant storage and search
rag/retrieval.py       Question retrieval and context building
rag/generation.py      OpenAI Agents SDK answer generation
rag/evaluation.py      Ragas evaluation runner
data/                  Evaluation data and results
assets/                README screenshot and demo video
```
