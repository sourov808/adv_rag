# Advanced RAG

A retrieval-augmented generation pipeline that answers questions about your own documents
and cites the pages it used. It combines semantic (vector) search with keyword (BM25)
search, reranks the results with a cross-encoder, and generates a grounded answer with an
LLM. If the answer is not in the document, it says so instead of making something up.

I built this to understand how a real RAG system fits together, beyond the basic
"embed and search" version. Each stage is its own small module of plain functions, and the
notebook walks through them one at a time.

## How it works

```
document  ->  load  ->  split  ->  embed  ->  store in Qdrant
                                                    |
question  ->  embed  ->  +-- dense search (meaning) --+
                         |                            |--> RRF fuse --> rerank --> LLM --> answer
                         +-- BM25 search (keywords) --+
```

1. **Load** (`loader.py`) reads PDF/TXT/MD files into page records.
2. **Split** (`splitter.py`) cuts pages into overlapping chunks.
3. **Embed** (`embedder.py`) turns chunks into 384-dim vectors with `all-MiniLM-L6-v2`.
4. **Store** (`vector_store.py`) uploads vectors to Qdrant Cloud.
5. **Dense search** finds chunks by meaning; **BM25** (`keyword_index.py`) finds them by
   exact words. The two cover each other's blind spots.
6. **Hybrid** (`hybrid.py`) fuses both rankings with Reciprocal Rank Fusion.
7. **Rerank** (`reranker.py`) re-scores the candidates with a cross-encoder so the most
   relevant passage ends up first.
8. **Generate** (`generator.py`) asks a Groq-hosted LLM to answer using only those
   passages, with page citations.

`main.py` calls these functions in order, the same sequence the notebook walks through.

## Stack

- Python 3.14, managed with [uv](https://docs.astral.sh/uv/)
- [sentence-transformers](https://www.sbert.net/) for embeddings and reranking
- [Qdrant Cloud](https://qdrant.tech/) as the vector database
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) for keyword search
- [Groq](https://groq.com/) for fast LLM inference (`llama-3.3-70b-versatile`)
- LangChain text splitters, pypdf

## Setup

You need a (free) Qdrant Cloud account and a (free) Groq API key.

1. Install dependencies:

   ```bash
   uv sync
   ```

2. Create a `.env` file in the project root (it is gitignored, never commit it):

   ```
   QDRANT_ENDPOINT=https://your-cluster-url.qdrant.io:6333
   QDRANT_API_KEY=your-qdrant-key
   GROQ_API_KEY=your-groq-key
   ```

3. Put a document to query inside the `document/` folder. Any PDF, `.txt`, or `.md` works.

## Usage

Ask a single question:

```bash
uv run main.py "How do I set up an Express server with npm?"
```

Or run interactively and ask several:

```bash
uv run main.py
```

Each answer prints the passages it was grounded on, with page and chunk numbers, so you can
check it against the source.

## The notebook

`data.ipynb` is a guided walkthrough of the same pipeline, stage by stage, with the output
of each step shown. It imports the real module functions rather than re-implementing
anything, so it stays in sync with the code. Open it if you want to understand how the
pieces behave individually before they are wired together in `main.py`.

## Project layout

```
config.py          all settings (paths, model names, search parameters)
loader.py          documents -> page records
splitter.py        pages -> chunks
embedder.py        chunks/queries -> vectors
vector_store.py    Qdrant: connect, collection, upload, dense search
keyword_index.py   BM25 keyword index
hybrid.py          Reciprocal Rank Fusion
reranker.py        cross-encoder reranking
generator.py       grounded answer generation (Groq)
main.py            entry point, calls the above in order
data.ipynb         stage-by-stage walkthrough
```

## Notes and limitations

- `create_collection` recreates the collection on every run, which is fine for a demo but
  would overwrite data in production. A real ingest would check for an existing collection
  first.
- The BM25 index is held in memory and rebuilt on each run, so it does not persist between
  processes. For a larger corpus this would move into Qdrant's own sparse vectors.
- Chunking is fixed-size character splitting. Smarter, structure-aware chunking would
  improve retrieval on documents with headings, tables, or code.
