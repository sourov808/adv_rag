# Entry point. Calls each module's functions in order, like the notebook walks through them.
# Usage:
#   uv run main.py                       interactive: ask questions in a loop
#   uv run main.py "your question here"  answer one question and exit

import os
import logging

# Quiet the chatty libraries BEFORE they are imported below, so the terminal stays readable.
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')   # no model-download bars
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
logging.basicConfig(level=logging.WARNING)
for _noisy in ('httpx', 'httpcore', 'sentence_transformers', 'transformers',
               'huggingface_hub', 'urllib3', 'pypdf', 'qdrant_client'):
    logging.getLogger(_noisy).setLevel(logging.ERROR)
try:
    from transformers.utils import logging as _hf_logging
    _hf_logging.disable_progress_bar()
    _hf_logging.set_verbosity_error()
except Exception:
    pass

import sys
from dotenv import load_dotenv

import config
from loader import load_directory
from splitter import split_records
from embedder import load_embedding_model, embed_chunks, embed_query
from vector_store import connect_qdrant, create_collection, upsert_chunks, dense_search
from keyword_index import build_bm25, bm25_search
from hybrid import reciprocal_rank_fusion
from reranker import load_reranker, rerank
from generator import connect_groq, generate_answer


def build_index():
    # One-time setup: load, split, embed, store, and load the models.
    print('Preparing... (this can take a moment on first run)')

    records = load_directory(config.DOCS_DIR, config.SUPPORTED_EXTENSIONS)
    if not records:
        raise RuntimeError(f'No documents found in {config.DOCS_DIR}. Add a PDF/TXT/MD file and retry.')
    chunks = split_records(records, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)

    embed_model = load_embedding_model()
    embed_chunks(embed_model, chunks)

    qdrant = connect_qdrant()
    create_collection(qdrant, embed_model.get_embedding_dimension())
    upsert_chunks(qdrant, chunks)

    bm25 = build_bm25(chunks)
    reranker = load_reranker()
    groq = connect_groq()

    print('Ready.')
    return embed_model, qdrant, bm25, chunks, reranker, groq


def answer_question(query, embed_model, qdrant, bm25, chunks, reranker, groq):
    # Retrieve two ways, fuse, rerank, then generate a grounded answer.
    query_vector = embed_query(embed_model, query)
    dense_hits = dense_search(qdrant, query_vector, config.RETRIEVE_POOL)
    keyword_hits = bm25_search(bm25, chunks, query, config.RETRIEVE_POOL)
    fused = reciprocal_rank_fusion(dense_hits, keyword_hits, config.RETRIEVE_POOL)
    passages = rerank(reranker, query, fused, config.FINAL_TOP_K)
    answer = generate_answer(groq, query, passages)
    return answer, passages


def print_answer(answer, passages):
    print('\n=== ANSWER ===')
    print(answer)
    print('\n=== SOURCES ===')
    for i, p in enumerate(passages, 1):
        print(f"  [{i}] page {p['page']} | chunk {p['chunk_id']} | rerank={p.get('rerank_score', 0):.3f}")


def main():
    load_dotenv()  # load keys before connecting to Qdrant and Groq
    pieces = build_index()

    if len(sys.argv) > 1:
        answer, passages = answer_question(' '.join(sys.argv[1:]), *pieces)
        print_answer(answer, passages)
        return

    print('\nAsk a question about the document (type "exit" to quit).')
    while True:
        try:
            question = input('\n> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {'exit', 'quit'}:
            break
        answer, passages = answer_question(question, *pieces)
        print_answer(answer, passages)


if __name__ == '__main__':
    main()
