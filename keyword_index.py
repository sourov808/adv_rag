# BM25 keyword search. Catches exact tokens (versions, names) that vector search misses.

import logging
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_bm25(chunks: list[dict]) -> BM25Okapi:
    bm25 = BM25Okapi([tokenize(chunk['text']) for chunk in chunks])
    logger.info(f'BM25 index built over {len(chunks)} chunks')
    return bm25


def bm25_search(bm25: BM25Okapi, chunks: list[dict], query: str, top_k: int) -> list[dict]:
    scores = bm25.get_scores(tokenize(query))
    # BM25 scores are unbounded, so we rank by them rather than compare to cosine scores.
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    hits = []
    for i in ranked_idx:
        chunk = chunks[i]
        hits.append({
            'chunk_id': chunk['chunk_id'],
            'text': chunk['text'],
            'source': chunk['source'],
            'page': chunk['page'],
            'score': float(scores[i]),
        })
    return hits
