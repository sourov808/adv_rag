# Cross-encoder reranking. Reads (query, passage) together for a true relevance score.
# Slower than search, so we only run it on the small fused pool.

import logging
from sentence_transformers import CrossEncoder

import config

logger = logging.getLogger(__name__)


def load_reranker(model_name: str = config.RERANK_MODEL_NAME) -> CrossEncoder:
    logger.info(f'Loading reranker model: {model_name}')
    return CrossEncoder(model_name)


def rerank(model: CrossEncoder, query: str, candidates: list[dict], top_k: int) -> list[dict]:
    if not candidates:
        return []
    scores = model.predict([(query, chunk['text']) for chunk in candidates])
    ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
    results = []
    for score, chunk in ranked[:top_k]:
        hit = dict(chunk)
        hit['rerank_score'] = float(score)
        results.append(hit)
    logger.info(f'Reranked {len(candidates)} candidates down to top {len(results)}')
    return results
