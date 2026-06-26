# Reciprocal Rank Fusion: merge dense and keyword rankings using rank position, not score.
# rrf_score(chunk) = sum over each list of 1 / (k + rank). High in both lists -> top.

import logging

import config

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    dense_hits: list[dict],
    keyword_hits: list[dict],
    top_k: int,
    k: int = config.RRF_K,
) -> list[dict]:
    scores: dict[int, float] = {}
    records: dict[int, dict] = {}

    for ranked_list in (dense_hits, keyword_hits):
        for rank, hit in enumerate(ranked_list):
            cid = hit['chunk_id']
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            records[cid] = hit

    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]

    fused = []
    for cid in ranked_ids:
        hit = dict(records[cid])
        hit['rrf_score'] = scores[cid]
        fused.append(hit)

    logger.info(f'RRF fused {len(dense_hits)} dense + {len(keyword_hits)} keyword hits into top {len(fused)}')
    return fused
