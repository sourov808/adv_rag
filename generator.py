# The "G" in RAG: ask a Groq LLM to answer using only the retrieved passages, with citations.

import os
import logging

from groq import Groq

import config

logger = logging.getLogger(__name__)

# Grounding (answer only from context) + citations are what make the answer trustworthy.
SYSTEM_PROMPT = (
    'You are a helpful assistant named RAG BOT you can answer questions directly from the document. '
    'Answer ONLY using the provided context. '
    "If the answer is not in the context, say 'I don't know based on the document.' "
)


def connect_groq() -> Groq:
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise RuntimeError('GROQ_API_KEY must be set (check your .env file).')
    logger.info('Groq client ready')
    return Groq(api_key=api_key)


def build_context(passages: list[dict]) -> str:
    # Number each passage and tag its page so the model can cite it.
    return '\n\n'.join(
        f"[Source {i} | page {p['page']}]\n{p['text']}" for i, p in enumerate(passages, 1)
    )


def generate_answer(client: Groq, query: str, passages: list[dict], model_name: str = config.GROQ_MODEL_NAME) -> str:
    user_prompt = f'Context:\n{build_context(passages)}\n\nQuestion: {query}'
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_prompt},
        ],
        temperature=0.2,  # low temperature keeps the answer factual
    )
    return response.choices[0].message.content
