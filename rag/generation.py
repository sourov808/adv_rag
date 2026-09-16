"""Generate answers with the OpenAI Agents SDK and OpenAI models."""

import os

from agents import (
    Agent,
    ModelSettings,
    OpenAIChatCompletionsModel,
    Runner,
    set_tracing_disabled,
)
from dotenv import load_dotenv
from openai import AsyncOpenAI

set_tracing_disabled(True)
load_dotenv()

llm_model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
maximum_answer_tokens = 300


def get_openai_api_key():
    """Read the OpenAI API key used for generation and evaluation."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Add OPENAI_API_KEY to your .env file.")

    return api_key


def create_openai_model(api_key=None, model_name=llm_model_name):
    """Connect the Agents SDK to an OpenAI chat-completions model."""
    if api_key is None:
        api_key = get_openai_api_key()

    client = AsyncOpenAI(api_key=api_key)

    return OpenAIChatCompletionsModel(
        model=model_name,
        openai_client=client,
    )


def generate_answer(model, question, context):
    """Run one grounded question-answering agent."""
    agent, user_message = prepare_answer(model, question, context)
    if agent is None:
        return "I could not find relevant information in the uploaded documents.", empty_usage()

    return format_answer(Runner.run_sync(agent, user_message))


async def generate_answer_async(model, question, context):
    """Run one grounded question-answering agent inside an async program."""
    agent, user_message = prepare_answer(model, question, context)
    if agent is None:
        return "I could not find relevant information in the uploaded documents.", empty_usage()

    return format_answer(await Runner.run(agent, user_message))


def prepare_answer(model, question, context):
    """Create the agent and message used by both answer functions."""
    if not question or not question.strip():
        raise ValueError("The question cannot be empty.")

    if not context:
        return None, None

    agent = Agent(
        name="Document assistant",
        instructions=(
            "Answer using only the provided document context. "
            "If the context does not contain the answer, say that you do not know. "
            "Keep the answer clear and include source numbers when possible."
        ),
        model=model,
        model_settings=ModelSettings(
            temperature=0.1,
            max_tokens=maximum_answer_tokens,
            include_usage=True,
        ),
    )

    user_message = f"""Document context:
{context}

Question:
{question}

Answer:"""

    return agent, user_message


def format_answer(result):
    """Return an agent result as answer text and token usage."""
    answer = result.final_output

    if not answer:
        answer = "The language model did not return an answer."

    run_usage = result.context_wrapper.usage
    usage = {
        "prompt_tokens": run_usage.input_tokens,
        "completion_tokens": run_usage.output_tokens,
        "total_tokens": run_usage.total_tokens,
    }

    return answer.strip(), usage


def empty_usage():
    """Return zero token counts when the language model is not called."""
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
