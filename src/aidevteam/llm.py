"""LLM factory shared by all agents.

Single vendor-neutral code path: any OpenAI-compatible endpoint works.
Point LLM_BASE_URL / LLM_MODEL (and LLM_API_KEY when the endpoint requires
one) at your provider — there are no per-vendor branches or defaults.
"""

from langchain_openai import ChatOpenAI

from aidevteam.config import get_settings


def get_llm(**kwargs) -> ChatOpenAI:
    settings = get_settings()
    model = kwargs.pop("model", None) or settings.resolved_model
    kwargs.setdefault("temperature", settings.temperature)

    return ChatOpenAI(
        model=model,
        api_key=settings.llm_api_key or "ollama",  # placeholder; unauthenticated endpoints ignore it
        base_url=settings.resolved_base_url,
        streaming=True,
        **kwargs,
    )


def get_structured_llm(schema: type, **kwargs):
    """Return a chat model bound to a Pydantic schema via tool calling.

    Uses ``method="function_calling"`` because some OpenAI-compatible
    endpoints (e.g. NIM) do not reliably support the ``json_schema``/parse
    structured-output mode.
    """
    return get_llm(**kwargs).with_structured_output(schema, method="function_calling")
