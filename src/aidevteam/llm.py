"""LLM factory shared by all agents.

Single vendor-neutral code path: any OpenAI-compatible endpoint works.
Point LLM_BASE_URL / LLM_MODEL (and LLM_API_KEY when the endpoint requires
one) at your provider — there are no per-vendor branches or defaults.

Every LLM call is wrapped by a transparent :class:`MonitoredLLM` proxy that
counts input/output tokens (via tiktoken) and logs context usage per agent
node. The wrapper is invisible to callers: ``get_structured_llm`` returns the
same shaped object it always did, so no agent code needs to change.
"""

import inspect
import json
from dataclasses import dataclass
from typing import Any

import tiktoken
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from rich.console import Console

from aidevteam.config import get_settings

console = Console()

# Agent node roles, used to attribute context usage to a node when the caller
# does not pass one explicitly (inferred from the call stack).
AGENT_NODES = ("planner", "coder", "tester", "reviewer")

# Default context window for models tiktoken cannot map to a tokenizer.
# Operators should set LLM_CONTEXT_LIMIT for their actual model.
DEFAULT_CONTEXT_LIMIT: int | None = None


@dataclass
class ContextUsage:
    """Result of a single monitored LLM call."""

    node: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    context_limit: int | None
    utilisation_pct: float | None
    exceeded: bool
    model: str | None


class ContextMonitor:
    """Tracks per-node LLM context usage and warns on high utilisation."""

    def __init__(self, threshold_pct: float = 85.0) -> None:
        self.threshold_pct = threshold_pct
        self._totals: dict[str, dict[str, Any]] = {}

    def record(
        self,
        node: str,
        input_tokens: int,
        output_tokens: int,
        model: str | None = None,
        context_limit: int | None = None,
    ) -> ContextUsage:
        total = input_tokens + output_tokens
        utilisation_pct: float | None = None
        exceeded = False
        if context_limit:
            utilisation_pct = total / context_limit * 100
            exceeded = total > context_limit

        bucket = self._totals.setdefault(
            node, {"input": 0, "output": 0, "total": 0, "calls": 0}
        )
        bucket["input"] += input_tokens
        bucket["output"] += output_tokens
        bucket["total"] += total
        bucket["calls"] += 1

        usage = ContextUsage(
            node=node,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total,
            context_limit=context_limit,
            utilisation_pct=utilisation_pct,
            exceeded=exceeded,
            model=model,
        )
        self._log(usage)
        return usage

    @staticmethod
    def _rich_escape(value: str) -> str:
        """Escape rich markup brackets in dynamic content."""
        return str(value).replace("[", "[[").replace("]", "]]")

    def _log(self, usage: ContextUsage) -> None:
        node = self._rich_escape(usage.node)
        extra = ""
        if usage.utilisation_pct is not None:
            extra = f" ({usage.utilisation_pct:.0f}% of {usage.context_limit})"
        console.print(
            f"  [dim]context[[{node}]]: {usage.input_tokens} in + "
            f"{usage.output_tokens} out = {usage.total_tokens} tokens{extra}[/dim]"
        )
        if usage.exceeded:
            console.print(
                f"  [yellow]⚠ context[[{node}]] exceeds limit "
                f"{usage.context_limit} ({usage.total_tokens} tokens)[/yellow]"
            )
        elif (
            usage.utilisation_pct is not None
            and usage.utilisation_pct >= self.threshold_pct
        ):
            console.print(
                f"  [yellow]⚠ context[[{node}]] at "
                f"{usage.utilisation_pct:.0f}% of {usage.context_limit} token window[/yellow]"
            )

    def totals(self) -> dict[str, dict[str, Any]]:
        """Return a copy of cumulative per-node usage."""
        return {k: dict(v) for k, v in self._totals.items()}


# Module-level monitor shared by every LLM call.
context_monitor = ContextMonitor()


def _flatten_text(value: Any) -> str:
    """Flatten an LLM input/output value into a single string for counting."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, BaseMessage):
        return _flatten_text(value.content)
    if isinstance(value, list):
        return "\n".join(_flatten_text(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, default=str)
    # Pydantic models (structured output) and anything else.
    if hasattr(value, "model_dump"):
        return json.dumps(value.model_dump(), default=str)
    return str(value)


def _infer_node_from_stack() -> str:
    """Infer the agent node name from the call stack, if possible."""
    for frame in inspect.stack():
        mod = frame.frame.f_globals.get("__name__", "")
        if mod.startswith("aidevteam.agents."):
            name = mod.split(".")[-1]
            if name in AGENT_NODES:
                return name
        if frame.function in AGENT_NODES:
            return frame.function
    return "unknown"


class MonitoredLLM:
    """Transparent proxy around a LangChain runnable that monitors context.

    ``invoke``/``ainvoke`` count tokens before delegating to the wrapped
    runnable and report usage via the module-level :data:`context_monitor`.
    Every other attribute is delegated to the wrapped object, so callers see
    the same interface they always did.
    """

    def __init__(
        self,
        wrapped: Any,
        node: str | None = None,
        context_limit: int | None = None,
    ) -> None:
        self.wrapped = wrapped
        self._node = node
        self._context_limit = context_limit

    @staticmethod
    def count_tokens(text: str, model: str | None = None) -> int:
        """Count tokens in ``text`` for ``model``.

        Falls back to the ``cl100k_base`` tokenizer when the model has no
        dedicated tiktoken encoding (common for OpenAI-compatible endpoints
        such as NIM / Ollama / vLLM).
        """
        if not text:
            return 0
        try:
            enc = tiktoken.encoding_for_model(model) if model else None
        except KeyError:
            enc = None
        if enc is None:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

    @property
    def model(self) -> str | None:
        return getattr(self.wrapped, "model", None) or getattr(
            self.wrapped, "model_name", None
        )

    def _node_name(self) -> str:
        return self._node or _infer_node_from_stack()

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        node = self._node_name()
        model = self.model
        input_text = _flatten_text(input)
        input_tokens = self.count_tokens(input_text, model)

        if config is not None:
            output = self.wrapped.invoke(input, config=config, **kwargs)
        else:
            output = self.wrapped.invoke(input, **kwargs)

        output_text = _flatten_text(output)
        output_tokens = self.count_tokens(output_text, model)
        context_monitor.record(
            node,
            input_tokens,
            output_tokens,
            model=model,
            context_limit=self._context_limit,
        )
        return output

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        node = self._node_name()
        model = self.model
        input_text = _flatten_text(input)
        input_tokens = self.count_tokens(input_text, model)

        if config is not None:
            output = await self.wrapped.ainvoke(input, config=config, **kwargs)
        else:
            output = await self.wrapped.ainvoke(input, **kwargs)

        output_text = _flatten_text(output)
        output_tokens = self.count_tokens(output_text, model)
        context_monitor.record(
            node,
            input_tokens,
            output_tokens,
            model=model,
            context_limit=self._context_limit,
        )
        return output

    def __getattr__(self, name: str) -> Any:
        wrapped = self.__dict__.get("wrapped")
        if wrapped is None:
            raise AttributeError(name)
        return getattr(wrapped, name)


def _build_chat(model: str, **kwargs: Any) -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=model,
        api_key=settings.llm_api_key or "ollama",  # placeholder; unauthenticated endpoints ignore it
        base_url=settings.resolved_base_url,
        streaming=True,
        timeout=settings.llm_timeout,
        **kwargs,
    )


def get_structured_llm(schema: type, **kwargs) -> MonitoredLLM:
    """Return a context-monitored chat model bound to a Pydantic schema.

    Uses ``method="json_schema"`` so the endpoint returns raw JSON conforming
    to the Pydantic schema, rather than a tool-call ``arguments`` string.
    """
    settings = get_settings()
    model = kwargs.pop("model", None) or settings.resolved_model
    kwargs.setdefault("temperature", settings.temperature)
    real = _build_chat(model, **kwargs)
    structured = real.with_structured_output(schema, method="json_schema")
    return MonitoredLLM(structured, context_limit=settings.llm_context_limit)
