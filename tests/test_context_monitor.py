"""Unit tests for the LLM context monitor in aidevteam.llm."""

import pytest
from pydantic import BaseModel

from aidevteam.llm import (
    AGENT_NODES,
    ContextMonitor,
    MonitoredLLM,
    context_monitor,
    get_structured_llm,
)


@pytest.fixture(autouse=True)
def _reset_module_monitor():
    """Isolate tests from the module-level monitor singleton."""
    context_monitor._totals.clear()
    yield
    context_monitor._totals.clear()


class _Schema(BaseModel):
    x: int


class _FakeRunnable:
    """A minimal stand-in for a LangChain runnable."""

    def __init__(self, output):
        self.output = output
        self.calls = []

    def invoke(self, input, config=None, **kwargs):
        self.calls.append((input, config, kwargs))
        return self.output

    async def ainvoke(self, input, config=None, **kwargs):
        self.calls.append((input, config, kwargs))
        return self.output


def test_count_tokens_known_string():
    assert MonitoredLLM.count_tokens("hello world") == 2  # cl100k_base


def test_count_tokens_empty_is_zero():
    assert MonitoredLLM.count_tokens("") == 0
    assert MonitoredLLM.count_tokens(None) == 0  # type: ignore[arg-type]


def test_count_tokens_unknown_model_falls_back():
    # glm-5.3-flash has no dedicated tiktoken encoding -> cl100k_base.
    assert MonitoredLLM.count_tokens("hello world", model="glm-5.3-flash") == 2


def test_record_computes_totals_and_flags():
    mon = ContextMonitor()
    usage = mon.record("planner", 100, 50, model="m", context_limit=200)
    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    assert usage.total_tokens == 150
    assert usage.utilisation_pct == 75.0
    assert usage.exceeded is False

    # Exceeds the limit.
    usage = mon.record("coder", 150, 100, model="m", context_limit=200)
    assert usage.exceeded is True
    assert usage.utilisation_pct == 125.0

    # No limit -> raw counts only, no percentage.
    usage = mon.record("tester", 10, 10, model="m")
    assert usage.utilisation_pct is None
    assert usage.exceeded is False


def test_record_threshold_warning_is_not_an_error():
    mon = ContextMonitor(threshold_pct=85.0)
    # 84% -> no warning flag; 85% -> at threshold.
    usage = mon.record("reviewer", 84, 0, model="m", context_limit=100)
    assert usage.exceeded is False
    usage = mon.record("reviewer", 85, 0, model="m", context_limit=100)
    assert usage.exceeded is False  # at limit, not over
    assert usage.utilisation_pct == 85.0


def test_cumulative_totals():
    mon = ContextMonitor()
    mon.record("planner", 10, 10, model="m")
    mon.record("planner", 20, 20, model="m")
    mon.record("coder", 5, 5, model="m")
    totals = mon.totals()
    assert totals["planner"] == {"input": 30, "output": 30, "total": 60, "calls": 2}
    assert totals["coder"] == {"input": 5, "output": 5, "total": 10, "calls": 1}


def test_monitored_llm_delegates_and_returns_unchanged():
    wrapped = _FakeRunnable("structured output")
    ml = MonitoredLLM(wrapped, node="planner", context_limit=1000)
    result = ml.invoke(["hello world"])
    assert result == "structured output"
    assert len(wrapped.calls) == 1
    assert wrapped.calls[0][0] == ["hello world"]
    assert wrapped.calls[0][1] is None

    totals = context_monitor.totals()
    assert totals["planner"]["calls"] == 1
    assert totals["planner"]["input"] == 2  # "hello world"
    assert totals["planner"]["output"] > 0


def test_monitored_llm_passes_config_and_kwargs():
    wrapped = _FakeRunnable("out")
    ml = MonitoredLLM(wrapped, node="coder")
    ml.invoke(["x"], config={"tags": ["a"]}, extra=1)
    assert wrapped.calls[0][1] == {"tags": ["a"]}
    assert wrapped.calls[0][2] == {"extra": 1}


def test_monitored_llm_getattr_delegation():
    wrapped = _FakeRunnable("out")
    wrapped.custom_attr = 42
    ml = MonitoredLLM(wrapped, node="tester")
    assert ml.custom_attr == 42


def test_monitored_llm_infers_node_from_stack():
    # A function named after an agent on the stack should be attributed.
    def planner():
        wrapped = _FakeRunnable("out")
        ml = MonitoredLLM(wrapped)  # no explicit node
        ml.invoke(["x"])
        return context_monitor.totals()["planner"]["calls"]

    assert planner() == 1


def test_get_structured_llm_returns_monitored_without_network():
    llm = get_structured_llm(_Schema)
    assert isinstance(llm, MonitoredLLM)
    assert llm.invoke is not None


def test_agent_nodes_constant():
    assert AGENT_NODES == ("planner", "coder", "tester", "reviewer")
