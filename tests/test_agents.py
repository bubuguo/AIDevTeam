"""Unit tests for the agent naming convention (loading + error reporting)."""

import sys
import types

import pytest

from aidevteam.agents import load_node


def test_load_node_resolves_by_convention():
    for name in ("planner", "coder", "tester", "reviewer"):
        node = load_node(name)
        assert callable(node)
        assert node.__name__ == name  # function name == agent name


def test_load_node_unknown_agent_has_clear_error():
    with pytest.raises(AttributeError, match="Unknown agent 'nonexistent'"):
        load_node("nonexistent")


def test_load_node_flags_convention_violation(monkeypatch):
    """A module that exists but does not define <name>() must fail loudly."""
    stub = types.ModuleType("aidevteam.agents.badagent")  # no `badagent` attr
    monkeypatch.setitem(sys.modules, "aidevteam.agents.badagent", stub)

    with pytest.raises(AttributeError, match="Naming convention violated"):
        load_node("badagent")