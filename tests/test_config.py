"""Unit tests for Settings (config precedence and env-var mapping)."""

import pytest

from aidevteam.config import Settings


@pytest.fixture()
def base_kwargs():
    """Satisfy the fail-fast LLM requirement without touching .env."""
    return {"llm_base_url": "http://llm.example", "llm_model": "test-model"}


def test_node_url_is_derived_from_role(base_kwargs):
    settings = Settings(**base_kwargs)
    assert settings.node_url("planner") == "http://planner:8000"
    assert settings.node_url("tester") == "http://tester:8000"


def test_node_settings_read_from_properties_file(tmp_path, monkeypatch, base_kwargs):
    """Node execution tuning is settable via aidevteam.properties."""
    props = tmp_path / "aidevteam.properties"
    props.write_text("node_port=9000\nnode_poll_interval=0.5\nnode_max_wait=60\n")
    monkeypatch.setenv("AIDEVTEAM_PROPERTIES_FILE", str(props))

    settings = Settings(**base_kwargs)
    assert settings.node_port == 9000
    assert settings.node_poll_interval == 0.5
    assert settings.node_max_wait == 60.0
    assert settings.node_url("planner") == "http://planner:9000"


def test_llm_timeout_read_from_properties_file(tmp_path, monkeypatch, base_kwargs):
    props = tmp_path / "aidevteam.properties"
    props.write_text("llm_timeout=45\n")
    monkeypatch.setenv("AIDEVTEAM_PROPERTIES_FILE", str(props))
    assert Settings(**base_kwargs).llm_timeout == 45.0
