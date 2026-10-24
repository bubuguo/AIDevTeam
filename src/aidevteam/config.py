"""Application settings loaded from environment variables / .env file.

Vendor-neutral by design: any OpenAI-compatible endpoint (NIM, Ollama, vLLM,
OpenRouter, …) works by setting LLM_BASE_URL / LLM_MODEL / LLM_API_KEY.
LLM_BASE_URL and LLM_MODEL are required — there are no defaults.

Runtime tuning (temperature, max_iterations, …) can be set in a Java-style
properties file (default: ./aidevteam.properties, override the path with the
AIDEVTEAM_PROPERTIES_FILE env var). Precedence, highest first:

    CLI flags > environment variables / .env > aidevteam.properties > defaults
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

DEFAULT_PROPERTIES_FILE = "aidevteam.properties"


class PropertiesConfigSource(PydanticBaseSettingsSource):
    """Settings source that reads a Java-style .properties file."""

    def _read_properties(self) -> dict[str, str]:
        path = Path(
            os.environ.get("AIDEVTEAM_PROPERTIES_FILE", DEFAULT_PROPERTIES_FILE)
        )
        if not path.is_file():
            return {}
        props: dict[str, str] = {}
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith(("#", "!")):
                continue
            key, sep, value = line.partition("=")
            if not sep:  # also accept "key: value"
                key, sep, value = line.partition(":")
                if not sep:
                    continue
            props[key.strip()] = value.strip()
        return props

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        props = self._read_properties()
        if field_name in props:
            return props[field_name], field_name, False
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for field_name in self.settings_cls.model_fields:
            value, key, _ = self.get_field_value(
                self.settings_cls.model_fields[field_name], field_name
            )
            if value is not None:
                data[key] = value
        return data


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI-compatible endpoint configuration (same API shape for all vendors)
    llm_api_key: str = ""   # optional; local Ollama needs none
    llm_base_url: str = ""  # REQUIRED
    llm_model: str = ""     # REQUIRED

    # Runtime tuning — settable via aidevteam.properties
    temperature: float = 0.2
    max_iterations: int = 3

    @model_validator(mode="after")
    def _require_llm_endpoint(self) -> "Settings":
        """Fail fast when the required endpoint configuration is missing."""
        missing = [
            env_var
            for field_name, env_var in (
                ("llm_base_url", "LLM_BASE_URL"),
                ("llm_model", "LLM_MODEL"),
            )
            if not getattr(self, field_name).strip()
        ]
        if missing:
            raise ValueError(
                f"Missing required LLM configuration: {', '.join(missing)}. "
                "Set them via environment variables or the .env file."
            )
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,                        # code-passed values (highest)
            env_settings,                         # OS env vars
            dotenv_settings,                      # .env file
            PropertiesConfigSource(settings_cls), # aidevteam.properties
            file_secret_settings,                 # secrets directory (lowest)
        )

    @property
    def resolved_base_url(self) -> str:
        if not self.llm_base_url.strip():
            raise RuntimeError(
                "LLM_BASE_URL is not set — add LLM_BASE_URL=<openai-compatible-endpoint> to your .env"
            )
        return self.llm_base_url.strip()

    @property
    def resolved_model(self) -> str:
        if not self.llm_model.strip():
            raise RuntimeError(
                "LLM_MODEL is not set — add LLM_MODEL=<model-id> to your .env"
            )
        return self.llm_model.strip()

    @property
    def host(self) -> str:
        """Network host of the endpoint, for display purposes."""
        return urlparse(self.resolved_base_url).netloc


@lru_cache
def get_settings() -> Settings:
    return Settings()
