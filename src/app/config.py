from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from dotenv import load_dotenv
from pydantic import Field
from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    ollama_base_url: str = Field(default="http://127.0.0.1:11434/v1", alias="OLLAMA_BASE_URL")
    advisor_llm: str = Field(default="ollama:llama3.2:1b", alias="ADVISOR_LLM")
    client_llm: str = Field(default="ollama:llama3.2:1b", alias="CLIENT_LLM")
    llm_timeout_seconds: int = Field(default=60, alias="LLM_TIMEOUT_SECONDS")
    database_url: str = Field(default="sqlite:///./advisor.db", alias="DATABASE_URL")
    chroma_dir: str = Field(default="./.chroma", alias="CHROMA_DIR")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    claude_model: str = Field(default="claude-3-5-sonnet-latest", alias="CLAUDE_MODEL")
    analyst_web_models: Annotated[list[str], NoDecode] = Field(
        default_factory=list, alias="ANALYST_WEB_MODELS"
    )
    analyst_summary_models: Annotated[list[str], NoDecode] = Field(
        default_factory=list, alias="ANALYST_SUMMARY_MODELS"
    )
    analyst_web_llms: Annotated[list[str], NoDecode] = Field(
        default_factory=list, alias="ANALYST_WEB_LLMS"
    )
    analyst_summary_llms: Annotated[list[str], NoDecode] = Field(
        default_factory=list, alias="ANALYST_SUMMARY_LLMS"
    )

    @field_validator(
        "anthropic_api_key",
        "openai_api_key",
        "gemini_api_key",
        mode="before",
    )
    @classmethod
    def normalize_placeholder_keys(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        cleaned = value.strip()
        if not cleaned:
            return ""
        lowered = cleaned.lower()
        if lowered.startswith("your_") or lowered in {"changeme", "replace_me"}:
            return ""
        return cleaned

    @field_validator(
        "analyst_web_models",
        "analyst_summary_models",
        "analyst_web_llms",
        "analyst_summary_llms",
        mode="before",
    )
    @classmethod
    def parse_csv_models(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
