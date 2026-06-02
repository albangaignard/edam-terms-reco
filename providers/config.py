import os
from dataclasses import dataclass
from typing import Any


SUPPORTED_PROVIDERS = ("ollama", "groq", "albert", "dev_openai")

MODELS_ENV_KEYS = {
    "ollama": "OLLAMA_MODELS",
    "groq": "GROQ_MODELS",
    "albert": "ALBERT_MODELS",
    "dev_openai": "DEV_OPENAI_MODELS",
}


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    max_retries: int = 2
    reasoning_format: str | None = None
    timeout: float | None = None
    streaming: bool = True
    max_tokens: int | None = None


def _non_empty(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def _require(name: str, value: str | None) -> str:
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def parse_models_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [m.strip() for m in value.split(",") if m.strip()]


def provider_models_from_env(provider: str) -> list[str]:
    env_key = MODELS_ENV_KEYS.get(provider)
    if not env_key:
        return []
    return parse_models_list(os.getenv(env_key))


def _provider_defaults(provider: str, *, require_models: bool = True) -> dict[str, Any]:
    env_key = MODELS_ENV_KEYS[provider]
    models = provider_models_from_env(provider)
    if require_models and not models:
        raise ValueError(
            f"Missing or empty {env_key}: provide a comma-separated model list"
        )
    model = models[0] if models else ""

    if provider == "ollama":
        return {
            "model": model,
            "base_url": _non_empty(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")),
            "api_key": None,
            "reasoning_format": None,
        }
    if provider == "groq":
        return {
            "model": model,
            "base_url": None,
            "api_key": _non_empty(os.getenv("GROQ_API_KEY")),
            "reasoning_format": "parsed",
        }
    if provider == "albert":
        return {
            "model": model,
            "base_url": _non_empty(os.getenv("ALBERT_BASE_URL")),
            "api_key": _non_empty(os.getenv("ALBERT_API_KEY")),
            "reasoning_format": None,
        }
    if provider == "dev_openai":
        return {
            "model": model,
            "base_url": _non_empty(os.getenv("DEV_OPENAI_BASE_URL", "http://localhost:8000/v1")),
            "api_key": _non_empty(os.getenv("DEV_OPENAI_API_KEY", "dummy-key")),
            "reasoning_format": None,
        }
    raise ValueError(
        f"Unsupported provider '{provider}'. Supported providers: {', '.join(SUPPORTED_PROVIDERS)}"
    )


def load_provider_config(
    provider_override: str | None = None,
    model_override: str | None = None,
    base_url_override: str | None = None,
    api_key_override: str | None = None,
    temperature_override: float | None = None,
) -> ProviderConfig:
    provider = (provider_override or os.getenv("DEFAULT_PROVIDER", "ollama")).strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported provider '{provider}'. Supported providers: {', '.join(SUPPORTED_PROVIDERS)}"
        )

    defaults = _provider_defaults(
        provider, require_models=not _non_empty(model_override)
    )
    model = _non_empty(model_override) or defaults["model"]
    if not model:
        env_key = MODELS_ENV_KEYS[provider]
        raise ValueError(
            f"Missing or empty {env_key}: provide a comma-separated model list"
        )

    config = ProviderConfig(
        provider=provider,
        model=model,
        base_url=_non_empty(base_url_override) or defaults["base_url"],
        api_key=_non_empty(api_key_override) or defaults["api_key"],
        temperature=(
            temperature_override
            if temperature_override is not None
            else float(os.getenv("LLM_TEMPERATURE", "0"))
        ),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
        reasoning_format=defaults["reasoning_format"],
        timeout=None,
        streaming=True,
        max_tokens=None,
    )
    return validate_provider_config(config)


def validate_provider_config(config: ProviderConfig) -> ProviderConfig:
    if config.provider == "groq":
        _require("GROQ_API_KEY", config.api_key)
    elif config.provider == "albert":
        _require("ALBERT_BASE_URL", config.base_url)
        _require("ALBERT_API_KEY", config.api_key)
    elif config.provider == "dev_openai":
        _require("DEV_OPENAI_BASE_URL", config.base_url)
    elif config.provider == "ollama":
        _require("OLLAMA_BASE_URL", config.base_url)
    return config
