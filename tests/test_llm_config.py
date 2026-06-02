from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from providers.config import load_provider_config
from providers.factory import build_chat_model


def test_load_provider_config_defaults_to_ollama(monkeypatch) -> None:
    monkeypatch.delenv("DEFAULT_PROVIDER", raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODELS", "qwen3:14b")

    config = load_provider_config()

    assert config.provider == "ollama"
    assert config.model == "qwen3:14b"
    assert config.base_url == "http://localhost:11434"


def test_load_provider_config_groq_requires_api_key(monkeypatch) -> None:
    monkeypatch.setenv("DEFAULT_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_MODELS", "openai/gpt-oss-120b")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        load_provider_config()


def test_load_provider_config_requires_models_list(monkeypatch) -> None:
    monkeypatch.setenv("DEFAULT_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.delenv("OLLAMA_MODELS", raising=False)

    with pytest.raises(ValueError, match="OLLAMA_MODELS"):
        load_provider_config()


def test_load_provider_config_albert_supports_overrides(monkeypatch) -> None:
    monkeypatch.setenv("ALBERT_BASE_URL", "https://api.albert.com/v1")
    monkeypatch.setenv("ALBERT_API_KEY", "env-key")
    monkeypatch.setenv("ALBERT_MODELS", "env-model,other-model")

    config = load_provider_config(
        provider_override="albert",
        model_override="override-model",
        base_url_override="https://override.example/v1",
        api_key_override="override-key",
        temperature_override=0.3,
    )

    assert config.provider == "albert"
    assert config.model == "override-model"
    assert config.base_url == "https://override.example/v1"
    assert config.api_key == "override-key"
    assert config.temperature == 0.3


def test_load_provider_config_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported provider"):
        load_provider_config(provider_override="unknown-provider")


def test_build_chat_model_groq_calls_chatgroq_with_expected_args() -> None:
    config = load_provider_config(
        provider_override="groq",
        model_override="openai/gpt-oss-120b",
        api_key_override="test-key",
    )

    with patch("providers.factory.ChatGroq") as mock_chat_groq:
        mock_chat_groq.return_value = MagicMock()
        result = build_chat_model(config)

        mock_chat_groq.assert_called_once()
        call_kwargs = mock_chat_groq.call_args.kwargs
        assert call_kwargs["model"] == "openai/gpt-oss-120b"
        assert call_kwargs["temperature"] == config.temperature
        assert call_kwargs["streaming"] is True
        assert result is mock_chat_groq.return_value


def test_build_chat_model_ollama_calls_chatollama_with_expected_args() -> None:
    config = load_provider_config(
        provider_override="ollama",
        model_override="qwen3:14b",
        base_url_override="http://localhost:11434",
    )

    with patch("providers.factory.ChatOllama") as mock_chat_ollama:
        mock_chat_ollama.return_value = MagicMock()
        result = build_chat_model(config)

        mock_chat_ollama.assert_called_once_with(
            model="qwen3:14b",
            base_url="http://localhost:11434",
            temperature=config.temperature,
        )
        assert result is mock_chat_ollama.return_value


def test_build_chat_model_openai_compatible_calls_chatopenai() -> None:
    config = load_provider_config(
        provider_override="dev_openai",
        model_override="local-model",
        base_url_override="http://localhost:8000/v1",
        api_key_override="dummy-key",
    )

    with patch("providers.factory.ChatOpenAI") as mock_chat_openai:
        mock_chat_openai.return_value = MagicMock()
        result = build_chat_model(config)

        mock_chat_openai.assert_called_once()
        call_kwargs = mock_chat_openai.call_args.kwargs
        assert call_kwargs["model"] == "local-model"
        assert call_kwargs["base_url"] == "http://localhost:8000/v1"
        assert call_kwargs["api_key"] == "dummy-key"
        assert result is mock_chat_openai.return_value


def test_build_chat_model_missing_dependency_raises(monkeypatch) -> None:
    config = load_provider_config(
        provider_override="groq",
        model_override="openai/gpt-oss-120b",
        api_key_override="test-key",
    )
    monkeypatch.setattr("providers.factory.ChatGroq", None)

    with pytest.raises(ImportError, match="langchain-groq"):
        build_chat_model(config)
