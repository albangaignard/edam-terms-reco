import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

import llm_config


def _write_config(tmp_path: Path, config: dict) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(config))
    return p


def test_load_llm_groq(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    config_path = _write_config(tmp_path, {
        "active_profile": "groq-default",
        "profiles": {
            "groq-default": {"provider": "groq", "model": "some-model", "temperature": 0}
        },
    })
    with patch.object(llm_config, "ChatGroq") as mock_cls:
        mock_cls.return_value = MagicMock()
        result = llm_config.load_llm(config_path=config_path)
        mock_cls.assert_called_once_with(model="some-model", temperature=0, streaming=True)
        assert result is mock_cls.return_value


def test_load_llm_ollama(tmp_path):
    config_path = _write_config(tmp_path, {
        "active_profile": "ollama-local",
        "profiles": {
            "ollama-local": {
                "provider": "ollama",
                "model": "llama3.2",
                "base_url": "http://localhost:11434",
                "temperature": 0,
            }
        },
    })
    with patch.object(llm_config, "ChatOllama") as mock_cls:
        mock_cls.return_value = MagicMock()
        result = llm_config.load_llm(config_path=config_path)
        mock_cls.assert_called_once_with(
            model="llama3.2", temperature=0, base_url="http://localhost:11434"
        )
        assert result is mock_cls.return_value


def test_load_llm_ollama_default_base_url(tmp_path):
    config_path = _write_config(tmp_path, {
        "active_profile": "ollama-local",
        "profiles": {
            "ollama-local": {"provider": "ollama", "model": "llama3.2", "temperature": 0}
        },
    })
    with patch.object(llm_config, "ChatOllama") as mock_cls:
        mock_cls.return_value = MagicMock()
        llm_config.load_llm(config_path=config_path)
        mock_cls.assert_called_once_with(
            model="llama3.2", temperature=0, base_url="http://localhost:11434"
        )


def test_load_llm_missing_config_raises():
    with pytest.raises(FileNotFoundError, match="nonexistent.yaml"):
        llm_config.load_llm(config_path="nonexistent.yaml")


def test_load_llm_unknown_profile_raises(tmp_path):
    config_path = _write_config(tmp_path, {
        "active_profile": "missing",
        "profiles": {"groq-default": {"provider": "groq", "model": "m", "temperature": 0}},
    })
    with pytest.raises(KeyError, match="missing"):
        llm_config.load_llm(config_path=config_path)


def test_load_llm_unknown_provider_raises(tmp_path):
    config_path = _write_config(tmp_path, {
        "active_profile": "bad",
        "profiles": {"bad": {"provider": "unknown", "model": "m", "temperature": 0}},
    })
    with pytest.raises(ValueError, match="unknown"):
        llm_config.load_llm(config_path=config_path)
