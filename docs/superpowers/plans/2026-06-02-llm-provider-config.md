# LLM Provider Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `config.yaml` with named LLM profiles and a `load_llm()` function so switching between Groq and Ollama requires editing one line in the config file.

**Architecture:** A new `llm_config.py` module owns `load_llm()`, which reads `config.yaml`, resolves `active_profile`, and returns the correct LangChain chat model. `main.py`'s `on_chat_start()` calls `load_llm()` instead of directly instantiating `ChatGroq`. The Groq API key check moves inside `load_llm()` and is skipped for Ollama.

**Tech Stack:** Python 3.11+, PyYAML, langchain-groq, langchain-ollama, pytest

---

## File map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `llm_config.py` | `load_llm()` — reads config, returns chat model |
| Create | `config.yaml` | Named LLM profiles with `active_profile` selector |
| Create | `tests/test_llm_config.py` | Unit tests for `load_llm()` |
| Modify | `pyproject.toml` | Add `langchain-ollama` dependency |
| Modify | `main.py` | Remove hardcoded `ChatGroq`, wire in `load_llm()` |

---

### Task 1: Add `langchain-ollama` dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the dependency via uv**

```bash
uv add langchain-ollama
```

- [ ] **Step 2: Verify the import resolves**

```bash
python -c "from langchain_ollama import ChatOllama; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add langchain-ollama dependency"
```

---

### Task 2: Create `llm_config.py` with `load_llm()` (TDD)

**Files:**
- Create: `tests/test_llm_config.py`
- Create: `llm_config.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_llm_config.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_llm_config.py -v
```
Expected: `ModuleNotFoundError: No module named 'llm_config'`

- [ ] **Step 3: Create `llm_config.py`**

Create `llm_config.py` at project root:

```python
from __future__ import annotations

import getpass
import os
from pathlib import Path

import yaml
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama


def load_llm(config_path: str | Path = "config.yaml"):
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    active = config.get("active_profile")
    profiles = config.get("profiles", {})

    if active not in profiles:
        raise KeyError(
            f"Profile '{active}' not found in config. Available: {list(profiles.keys())}"
        )

    profile = profiles[active]
    provider = profile["provider"]
    model = profile["model"]
    temperature = profile.get("temperature", 0)

    if provider == "groq":
        if "GROQ_API_KEY" not in os.environ:
            os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")
        return ChatGroq(model=model, temperature=temperature, streaming=True)

    if provider == "ollama":
        base_url = profile.get("base_url", "http://localhost:11434")
        return ChatOllama(model=model, temperature=temperature, base_url=base_url)

    raise ValueError(f"Unknown provider '{provider}'. Valid options: groq, ollama")
```

- [ ] **Step 4: Run tests to verify they all pass**

```bash
pytest tests/test_llm_config.py -v
```
Expected: 6 tests pass, 0 failures.

- [ ] **Step 5: Commit**

```bash
git add llm_config.py tests/test_llm_config.py
git commit -m "feat: add llm_config.py with load_llm() for groq and ollama"
```

---

### Task 3: Create `config.yaml`

**Files:**
- Create: `config.yaml`

- [ ] **Step 1: Create the config file**

Create `config.yaml` at project root:

```yaml
active_profile: groq-default

profiles:
  groq-default:
    provider: groq
    model: openai/gpt-oss-120b
    temperature: 0

  ollama-local:
    provider: ollama
    model: llama3.2
    base_url: http://localhost:11434
    temperature: 0
```

- [ ] **Step 2: Verify the config loads correctly**

```bash
python -c "
import yaml
from llm_config import load_llm
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
print('active_profile:', cfg['active_profile'])
print('profiles:', list(cfg['profiles'].keys()))
"
```
Expected:
```
active_profile: groq-default
profiles: ['groq-default', 'ollama-local']
```

- [ ] **Step 3: Commit**

```bash
git add config.yaml
git commit -m "feat: add config.yaml with groq-default and ollama-local profiles"
```

---

### Task 4: Update `main.py`

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Replace the `ChatGroq` import with `load_llm`**

In `main.py`, replace line 17:
```python
from langchain_groq import ChatGroq
```
with:
```python
from llm_config import load_llm
```

- [ ] **Step 2: Remove the module-level `GROQ_API_KEY` check**

In `main.py`, remove these two lines (currently lines 24–25):
```python
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")
```
Leave the `Albert_API_KEY` block immediately after it untouched.

- [ ] **Step 3: Replace the hardcoded `ChatGroq(...)` block in `on_chat_start()`**

In `main.py` inside `on_chat_start()`, replace:
```python
model = ChatGroq(
    model="openai/gpt-oss-120b",
    # model="deepseek-r1-distill-llama-70b",
    # model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
    streaming=True,
    # other params...
)
```
with:
```python
model = load_llm()
```

- [ ] **Step 4: Run the full test suite**

```bash
pytest tests/ -v
```
Expected: all tests pass (both `test_mapping_v1.py` and `test_llm_config.py`).

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "feat: wire load_llm() into main.py, remove hardcoded ChatGroq"
```
