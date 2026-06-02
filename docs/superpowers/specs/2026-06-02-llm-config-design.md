# LLM Provider Config — Design Spec

**Date:** 2026-06-02  
**Scope:** `main.py` only  
**Status:** Approved

---

## Goal

Allow switching between Groq (cloud) and Ollama (local) without touching Python source code. A single `config.yaml` at the project root controls which provider and model are active.

---

## Config file (`config.yaml`)

New file at repo root. `active_profile` selects which named profile block is used at startup.

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

Supported keys per profile:

| Key | Required | Default | Notes |
|-----|----------|---------|-------|
| `provider` | yes | — | `groq` or `ollama` |
| `model` | yes | — | Model identifier string |
| `temperature` | no | `0` | Float |
| `base_url` | no | `http://localhost:11434` | Ollama only |

---

## Changes to `main.py`

### New `load_llm()` function

Added near the top of the file (after imports). Reads `config.yaml`, resolves `active_profile`, and returns a LangChain chat model:

- `provider: groq` → returns `ChatGroq(...)`
- `provider: ollama` → returns `ChatOllama(...)` from `langchain-ollama`
- Unknown provider → raises `ValueError` with a clear message

### `on_chat_start()` update

Replace the hardcoded `ChatGroq(...)` block with:

```python
model = load_llm()
```

### Conditional API key check

The `GROQ_API_KEY` env-var check at module top becomes conditional — only runs when the active profile's provider is `groq`. Ollama requires no API key.

---

## Dependencies

Add `langchain-ollama` to `pyproject.toml` dependencies. `langchain-groq` is already present.

---

## Error handling

- Missing `config.yaml`: raise `FileNotFoundError` with a message pointing to the file.
- `active_profile` not found in `profiles`: raise `KeyError` with the profile name.
- Unknown `provider` value: raise `ValueError` naming the bad value and listing valid options.

---

## Out of scope

- `main_biotools.py` and `main_hpo.py` — unchanged.
- Hot-reloading config mid-session.
- Env-var override of `active_profile`.
