from __future__ import annotations

import os
from pathlib import Path

import yaml
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


_PROVIDER_DISPLAY = {
    "groq": "Groq",
    "albert": "Albert",
    "ollama": "Ollama",
    "lmstudio": "LMStudio",
}


def build_provider_profiles(config_path: str | Path = "config.yaml") -> dict[str, str]:
    """Return {display_label: profile_key} for every profile in config.yaml."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        config = yaml.safe_load(f)
    result = {}
    for key, profile in config.get("profiles", {}).items():
        provider = profile.get("provider", key)
        model = profile.get("model", "")
        model_basename = model.split("/")[-1]
        display = _PROVIDER_DISPLAY.get(provider, provider.capitalize())
        result[f"{display} - {model_basename}"] = key
    return result


def load_profile(config_path: str | Path = "config.yaml", profile_name: str | None = None) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        config = yaml.safe_load(f)
    active = profile_name if profile_name is not None else config.get("active_profile")
    profiles = config.get("profiles", {})
    if active not in profiles:
        raise KeyError(f"Profile '{active}' not found in config. Available: {list(profiles.keys())}")
    return profiles[active]


def load_llm(config_path: str | Path = "config.yaml", profile_name: str | None = None):
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    active = profile_name if profile_name is not None else config.get("active_profile")
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
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY environment variable is not set.")
        return ChatGroq(model=model, temperature=temperature, streaming=True)

    if provider == "albert":
        api_key_env = profile.get("api_key_env", "ALBERT_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(f"{api_key_env} environment variable is not set.")
        base_url = profile.get("base_url", "https://albert.api.etalab.gouv.fr/v1")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
            streaming=True,
        )

    if provider == "lmstudio":
        base_url = profile.get("base_url", "http://localhost:1234/v1")
        api_key = os.environ.get(profile.get("api_key_env", ""), "lm-studio")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
            streaming=True,
        )

    raise ValueError(
        f"Unknown provider '{provider}'. Valid options: groq, ollama, albert, lmstudio"
    )
