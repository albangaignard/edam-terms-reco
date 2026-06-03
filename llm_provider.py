from __future__ import annotations

import os
from pathlib import Path

import yaml
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


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

    if provider == "ollama":
        base_url = profile.get("base_url", "http://localhost:11434")
        return ChatOllama(model=model, temperature=temperature, base_url=base_url)

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

    raise ValueError(
        f"Unknown provider '{provider}'. Valid options: groq, ollama, albert"
    )
