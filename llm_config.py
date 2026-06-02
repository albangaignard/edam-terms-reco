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
