try:
    from langchain_core.language_models.chat_models import BaseChatModel
except ImportError:
    BaseChatModel = object

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

from providers.config import ProviderConfig


def _ensure_dependency(dep: object, package_name: str, provider: str) -> None:
    if dep is None:
        raise ImportError(
            f"Provider '{provider}' requires {package_name}. Install with: pip install {package_name}"
        )


def build_chat_model(config: ProviderConfig) -> BaseChatModel:
    if config.provider == "ollama":
        _ensure_dependency(ChatOllama, "langchain-ollama", config.provider)
        return ChatOllama(
            model=config.model,
            base_url=config.base_url,
            temperature=config.temperature,
        )

    if config.provider == "groq":
        _ensure_dependency(ChatGroq, "langchain-groq", config.provider)
        return ChatGroq(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            reasoning_format=config.reasoning_format,
            timeout=config.timeout,
            max_retries=config.max_retries,
            streaming=config.streaming,
        )

    if config.provider in ("albert", "dev_openai"):
        _ensure_dependency(ChatOpenAI, "langchain-openai", config.provider)
        return ChatOpenAI(
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key,
            temperature=config.temperature,
            timeout=config.timeout,
            max_retries=config.max_retries,
            streaming=config.streaming,
            max_tokens=config.max_tokens,
        )

    raise ValueError(f"Unsupported provider '{config.provider}'.")
