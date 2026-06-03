import os

import pytest
import httpx
from openai import OpenAI

ALBERT_BASE_URL = "https://albert.api.etalab.gouv.fr/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
LMSTUDIO_BASE_URL = "http://localhost:1234/v1"


def _lmstudio_reachable() -> bool:
    try:
        httpx.get(f"{LMSTUDIO_BASE_URL}/models", timeout=2.0)
        return True
    except Exception:
        return False

@pytest.mark.skipif(
    not os.environ.get("ALBERT_API_KEY"),
    reason="ALBERT_API_KEY not set — skipping live Albert API test",
)
def test_albert_lists_models():
    client = OpenAI(
        api_key=os.environ["ALBERT_API_KEY"],
        base_url=ALBERT_BASE_URL,
    )
    models = client.models.list()
    model_ids = [m.id for m in models.data]
    assert len(model_ids) > 0, "Albert API returned an empty model list"

@pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set — skipping live GROQ API test",
)
def test_groq_lists_models():
    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url=GROQ_BASE_URL,
    )
    models = client.models.list()
    model_ids = [m.id for m in models.data]
    assert len(model_ids) > 0, "GROQ API returned an empty model list"


@pytest.mark.skipif(
    not _lmstudio_reachable(),
    reason="LM Studio not reachable at localhost:1234 — skipping live LM Studio test",
)
def test_lmstudio_lists_models():
    client = OpenAI(api_key="lm-studio", base_url=LMSTUDIO_BASE_URL)
    models = client.models.list()
    model_ids = [m.id for m in models.data]
    assert len(model_ids) > 0, "LM Studio returned an empty model list"
