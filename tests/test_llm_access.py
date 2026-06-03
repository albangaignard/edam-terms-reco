import os

import pytest
from openai import OpenAI

ALBERT_BASE_URL = "https://albert.api.etalab.gouv.fr/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

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
