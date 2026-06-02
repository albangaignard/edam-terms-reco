import json
import logging
import os
import re
from enum import Enum
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import chainlit as cl
import chromadb
import numpy as np
import pandas as pd
from chainlit.input_widget import Select
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from rdflib import Graph
from sentence_transformers import SentenceTransformer

from providers.config import (
    SUPPORTED_PROVIDERS,
    load_provider_config,
    provider_models_from_env,
)
from providers.factory import build_chat_model

DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "ollama").strip().lower()
ENABLE_DYNAMIC_MODEL_LIST = (
    os.getenv("ENABLE_DYNAMIC_MODEL_LIST", "false").strip().lower() == "true"
)

url = "data/EDAM_1.25.csv"

df = pd.read_csv(url)
df = df[["Class ID", "Preferred Label", "Synonyms", "Definitions", "Parents"]]

edam_graph = Graph()
edam_graph.parse("data/EDAM_1.25.owl", format="xml")


def count_sub_classes(uri, kg):
    query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT (count(?subClass) as ?count) 
    WHERE {{
        ?subClass rdfs:subClassOf+ <{uri}> .
    }}
    """
    res = kg.query(query)
    r = [row["count"] for row in res]
    # print(f"Subclasses of {uri}: {r}")
    return r


def gen_full_desc(row):
    ## Generate a full description of the ontology class
    ## TODO take into account inherited definitions from parents
    full_desc = f"""{row["Preferred Label"]}, also known as {row["Synonyms"]} is an ontology class. It is defined as follows: {row["Definitions"]}. This class is identified with the following identifier: {row["Class ID"]}. """
    return full_desc


df["full_desc"] = df.apply(gen_full_desc, axis=1)

## Embeddings
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize persistent Chroma client
client = chromadb.PersistentClient(path="./chroma_data")

# Get or create collection for all EDAM terms
try:
    index = client.get_collection(name="edam_all")
    print("Collection loaded from disk")
except Exception:
    print("Collection not found, creating a new one")
    # Encode all EDAM terms
    vectors = encoder.encode(df["full_desc"].tolist())

    # Create collection with embeddings
    index = client.create_collection(name="edam_all", metadata={"hnsw:space": "cosine"})

    # Add documents with embeddings and metadata
    index.add(
        ids=[str(i) for i in range(len(df))],
        embeddings=vectors.tolist(),
        documents=df["full_desc"].tolist(),
        metadatas=[
            {
                "class_id": row["Class ID"],
                "preferred_label": row["Preferred Label"],
                "synonyms": str(row["Synonyms"]),
                "definitions": str(row["Definitions"]),
            }
            for _, row in df.iterrows()
        ],
    )


# a python enum  for "topic", "operation", "data", "format"
class EDAMTerms(Enum):
    TOPIC = "topic"
    OPERATION = "operation"
    DATA = "data"
    FORMAT = "format"


class AppMode(Enum):
    RETRIEVER_ONLY = "retriever_only"
    GENERATE_THEN_RETRIEVE = "generate_then_retrieve"
    PER_FIELD_TOP1 = "per_field_top1"


PROFILE_TO_MODE = {
    "EDAM retriever V1": AppMode.RETRIEVER_ONLY,
    "EDAM generator and retriever V2": AppMode.GENERATE_THEN_RETRIEVE,
    "EDAM generator and retriever V3": AppMode.GENERATE_THEN_RETRIEVE,
    "EDAM generator and retriever V4": AppMode.PER_FIELD_TOP1,
}


def _resolve_provider_and_model(settings: dict) -> tuple[str, str]:
    provider = str(settings.get("llm_provider", DEFAULT_PROVIDER)).strip().lower()
    model = str(settings.get("llm_model", "")).strip()
    if not model:
        models = _get_provider_models(provider)
        if not models:
            from providers.config import MODELS_ENV_KEYS

            env_key = MODELS_ENV_KEYS.get(provider, "MODELS")
            raise ValueError(
                f"No models available for provider '{provider}'. "
                f"Set {env_key} or enable dynamic model listing."
            )
        model = models[0]
    return provider, model


def _provider_base_url(provider: str) -> str | None:
    if provider == "ollama":
        return (
            os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip().rstrip("/")
        )
    if provider == "groq":
        return (
            (os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1")
            .strip()
            .rstrip("/")
        )
    if provider == "albert":
        base = os.getenv("ALBERT_BASE_URL", "").strip()
        return base.rstrip("/") if base else None
    if provider == "dev_openai":
        return (
            os.getenv("DEV_OPENAI_BASE_URL", "http://localhost:8000/v1")
            .strip()
            .rstrip("/")
        )
    return None


def _models_url_from_base(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def _fetch_remote_models(provider: str) -> list[str]:
    api_key_env = {
        "groq": "GROQ_API_KEY",
        "albert": "ALBERT_API_KEY",
        "dev_openai": "DEV_OPENAI_API_KEY",
    }
    base_url = _provider_base_url(provider)
    if not base_url:
        return []

    # Ollama exposes installed models at /api/tags, not /models.
    url = (
        f"{base_url}/api/tags"
        if provider == "ollama"
        else _models_url_from_base(base_url)
    )
    headers = {"Accept": "application/json"}
    api_key = (
        os.getenv(api_key_env.get(provider, ""), "").strip()
        if provider in api_key_env
        else ""
    )
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = Request(url, headers=headers, method="GET")
    with urlopen(request, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))

    models = []
    if provider == "ollama":
        # Expected shape: {"models": [{"name": "qwen3:14b", ...}, ...]}
        data = payload.get("models", []) if isinstance(payload, dict) else []
        for item in data:
            model_name = item.get("name") if isinstance(item, dict) else None
            if isinstance(model_name, str) and model_name.strip():
                models.append(model_name.strip())
    else:
        # OpenAI-compatible shape: {"data": [{"id": "..."}]}
        data = payload.get("data", []) if isinstance(payload, dict) else []
        for item in data:
            model_id = item.get("id") if isinstance(item, dict) else None
            if isinstance(model_id, str) and model_id.strip():
                models.append(model_id.strip())
    return models


def _get_provider_models(provider: str) -> list[str]:
    env_models = provider_models_from_env(provider)
    if not ENABLE_DYNAMIC_MODEL_LIST:
        return env_models

    try:
        remote_models = _fetch_remote_models(provider)
        if remote_models:
            return remote_models
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        logging.warning("Could not load dynamic model list for %s: %s", provider, exc)
    return env_models


async def _send_settings_form(initial_provider: str, initial_model: str):
    provider_models = _get_provider_models(initial_provider)
    if not provider_models:
        raise ValueError(
            f"No models available for provider '{initial_provider}'. "
            "Configure MODELS in .env or enable dynamic model listing."
        )

    if initial_model not in provider_models:
        initial_model = provider_models[0]
    widgets = _settings_widgets(initial_provider, initial_model)
    settings = await cl.ChatSettings(widgets).send()
    return settings


async def _refresh_settings_form(provider: str, model: str) -> None:
    widgets = _settings_widgets(provider, model)
    await cl.ChatSettings(widgets).refresh()


def _settings_widgets(provider: str, model: str) -> list[Select]:
    provider_models = _get_provider_models(provider)
    if not provider_models:
        raise ValueError(
            f"No models available for provider '{provider}'. "
            "Configure MODELS in .env or enable dynamic model listing."
        )
    if model not in provider_models:
        model = provider_models[0]
    return [
        Select(
            id="llm_provider",
            label="LLM provider",
            values=list(SUPPORTED_PROVIDERS),
            initial_index=list(SUPPORTED_PROVIDERS).index(provider),
        ),
        Select(
            id="llm_model",
            label="Model",
            values=provider_models,
            initial_index=provider_models.index(model),
        ),
    ]


def _normalized_settings(settings: dict) -> dict:
    provider, model = _resolve_provider_and_model(settings)
    models = _get_provider_models(provider)
    if models and model not in models:
        model = models[0]
    return {"llm_provider": provider, "llm_model": model}


def _merge_settings_update(settings_update: dict) -> dict:
    previous = cl.user_session.get("llm_settings", {}) or {}
    merged = {**previous, **settings_update}
    normalized = _normalized_settings(merged)
    return normalized


def _build_runnable_with_settings(settings: dict) -> tuple[Runnable, str, str]:
    selected_provider, selected_model = _resolve_provider_and_model(settings)
    provider_config = load_provider_config(
        provider_override=selected_provider,
        model_override=selected_model,
    )
    model = build_chat_model(provider_config)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a very knowledgeable computational biologist who provides accurate and eloquent answers to biological questions.",
            ),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    return runnable, provider_config.provider, provider_config.model


async def _apply_llm_settings(settings: dict) -> None:
    settings = _normalized_settings(settings)

    runnable, provider, model = _build_runnable_with_settings(settings)
    cl.user_session.set("runnable", runnable)
    cl.user_session.set("llm_provider", provider)
    cl.user_session.set("llm_model", model)
    cl.user_session.set("llm_settings", settings)


async def _sync_llm_settings_from_session() -> None:
    """
    Fallback for environments where on_settings_update is not emitted reliably.
    """
    session_settings = cl.user_session.get("chat_settings")
    if not isinstance(session_settings, dict):
        return

    current = cl.user_session.get("llm_settings", {}) or {}
    merged = {**current, **session_settings}
    normalized = _normalized_settings(merged)

    if normalized != current:
        await _apply_llm_settings(normalized)


def normalize_embedding_vectors(vectors):
    """Normalize vectors to unit length."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


# function to normalize a query vector
def normalize_query_vector(vector):
    """Normalize a single query vector to unit length."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


indexes = {}
split_classes = {}

for term in EDAMTerms:
    # split_classes[term.value] = df[df["Class ID"].str.contains(".org/" + term.value + "_")].reset_index(drop=True)
    ## copy df where Class ID contains ".org/" + term.value + "_" and reset the index
    split_classes[term.value] = df[
        df["Class ID"].str.contains(".org/" + term.value + "_")
    ].reset_index(drop=True)

    # Try to get existing collection or create new one
    try:
        indexes[term.value] = client.get_collection(name=f"edam_{term.value}")
        print(f"{term.value} index loaded from disk")
    except Exception:
        print(f"{term.value} index not found, creating a new one")
        text_df = df[
            df["Class ID"].str.contains(".org/" + term.value + "_")
        ].reset_index(drop=True)
        print(f"Number of {term.value} terms: {len(text_df)}")

        vectors = encoder.encode(text_df["full_desc"].tolist())
        vectors = normalize_embedding_vectors(vectors)

        ## Create Chroma collection for this term type
        index = client.create_collection(
            name=f"edam_{term.value}", metadata={"hnsw:space": "cosine"}
        )

        # Add documents with embeddings and metadata
        index.add(
            ids=[str(i) for i in range(len(text_df))],
            embeddings=vectors.tolist(),
            documents=text_df["full_desc"].tolist(),
            metadatas=[
                {
                    "class_id": row["Class ID"],
                    "preferred_label": row["Preferred Label"],
                    "synonyms": str(row["Synonyms"]),
                    "definitions": str(row["Definitions"]),
                }
                for _, row in text_df.iterrows()
            ],
        )
        indexes[term.value] = index


print("Setup complete.")
print("Available EDAM term types: ", indexes.keys())
logging.info("Setup complete.")


async def retrieve_from_all_classes(question: str) -> str:
    """Retrieve relevant EDAM classes for a given question by querying the Chroma collection."""
    search_vector = encoder.encode(question)
    search_vector = normalize_query_vector(search_vector)

    # Query Chroma collection
    k = 10
    results = index.query(query_embeddings=[search_vector.tolist()], n_results=k)

    # Extract similarities and class IDs from results
    distances = results["distances"][0] if results["distances"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []

    # Convert distances to similarities (Chroma returns distances, convert them)
    similarities = [
        1 - d for d in distances
    ]  # For cosine distance, similarity = 1 - distance

    dist_df = pd.DataFrame(
        {
            "Similarity": [round(s, 2) for s in similarities],
            "Class ID": [m["class_id"] for m in metadatas],
            "Preferred Label": [m["preferred_label"] for m in metadatas],
        }
    )

    top_k = dist_df.copy()

    for c in top_k["Class ID"]:
        n_sub_classes = count_sub_classes(c, edam_graph)[0]
        top_k.loc[top_k["Class ID"] == c, "child_terms"] = int(n_sub_classes)

    return top_k[["Preferred Label", "Similarity", "child_terms", "Class ID"]]


async def retrieve_classes(question: str, term_type: EDAMTerms, k=5) -> str:
    """Retrieve relevant EDAM classes for a given question and term type (TOPIC, OPERATION, DATA, FORMAT) by querying the corresponding Chroma collection."""
    search_vector = encoder.encode(question)
    search_vector = normalize_query_vector(search_vector)

    collection = indexes[term_type.value]

    # Query Chroma collection
    results = collection.query(query_embeddings=[search_vector.tolist()], n_results=k)

    # Extract results
    distances = results["distances"][0] if results["distances"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []

    if not distances or len(distances) == 0:
        return pd.DataFrame(
            columns=["Preferred Label", "Similarity", "child_terms", "Class ID"]
        )

    # Convert distances to similarities
    similarities = [1 - d for d in distances]

    top_k = pd.DataFrame(
        {
            "Similarity": [round(s, 2) for s in similarities],
            "Class ID": [m["class_id"] for m in metadatas],
            "Preferred Label": [m["preferred_label"] for m in metadatas],
        }
    )

    if len(top_k) == 0:
        return pd.DataFrame(
            columns=["Preferred Label", "Similarity", "child_terms", "Class ID"]
        )
    else:
        for c in top_k["Class ID"]:
            n_sub_classes = count_sub_classes(c, edam_graph)[0]
            top_k.loc[top_k["Class ID"] == c, "child_terms"] = int(n_sub_classes)

        return top_k[["Preferred Label", "Similarity", "child_terms", "Class ID"]]


# a prompt template for the LLM, asking to generate EDAM terms in JSON
# format with 4 fields: topic, operation, data, format

templateV1 = """
QUERY: 
Can you annotate with computational biology terms the following text: 

CONTEXT:
"
{input}
"

INSTRUCTIONS:
    - generate ONLY a JSON structure with the following 4 fields:
        - "topic": the specific bioinformatics or computational biologic scientific area
        - "operation": the specific bioinformatics data processing 
        - "data": the specific type of data
        - "format": the specific data format
    - for each of these fields, provide key value pairs with the top 5 most relevant terms, the key is the term label, the value is a short scientific definition.
    - if you don't find any relevant term for a field, return an empty dictionary for that field
    - ensure the JSON is correctly formatted
    - wrap the JSON in fenced code block like this: ```json ... ```
"""

templateV2 = """
QUERY: 
Can you annotate with computational biology terms the following text: 

CONTEXT:
"
{input}
"

INSTRUCTIONS:
    - generate ONLY a JSON structure with the following 4 fields:
        - "topic": the specific bioinformatics or computational biologic scientific area
        - "operation": the specific bioinformatics data processing 
        - "data": the specific type of data
        - "format": the specific data format
    - for each of these fields, provide key value pairs with the top 5 most relevant terms, 
        the key is the term label, 
        the value is a key-value object 
            ["definition": "short scientific definition",
             "prompt_subset": "the most relevant sentence extracted from the provided CONTEXT used to generate the term"].
    - if you don't find any relevant term for a field, return an empty dictionary for that field
    - ensure the JSON is correctly formatted
    - wrap the JSON in fenced code block like this: ```json ... ```
"""


@cl.set_chat_profiles
async def chat_profile():
    """Define different chat profiles for the assistant. Each profile can represent a different version of the EDAM term retrieval/generation process."""
    return [
        cl.ChatProfile(
            name="EDAM retriever V1",
            markdown_description="Embedding-search over EDAM ontology terms.",
            # icon="https://picsum.photos/200",
        ),
        cl.ChatProfile(
            name="EDAM generator and retriever V2",
            markdown_description="LLM generation of Bioinformatics tags + embedding-search over EDAM ontology terms.",
            # icon="https://picsum.photos/250",
        ),
        cl.ChatProfile(
            name="EDAM generator and retriever V3",
            markdown_description="LLM generation of Bioinformatics tags, including a subset of the prompt + embedding search over EDAM ontology terms.",
            # icon="https://picsum.photos/250",
        ),
        cl.ChatProfile(
            name="EDAM generator and retriever V4",
            markdown_description="LLM generation of Bioinformatics tags, including a subset of the prompt + per-term embedding search over EDAM to retrieve the Top-1 class",
            # icon="https://picsum.photos/250",
        ),
    ]


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Cell counting microscopy",
            message="I want to count my cells on a microscopy image in the context of cardiology.",
        ),
        cl.Starter(
            label="Qiime2 (chatGPT) ",
            message="""
                Rapid advances in DNA-sequencing and bioinformatics technologies in the past two decades have substantially improved understanding of the microbial world. This growing understanding relates to the vast diversity of microorganisms; how microbiota and microbiomes affect disease1 and medical treatment; how microorganisms affect the health of the planet; and the nascent exploration of the medical, forensic, environmental and agricultural applications of microbiome biotechnology. Much of this work has been driven by marker-gene surveys (for example, bacterial/archaeal 16S rRNA genes, fungal internal-transcribed-spacer regions and eukaryotic 18S rRNA genes), which profile microbiota with varying degrees of taxonomic specificity and phylogenetic information. The field is now transitioning to integrate other data types, such as metabolite, metaproteome or metatranscriptome profiles.
            """,
        ),
        cl.Starter(
            label="KEGGGraph Bioconductor package",
            message="""
                KEGGGraph is an interface between KEGG pathway and graph object as well as a collection of tools to analyze, dissect and visualize these graphs. It parses the regularly updated KGML (KEGG XML) files into graph models maintaining all essential pathway attributes. The package offers functionalities including parsing, graph operation, visualization and etc.
                """,
        ),
        cl.Starter(
            label="Melissa Bioconductor package",
            message="""
                Melissa is a Baysian probabilistic model for jointly clustering and imputing single cell methylomes. This is done by taking into account local correlations via a Generalised Linear Model approach and global similarities using a mixture modelling approach.
                """,
        ),
        cl.Starter(
            label="ChEMBL webinar",
            message="""
                ChEMBL is a database of bioactivity for drugs and drug like compounds. This webinar provides an overview of the chemical biology resources at EMBL-EBI focusing on the ChEMBL database. In the first part, the webinar describes the structure of ChEMBL, how data is integrated and curated, and how you can access the information within ChEMBL. In the second part, some worked examples based on drug discovery scenarios have been illustrated. Each worked example will be accompanied by guidance and a demonstration showing how the ChEMBL interface can be used to extract relevant information for drug discovery initiatives.
            """,
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    chat_profile = cl.user_session.get("chat_profile")
    print(f"Chat profile: {chat_profile}")
    mode = PROFILE_TO_MODE.get(chat_profile, AppMode.GENERATE_THEN_RETRIEVE)

    default_provider = (
        DEFAULT_PROVIDER if DEFAULT_PROVIDER in SUPPORTED_PROVIDERS else "ollama"
    )
    default_models = _get_provider_models(default_provider)
    default_model = default_models[0] if default_models else ""
    settings = await _send_settings_form(default_provider, default_model)

    await _apply_llm_settings(settings)
    cl.user_session.set("app_mode", mode.value)


@cl.on_settings_update
async def on_settings_update(settings: dict):
    # Confirmed settings: commit provider/model and rebuild runnable.
    merged_settings = _merge_settings_update(settings)
    await _apply_llm_settings(merged_settings)


@cl.on_settings_edit
async def on_settings_edit(settings: dict):
    # Live UI update while editing (without committing session settings).
    merged = _merge_settings_update(settings)
    provider = merged["llm_provider"]
    model = merged["llm_model"]
    await _refresh_settings_form(provider, model)


def extract_json(text: str):
    # Look for fenced JSON
    match = re.search(r"```json(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: try parsing whole text
    return text.strip()


@cl.action_callback("download_edam_terms")
async def download_message(action):
    content = pd.read_json(json.dumps(action.payload))

    file_name = "edam_terms.csv"
    if "topics" in action.label.lower():
        file_name = "edam_topics.csv"
    elif "operations" in action.label.lower():
        file_name = "edam_operations.csv"
    elif "data" in action.label.lower():
        file_name = "edam_data.csv"
    elif "formats" in action.label.lower():
        file_name = "edam_formats.csv"

    content.to_csv(file_name, index=False)

    await cl.Message(
        content="Here are your EDAM terms:",
        elements=[cl.File(name=file_name, path=file_name, display="inline")],
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Main function to handle when user send a message to the assistant."""
    mode = AppMode(
        cl.user_session.get("app_mode", AppMode.GENERATE_THEN_RETRIEVE.value)
    )

    if mode == AppMode.RETRIEVER_ONLY:
        relevant_classes = await retrieve_from_all_classes(message.content)
        dict_rel_classes = json.loads(relevant_classes.to_json())
        print(dict_rel_classes)

        answer = cl.Message(
            content="Here are the relevant EDAM classes: ",
            elements=[
                cl.Dataframe(data=relevant_classes, display="inline", name="Dataframe"),
            ],
            actions=[
                cl.Action(
                    name="download_edam_terms",
                    label="EDAM terms in CSV",
                    icon="file",
                    payload=dict_rel_classes,
                )
            ],
        )
        await answer.send()

        mean_similarity = round(relevant_classes["Similarity"].mean(), 2)
        mean_child_terms = round(relevant_classes["child_terms"].mean(), 2)
        answer = cl.Message(
            content=f"Here are the merged annotations statistics:\n"
            f"- Mean Similarity: {mean_similarity}\n"
            f"- Mean Child Terms: {mean_child_terms}"
        )
        await answer.send()

    elif mode in (AppMode.GENERATE_THEN_RETRIEVE, AppMode.PER_FIELD_TOP1):
        runnable = cast(Runnable, cl.user_session.get("runnable"))  # type: Runnable

        msg = cl.Message(content="")

        ## by default select templateV2
        q = templateV2.format(input=message.content)

        if cl.user_session.get("chat_profile") == "EDAM generator and retriever V2":
            q = templateV1.format(input=message.content)

        async for chunk in runnable.astream(
            {"question": q},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await msg.stream_token(chunk)

        await msg.send()

        # now that we have LLM-generated JSON, we can parse it and retrieve relevant EDAM terms
        json_data = json.loads(extract_json(msg.content))
        topic_terms = json_data["topic"]
        operation_terms = json_data["operation"]
        data_terms = json_data["data"]
        format_terms = json_data["format"]

        merged_annotations = pd.DataFrame()
        if mode == AppMode.GENERATE_THEN_RETRIEVE:
            # align generated topic tags to EDAM topics
            if len(topic_terms.keys()) > 0:
                relevant_topics = await retrieve_classes(
                    json.dumps(topic_terms), EDAMTerms.TOPIC
                )
                merged_annotations = pd.concat(
                    [merged_annotations, relevant_topics], ignore_index=True
                )
                answer = cl.Message(
                    content="Here are the relevant EDAM Topics: ",
                    elements=[
                        cl.Dataframe(
                            data=relevant_topics, display="inline", name="Dataframe"
                        ),
                    ],
                    actions=[
                        cl.Action(
                            name="download_edam_terms",
                            label="EDAM topics in CSV",
                            icon="file",
                            payload=json.loads(relevant_topics.to_json()),
                        )
                    ],
                )
                await answer.send()

            # align generated operation tags to EDAM operations
            if len(operation_terms.keys()) > 0:
                relevant_operation_terms = await retrieve_classes(
                    json.dumps(operation_terms), EDAMTerms.OPERATION
                )
                merged_annotations = pd.concat(
                    [merged_annotations, relevant_operation_terms], ignore_index=True
                )
                answer = cl.Message(
                    content="Here are the relevant EDAM Operations: ",
                    elements=[
                        cl.Dataframe(
                            data=relevant_operation_terms,
                            display="inline",
                            name="Dataframe",
                        ),
                    ],
                    actions=[
                        cl.Action(
                            name="download_edam_terms",
                            label="EDAM operations in CSV",
                            icon="file",
                            payload=json.loads(relevant_operation_terms.to_json()),
                        )
                    ],
                )
                await answer.send()

            # align generated data tags to EDAM data
            if len(data_terms.keys()) > 0:
                relevant_data_terms = await retrieve_classes(
                    json.dumps(data_terms), EDAMTerms.DATA
                )
                merged_annotations = pd.concat(
                    [merged_annotations, relevant_data_terms], ignore_index=True
                )
                answer = cl.Message(
                    content="Here are the relevant EDAM Data: ",
                    elements=[
                        cl.Dataframe(
                            data=relevant_data_terms, display="inline", name="Dataframe"
                        ),
                    ],
                    actions=[
                        cl.Action(
                            name="download_edam_terms",
                            label="EDAM data in CSV",
                            icon="file",
                            payload=json.loads(relevant_data_terms.to_json()),
                        )
                    ],
                )
                await answer.send()

            # align generated format tags to EDAM formats
            if len(format_terms.keys()) > 0:
                relevant_format_terms = await retrieve_classes(
                    json.dumps(format_terms), EDAMTerms.FORMAT
                )
                merged_annotations = pd.concat(
                    [merged_annotations, relevant_format_terms], ignore_index=True
                )
                answer = cl.Message(
                    content="Here are the relevant EDAM Formats: ",
                    elements=[
                        cl.Dataframe(
                            data=relevant_format_terms,
                            display="inline",
                            name="Dataframe",
                        ),
                    ],
                    actions=[
                        cl.Action(
                            name="download_edam_terms",
                            label="EDAM formats in CSV",
                            icon="file",
                            payload=json.loads(relevant_format_terms.to_json()),
                        )
                    ],
                )
                await answer.send()

            mean_similarity = round(merged_annotations["Similarity"].mean(), 2)
            mean_child_terms = round(merged_annotations["child_terms"].mean(), 2)
            answer = cl.Message(
                content=f"Here are the merged annotations statistics:\n"
                f"- Mean Similarity: {mean_similarity}\n"
                f"- Mean Child Terms: {mean_child_terms}"
            )
            await answer.send()

        elif mode == AppMode.PER_FIELD_TOP1:
            # for each of the 4 fields, we have a dictionary with key = term label, value = {definition, prompt_subset}
            # we will use the prompt_subset to retrieve the most relevant EDAM class

            # retrieve a single EDAM topic for each generated topic term
            edam_topics = pd.DataFrame()
            for k, v in topic_terms.items():
                # top_class = await retrieve_classes(json.dumps(v), EDAMTerms.TOPIC, k=2)
                # # sort by child_terms ascending
                # top_class = top_class.sort_values(by=["child_terms"], ascending=True).head(1)
                top_class = await retrieve_classes(json.dumps(v), EDAMTerms.TOPIC, k=1)
                top_class["Generated term"] = k

                edam_topics = pd.concat([edam_topics, top_class], ignore_index=True)
            if not edam_topics.empty:
                edam_topics.sort_values(
                    by=["Similarity"], ascending=False, inplace=True
                )

            # remove duplicates based on "Class ID"
            edam_topics = edam_topics.drop_duplicates(subset=["Class ID"])

            answer = cl.Message(
                content="Here are the relevant EDAM Topics: ",
                elements=[
                    cl.Dataframe(data=edam_topics, display="inline", name="Dataframe"),
                ],
                actions=[
                    cl.Action(
                        name="download_edam_terms",
                        label="EDAM topics in CSV",
                        icon="file",
                        payload=json.loads(edam_topics.to_json()),
                    )
                ],
            )
            await answer.send()

            # retrieve a single EDAM operation for each generated operation term
            edam_operations = pd.DataFrame()
            for k, v in operation_terms.items():
                # top_class = await retrieve_classes(json.dumps(v), EDAMTerms.OPERATION, k=2)
                # # sort by child_terms ascending
                # top_class = top_class.sort_values(by=["child_terms"], ascending=True).head(1)
                top_class = await retrieve_classes(
                    json.dumps(v), EDAMTerms.OPERATION, k=1
                )
                top_class["Generated term"] = k
                edam_operations = pd.concat(
                    [edam_operations, top_class], ignore_index=True
                )
            if not edam_operations.empty:
                edam_operations.sort_values(
                    by=["Similarity"], ascending=False, inplace=True
                )

            # remove duplicates based on "Class ID"
            edam_operations = edam_operations.drop_duplicates(subset=["Class ID"])

            answer = cl.Message(
                content="Here are the relevant EDAM Operations: ",
                elements=[
                    cl.Dataframe(
                        data=edam_operations, display="inline", name="Dataframe"
                    ),
                ],
                actions=[
                    cl.Action(
                        name="download_edam_terms",
                        label="EDAM operations in CSV",
                        icon="file",
                        payload=json.loads(edam_operations.to_json()),
                    )
                ],
            )
            await answer.send()

            # retrieve a single EDAM data for each generated data term
            edam_data = pd.DataFrame()
            for k, v in data_terms.items():
                # top_class = await retrieve_classes(json.dumps(v), EDAMTerms.DATA, k=2)
                # # sort by child_terms ascending
                # top_class = top_class.sort_values(by=["child_terms"], ascending=True).head(1)
                top_class = await retrieve_classes(json.dumps(v), EDAMTerms.DATA, k=1)
                top_class["Generated term"] = k
                edam_data = pd.concat([edam_data, top_class], ignore_index=True)
            if not edam_data.empty:
                edam_data.sort_values(by=["Similarity"], ascending=False, inplace=True)

            # remove duplicates based on "Class ID"
            edam_data = edam_data.drop_duplicates(subset=["Class ID"])

            answer = cl.Message(
                content="Here are the relevant EDAM Data: ",
                elements=[
                    cl.Dataframe(data=edam_data, display="inline", name="Dataframe"),
                ],
                actions=[
                    cl.Action(
                        name="download_edam_terms",
                        label="EDAM data in CSV",
                        icon="file",
                        payload=json.loads(edam_data.to_json()),
                    )
                ],
            )
            await answer.send()

            # retrieve a single EDAM format for each generated format term
            edam_formats = pd.DataFrame()
            for k, v in format_terms.items():
                # top_class = await retrieve_classes(json.dumps(v), EDAMTerms.FORMAT, k=2)
                # # sort by child_terms ascending
                # top_class = top_class.sort_values(by=["child_terms"], ascending=True).head(1)

                top_class = await retrieve_classes(json.dumps(v), EDAMTerms.FORMAT, k=1)
                top_class["Generated term"] = k
                edam_formats = pd.concat([edam_formats, top_class], ignore_index=True)
            if not edam_formats.empty:
                edam_formats.sort_values(
                    by=["Similarity"], ascending=False, inplace=True
                )

            # remove duplicates based on "Class ID"
            edam_formats = edam_formats.drop_duplicates(subset=["Class ID"])

            answer = cl.Message(
                content="Here are the relevant EDAM Formats: ",
                elements=[
                    cl.Dataframe(data=edam_formats, display="inline", name="Dataframe"),
                ],
                actions=[
                    cl.Action(
                        name="download_edam_terms",
                        label="EDAM formats in CSV",
                        icon="file",
                        payload=json.loads(edam_formats.to_json()),
                    )
                ],
            )
            await answer.send()

            merged_annotations = pd.DataFrame()
            if not edam_topics.empty:
                merged_annotations = pd.concat(
                    [merged_annotations, edam_topics], ignore_index=True
                )
            if not edam_operations.empty:
                merged_annotations = pd.concat(
                    [merged_annotations, edam_operations], ignore_index=True
                )
            if not edam_data.empty:
                merged_annotations = pd.concat(
                    [merged_annotations, edam_data], ignore_index=True
                )
            if not edam_formats.empty:
                merged_annotations = pd.concat(
                    [merged_annotations, edam_formats], ignore_index=True
                )

            mean_similarity = round(merged_annotations["Similarity"].mean(), 2)
            mean_child_terms = round(merged_annotations["child_terms"].mean(), 2)
            answer = cl.Message(
                content=f"Here are the merged annotations statistics:\n"
                f"- Mean Similarity: {mean_similarity}\n"
                f"- Mean Child Terms: {mean_child_terms}"
            )
            await answer.send()

        else:
            print("Unknown chat profile, please select a valid chat profile.")
            answer = cl.Message(
                content="Unknown chat profile, please select a valid chat profile."
            )
            await answer.send()
