import chainlit as cl

from rdflib import Graph
from sentence_transformers import SentenceTransformer
import pandas as pd

import faiss
import numpy as np

import getpass
import os
import json
import re

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from typing import cast

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

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

text = df["full_desc"]

## Embeddings
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# if edam_faiss.index exists, load it
try:
    index = faiss.read_index("edam_faiss.index")
    print("Index loaded from disk")
except:
    print("Index not found, creating a new one")    

    # encoder = SentenceTransformer("sentence-transformers/LaBSE")
    # encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
    vectors = encoder.encode(text)

    ## Indexing
    vector_dimension = vectors.shape[1]
    
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vector_dimension)
    #index = faiss.IndexFlatL2(vector_dimension)
    
    index.add(vectors)

    ## serialize the index to disk
    faiss.write_index(index, "edam_faiss.index")

# nlist = 50
# quantizer = faiss.IndexFlatL2(vector_dimension)
# index = faiss.IndexIVFFlat(quantizer, vector_dimension, nlist)
# index.train(vectors)

async def retrieve_classes(question: str) -> str:

    search_vector = encoder.encode(question)
    _vector = np.array([search_vector])
    faiss.normalize_L2(_vector)

    # k = index.ntotal
    # top-K :
    k = 10
    distances, ann = index.search(_vector, k=k)

    # print(f"Distances: {distances}")
    # print(f"ANN: {ann}")

    dist_df = pd.DataFrame(
        {
            "Similarity": pd.Series(distances[0]).apply(lambda x: round(x, 2)),
            "ann": ann[0],
        }
    )

    # join by: dfl.ann = data.index
    top_5 = pd.merge(dist_df, df, left_on="ann", right_index=True)

    for c in top_5["Class ID"]:
        n_sub_classes = count_sub_classes(c, edam_graph)[0]
        top_5.loc[top_5["Class ID"] == c, "child_terms"] = n_sub_classes

    return top_5[["Preferred Label", "Class ID", "Similarity", "child_terms"]]

template = """
Can you annotate with Bioinformatics terms the following text: 

"
{input}
"

INSTRUCTIONS:
    - generate ONLY a JSON structure with the following 4 fields:
        - "topic": the specific bioinformatics or computational biologic scientific area
        - "operation": the specific bioinformatics data processing 
        - "data": the specific type of data
        - "format": the specific data format
    - for each of these fields, provide key value pairs with the top 5 most relevant terms, the key is the term label, the value is a short definitinon.     
    - if you don't find any relevant term for a field, return an empty dictionary for that field    
"""

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
    ]

@cl.on_chat_start
async def on_chat_start():
    
    model = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
    streaming=True,
    # other params...
    )
    #model = ChatOpenAI(streaming=True)
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
    cl.user_session.set("runnable", runnable)

def extract_json(text: str):
    # Look for fenced JSON
    match = re.search(r"```json(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: try parsing whole text
    return text.strip()

@cl.on_message
async def on_message(message: cl.Message):
    """Main function to handle when user send a message to the assistant."""

    runnable = cast(Runnable, cl.user_session.get("runnable"))  # type: Runnable

    msg = cl.Message(content="")
    q = template.format(input=message.content)

    async for chunk in runnable.astream(
        {"question": q},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()

    json_data = json.loads(extract_json(msg.content))
    topic_terms = json_data["topic"]
    operation_terms = json_data["operation"]
    data_terms = json_data["data"]
    format_terms = json_data["format"]

    if len(topic_terms.keys()) > 0:
        relevant_topics = await retrieve_classes(json.dumps(topic_terms))
        answer = cl.Message(
            content="Here are the relevant EDAM Topics: ",
            elements=[
                cl.Dataframe(data=relevant_topics, display="inline", name="Dataframe"),
            ],
        )
        await answer.send()

    if len(operation_terms.keys()) > 0:
        relevant_operation_terms = await retrieve_classes(json.dumps(operation_terms))
        answer = cl.Message(
            content="Here are the relevant EDAM Operations: ",
            elements=[
                cl.Dataframe(data=relevant_operation_terms, display="inline", name="Dataframe"),
            ],
        )
        await answer.send()

    if len(data_terms.keys()) > 0:
        relevant_data_terms = await retrieve_classes(json.dumps(data_terms))
        answer = cl.Message(
            content="Here are the relevant EDAM Data: ",
            elements=[
                cl.Dataframe(data=relevant_data_terms, display="inline", name="Dataframe"),
            ],
        )
        await answer.send()

    if len(format_terms.keys()) > 0:
        relevant_format_terms = await retrieve_classes(json.dumps(format_terms))
        answer = cl.Message(
            content="Here are the relevant EDAM Formats: ",
            elements=[
                cl.Dataframe(data=relevant_format_terms, display="inline", name="Dataframe"),
            ],
        )
        await answer.send()
