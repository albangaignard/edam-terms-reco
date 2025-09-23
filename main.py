from enum import Enum
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
    vectors = encoder.encode(df["full_desc"])

    ## Indexing
    vector_dimension = vectors.shape[1]
    
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vector_dimension)
    index.add(vectors)

    ## serialize the index to disk
    faiss.write_index(index, "edam_faiss.index")


# a python enum  for "topic", "operation", "data", "format"
class EDAMTerms(Enum):
    TOPIC = "topic"
    OPERATION = "operation"
    DATA = "data"
    FORMAT = "format"

indexes = {}
split_classes = {} 

for term in EDAMTerms:

    #split_classes[term.value] = df[df["Class ID"].str.contains(".org/" + term.value + "_")].reset_index(drop=True)
    ## copy df where Class ID contains ".org/" + term.value + "_" and reset the index
    split_classes[term.value] = df[df["Class ID"].str.contains(".org/" + term.value + "_")].reset_index(drop=True)

    # if edam_faiss.index exists, load it
    try:
        indexes[term.value] = faiss.read_index(f"edam_{term.value}_faiss.index")
        print(f"{term.value} index loaded from disk")
    except:
        print(f"{term.value} index not found, creating a new one")
        text = df[df["Class ID"].str.contains(".org/" + term.value + "_")]["full_desc"]
        print(f"Number of {term.value} terms: {len(text)}")

        vectors = encoder.encode(text.to_list())

        ## Indexing
        vector_dimension = vectors.shape[1]
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vector_dimension)
        #index = faiss.IndexFlatL2(vector_dimension)
        index.add(vectors)

        ## serialize the index to disk
        faiss.write_index(index, f"edam_{term.value}_faiss.index")
        indexes[term.value] = index


print("Setup complete.")
print("Available EDAM term types: ", indexes.keys())

async def retrieve_from_all_classes(question: str) -> str:

    search_vector = encoder.encode(question)
    _vector = np.array([search_vector])
    faiss.normalize_L2(_vector)

    # top-K :
    k = 20
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
        top_5.loc[top_5["Class ID"] == c, "child_terms"] = int(n_sub_classes)

    return top_5[["Preferred Label", "Similarity", "child_terms", "Class ID"]]

async def retrieve_classes(question: str, term_type: EDAMTerms, k = 5) -> str:

    search_vector = encoder.encode(question)
    _vector = np.array([search_vector])
    faiss.normalize_L2(_vector)

    index = indexes[term_type.value]

    distances, ann = index.search(_vector, k=k)

    #print(f"Distances: {distances[0]}")
    #print(f"ANN: {ann[0]}")
    dist_df = pd.DataFrame(
        {
            "Similarity": pd.Series(distances[0]).apply(lambda x: round(x, 2)),
            "ann": ann[0],
        }
    )

    top_k = pd.merge(dist_df, split_classes[term_type.value], left_on="ann", right_index=True)

    if len(top_k) == 0:
        return pd.DataFrame(columns=["Preferred Label", "Similarity", "child_terms", "Class ID"])
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
    return [
        cl.ChatProfile(
            name="EDAM retriever V1",
            markdown_description="FAISS index over EDAM ontology terms.",
            # icon="https://picsum.photos/200",
        ),
        cl.ChatProfile(
            name="EDAM generator and retriever V2",
            markdown_description="LLM generation of Bioinformatics tags + global FAISS index search over EDAM ontology terms.",
            #icon="https://picsum.photos/250",
        ),
        cl.ChatProfile(
            name="EDAM generator and retriever V3",
            markdown_description="LLM generation of Bioinformatics tags, including a subset of the prompt + global FAISS index search over EDAM ontology terms.",
            # icon="https://picsum.photos/250",
        ),
        cl.ChatProfile(
            name="EDAM generator and retriever V4",
            markdown_description="LLM generation of Bioinformatics tags, including a subset of the prompt + per-term FAISS index search over EDAM to retrieve the Top-1 class",
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
    
    model = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    #model="qwen/qwen3-32b",
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

    if cl.user_session.get("chat_profile") == "EDAM retriever V1":
        relevant_classes = await retrieve_from_all_classes(message.content)
        dict_rel_classes = json.loads(relevant_classes.to_json())
        print(dict_rel_classes)

        answer = cl.Message(
            content="Here are the relevant EDAM classes: ",
            elements=[
                cl.Dataframe(data=relevant_classes, display="inline", name="Dataframe"),
            ],
            actions=[cl.Action(name="download_edam_terms", label="EDAM terms in CSV", icon="file", payload=dict_rel_classes)]
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

    elif "EDAM generator" in cl.user_session.get("chat_profile"):

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
        if ("V2" in cl.user_session.get("chat_profile")) or ("V3" in cl.user_session.get("chat_profile")):
            # align generated topic tags to EDAM topics
            if len(topic_terms.keys()) > 0:
                relevant_topics = await retrieve_classes(json.dumps(topic_terms), EDAMTerms.TOPIC)
                merged_annotations = pd.concat([merged_annotations, relevant_topics], ignore_index=True)
                answer = cl.Message(
                    content="Here are the relevant EDAM Topics: ",
                    elements=[
                        cl.Dataframe(data=relevant_topics, display="inline", name="Dataframe"),
                    ],
                    actions=[cl.Action(name="download_edam_terms", label="EDAM topics in CSV", icon="file", payload=json.loads(relevant_topics.to_json()))]
                )
                await answer.send()

            # align generated operation tags to EDAM operations
            if len(operation_terms.keys()) > 0:
                relevant_operation_terms = await retrieve_classes(json.dumps(operation_terms), EDAMTerms.OPERATION)
                merged_annotations = pd.concat([merged_annotations, relevant_operation_terms], ignore_index=True)
                answer = cl.Message(
                    content="Here are the relevant EDAM Operations: ",
                    elements=[
                        cl.Dataframe(data=relevant_operation_terms, display="inline", name="Dataframe"),
                    ],
                    actions=[cl.Action(name="download_edam_terms", label="EDAM operations in CSV", icon="file", payload=json.loads(relevant_operation_terms.to_json()))]    
                )
                await answer.send()

            # align generated data tags to EDAM data
            if len(data_terms.keys()) > 0:
                relevant_data_terms = await retrieve_classes(json.dumps(data_terms), EDAMTerms.DATA)
                merged_annotations = pd.concat([merged_annotations, relevant_data_terms], ignore_index=True)
                answer = cl.Message(
                    content="Here are the relevant EDAM Data: ",
                    elements=[
                        cl.Dataframe(data=relevant_data_terms, display="inline", name="Dataframe"),
                    ],
                    actions=[cl.Action(name="download_edam_terms", label="EDAM data in CSV", icon="file", payload=json.loads(relevant_data_terms.to_json()))]
                )
                await answer.send()

            # align generated format tags to EDAM formats
            if len(format_terms.keys()) > 0:
                relevant_format_terms = await retrieve_classes(json.dumps(format_terms), EDAMTerms.FORMAT)
                merged_annotations = pd.concat([merged_annotations, relevant_format_terms], ignore_index=True)
                answer = cl.Message(
                    content="Here are the relevant EDAM Formats: ",
                    elements=[
                        cl.Dataframe(data=relevant_format_terms, display="inline", name="Dataframe"),
                    ],
                    actions=[cl.Action(name="download_edam_terms", label="EDAM formats in CSV", icon="file", payload=json.loads(relevant_format_terms.to_json()))]
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

        elif "V4" in cl.user_session.get("chat_profile"):
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
                edam_topics.sort_values(by=["Similarity"], ascending=False, inplace=True)
        
            answer = cl.Message(
                content="Here are the relevant EDAM Topics: ",
                elements=[
                    cl.Dataframe(data=edam_topics, display="inline", name="Dataframe"),
                ],
                actions=[cl.Action(name="download_edam_terms", label="EDAM topics in CSV", icon="file", payload=json.loads(edam_topics.to_json()))]
            )
            await answer.send()

            # retrieve a single EDAM operation for each generated operation term
            edam_operations = pd.DataFrame()
            for k, v in operation_terms.items():
                # top_class = await retrieve_classes(json.dumps(v), EDAMTerms.OPERATION, k=2)
                # # sort by child_terms ascending
                # top_class = top_class.sort_values(by=["child_terms"], ascending=True).head(1)
                top_class = await retrieve_classes(json.dumps(v), EDAMTerms.OPERATION, k=1)
                top_class["Generated term"] = k
                edam_operations = pd.concat([edam_operations, top_class], ignore_index=True)
            if not edam_operations.empty:
                edam_operations.sort_values(by=["Similarity"], ascending=False, inplace=True)

            answer = cl.Message(
                content="Here are the relevant EDAM Operations: ",
                elements=[
                    cl.Dataframe(data=edam_operations, display="inline", name="Dataframe"),
                ],
                actions=[cl.Action(name="download_edam_terms", label="EDAM operations in CSV", icon="file", payload=json.loads(edam_operations.to_json()))]    
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

            answer = cl.Message(
                content="Here are the relevant EDAM Data: ",
                elements=[
                    cl.Dataframe(data=edam_data, display="inline", name="Dataframe"),       
                ],
                actions=[cl.Action(name="download_edam_terms", label="EDAM data in CSV", icon="file", payload=json.loads(edam_data.to_json()))]
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
                edam_formats.sort_values(by=["Similarity"], ascending=False, inplace=True)

            answer = cl.Message(
                content="Here are the relevant EDAM Formats: ",
                elements=[
                    cl.Dataframe(data=edam_formats, display="inline", name="Dataframe"),
                ],
                actions=[cl.Action(name="download_edam_terms", label="EDAM formats in CSV", icon="file", payload=json.loads(edam_formats.to_json()))]
            )
            await answer.send()

            merged_annotations = pd.DataFrame()
            if not edam_topics.empty:
                merged_annotations = pd.concat([merged_annotations, edam_topics], ignore_index=True)
            if not edam_operations.empty:
                merged_annotations = pd.concat([merged_annotations, edam_operations], ignore_index=True)
            if not edam_data.empty:
                merged_annotations = pd.concat([merged_annotations, edam_data], ignore_index=True)
            if not edam_formats.empty:
                merged_annotations = pd.concat([merged_annotations, edam_formats], ignore_index=True)
            
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
