import chainlit as cl

from rdflib import Graph
from sentence_transformers import SentenceTransformer
import pandas as pd

import faiss
import numpy as np

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

SYSTEM_PROMPT = """
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

@cl.on_message
async def on_message(msg: cl.Message):
    """Main function to handle when user send a message to the assistant."""

    relevant_classes = await retrieve_classes(msg.content)

    md_classes = [
        f"[{row['Preferred Label']}]({row['Class ID']})"
        for idx, row in relevant_classes.iterrows()
    ]

    md_classes_str = ", ".join(md_classes)

    answer = cl.Message(
        content="Here are the relevant EDAM classes: ",
        elements=[
            cl.Dataframe(data=relevant_classes, display="inline", name="Dataframe"),
            #cl.Text(content=md_classes_str),
        ],
    )
    # for resp in llm.stream(messages):
    #     await answer.stream_token(resp.content)
    #     if resp.usage_metadata:
    #         print(resp.usage_metadata)
    await answer.send()
