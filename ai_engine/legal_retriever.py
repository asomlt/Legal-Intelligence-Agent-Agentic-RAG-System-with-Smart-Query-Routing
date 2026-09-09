import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from llm_engine import generate_response

FAISS_INDEX_PATH = "vector_store/legal_index.faiss"

METADATA_PATH = "vector_store/legal_metadata.pkl"


print("\nLoading Legal Retrieval Engine...\n")

# embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("\nLegal Retrieval Engine Ready 🔥\n")


def load_faiss_index():

    print("\nLoading FAISS Index...\n")

    index = faiss.read_index(
        FAISS_INDEX_PATH
    )

    return index


def load_metadata():

    print("\nLoading Legal Metadata...\n")

    with open(METADATA_PATH, "rb") as file:

        metadata = pickle.load(file)

    return metadata


# load once at startup
INDEX = load_faiss_index()

METADATA = load_metadata()


def generate_query_embedding(query):

    embedding = model.encode(query)

    return np.array(
        [embedding],
        dtype="float32"
    )


def retrieve_similar_chunks(

    query,
    top_k=3
):

    print("\nGenerating Query Embedding...\n")

    query_vector = generate_query_embedding(
        query
    )

    print("\nSearching Legal Memory...\n")

    distances, indices = INDEX.search(

        query_vector,
        top_k
    )

    retrieved_chunks = []

    for index in indices[0]:

        retrieved_chunks.append(

            METADATA[index]
        )

    return retrieved_chunks


def display_retrieved_chunks(chunks):

    print("\n===== RETRIEVED LEGAL CHUNKS =====\n")

    for index, chunk in enumerate(chunks):

        print(f"\nRESULT {index + 1}")

        print("\nChunk ID:")

        print(chunk["chunk_id"])

        print("\nMetadata:")

        print(chunk["metadata"])

        print("\nText Preview:\n")

        print(chunk["text"][:1000])

        print("\n" + "=" * 80)


def run_legal_pipeline(query):

    print("\nLEGAL RAG PIPELINE ACTIVATED 🔥\n")

    print(f"Legal Query:\n{query}")

    retrieved_chunks = retrieve_similar_chunks(
        query
    )

    display_retrieved_chunks(
        retrieved_chunks
    )
    # combine retrieved chunks into context
    context = "\n\n".join(
        chunk["text"] for chunk in retrieved_chunks
    )

    # generate response using LLM
    response = generate_response(
        user_query=query,
        legal_context=context
    )

    print(f"\nGenerated Legal Response:\n{response}")