import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer


FAISS_INDEX_PATH = "vector_store/legal_index.faiss"

METADATA_PATH = "vector_store/legal_metadata.pkl"


# same embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def load_faiss_index():

    index = faiss.read_index(
        FAISS_INDEX_PATH
    )

    return index


def load_metadata():

    with open(METADATA_PATH, "rb") as file:

        metadata = pickle.load(file)

    return metadata


def generate_query_embedding(query):

    embedding = model.encode(query)

    return np.array(
        [embedding],
        dtype="float32"
    )


def search_similar_chunks(

    query,
    index,
    metadata,
    top_k=3
):

    query_vector = generate_query_embedding(
        query
    )

    distances, indices = index.search(
        query_vector,
        top_k
    )

    results = []

    for i in indices[0]:

        results.append(metadata[i])

    return results


def display_results(results):

    print("\n===== RETRIEVED CHUNKS =====\n")

    for index, result in enumerate(results):

        print(f"\nRESULT {index + 1}")

        print("\nChunk ID:")

        print(result["chunk_id"])

        print("\nMetadata:")

        print(result["metadata"])

        print("\nText Preview:\n")

        print(result["text"][:1000])

        print("\n" + "=" * 80)


def main():

    print("\nLegal Intelligence Retrieval Test 🔥\n")

    query = input("Enter Query: ")

    index = load_faiss_index()

    metadata = load_metadata()

    results = search_similar_chunks(

        query,
        index,
        metadata
    )

    display_results(results)


if __name__ == "__main__":

    main()