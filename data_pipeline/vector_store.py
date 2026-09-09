import os
import json
import faiss
import numpy as np
import pickle


EMBEDDED_FOLDER = "embedded_chunks"

FAISS_INDEX_FOLDER = "vector_store"

INDEX_FILE = "legal_index.faiss"

METADATA_FILE = "legal_metadata.pkl"


def get_all_embedded_files():

    embedded_files = []

    for root, dirs, files in os.walk(EMBEDDED_FOLDER):

        for file in files:

            if file.endswith(".json"):

                file_path = os.path.join(root, file)

                embedded_files.append(file_path)

    return embedded_files


def load_embedded_chunks():

    all_chunks = []

    embedded_files = get_all_embedded_files()

    for file_path in embedded_files:

        with open(file_path, "r", encoding="utf-8") as file:

            chunks = json.load(file)

            all_chunks.extend(chunks)

    return all_chunks


def extract_vectors_and_metadata(chunks):

    vectors = []

    metadata_store = []

    for chunk in chunks:

        vectors.append(chunk["embedding"])

        metadata_store.append({

            "chunk_id": chunk["chunk_id"],

            "text": chunk["text"],

            "metadata": chunk["metadata"]
        })

    return vectors, metadata_store


def create_faiss_index(vectors):

    vector_array = np.array(
        vectors,
        dtype="float32"
    )

    dimension = vector_array.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(vector_array)

    return index


def save_faiss_index(index):

    os.makedirs(
        FAISS_INDEX_FOLDER,
        exist_ok=True
    )

    index_path = os.path.join(
        FAISS_INDEX_FOLDER,
        INDEX_FILE
    )

    faiss.write_index(
        index,
        index_path
    )


def save_metadata(metadata_store):

    metadata_path = os.path.join(
        FAISS_INDEX_FOLDER,
        METADATA_FILE
    )

    with open(metadata_path, "wb") as file:

        pickle.dump(
            metadata_store,
            file
        )


def build_vector_database():

    print("\nLoading Embedded Chunks...\n")

    chunks = load_embedded_chunks()

    print(f"Total Chunks: {len(chunks)}")

    print("\nExtracting Vectors...\n")

    vectors, metadata_store = extract_vectors_and_metadata(
        chunks
    )

    print("\nCreating FAISS Index...\n")

    index = create_faiss_index(
        vectors
    )

    print("\nSaving Vector Database...\n")

    save_faiss_index(index)

    save_metadata(metadata_store)

    print("\nFAISS Vector DB Saved Successfully 🔥")


if __name__ == "__main__":

    build_vector_database()