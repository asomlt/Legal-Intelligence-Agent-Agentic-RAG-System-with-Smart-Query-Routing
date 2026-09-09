import os
import json

from concurrent.futures import ThreadPoolExecutor

from sentence_transformers import SentenceTransformer


CHUNKS_FOLDER = "chunks"

EMBEDDED_OUTPUT_FOLDER = "embedded_chunks"

MAX_WORKERS = 4


# local embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def get_all_chunk_files():

    chunk_files = []

    for root, dirs, files in os.walk(CHUNKS_FOLDER):

        for file in files:

            if file.endswith(".json"):

                file_path = os.path.join(root, file)

                chunk_files.append(file_path)

    return chunk_files


def load_chunk_file(file_path):

    with open(file_path, "r", encoding="utf-8") as file:

        return json.load(file)


def generate_embedding(text):

    embedding = model.encode(text)

    return embedding.tolist()


def add_embeddings_to_chunks(chunk_objects):

    updated_chunks = []

    for chunk in chunk_objects:

        chunk["embedding"] = generate_embedding(
            chunk["text"]
        )

        updated_chunks.append(chunk)

    return updated_chunks


def save_embedded_chunks(chunk_objects, source_file):

    os.makedirs(
        EMBEDDED_OUTPUT_FOLDER,
        exist_ok=True
    )

    output_path = os.path.join(
        EMBEDDED_OUTPUT_FOLDER,
        source_file
    )

    with open(output_path, "w", encoding="utf-8") as file:

        json.dump(
            chunk_objects,
            file,
            indent=4,
            ensure_ascii=False
        )


def process_single_chunk_file(file_path):

    try:

        print(f"\nEmbedding: {file_path}")

        chunk_objects = load_chunk_file(file_path)

        embedded_chunks = add_embeddings_to_chunks(
            chunk_objects
        )

        source_file = os.path.basename(file_path)

        save_embedded_chunks(
            embedded_chunks,
            source_file
        )

        print(f"\nSaved: embedded_chunks/{source_file}")

    except Exception as error:

        print(f"\nError Processing: {file_path}")

        print(error)


def process_all_chunk_files():

    chunk_files = get_all_chunk_files()

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        executor.map(
            process_single_chunk_file,
            chunk_files
        )


if __name__ == "__main__":

    process_all_chunk_files()