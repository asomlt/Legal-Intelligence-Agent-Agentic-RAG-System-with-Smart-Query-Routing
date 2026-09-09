import os
import re
import json

from concurrent.futures import ThreadPoolExecutor

from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_FOLDER = "output"
CHUNK_OUTPUT_FOLDER = "chunks"

MAX_WORKERS = 4


def get_all_text_files():

    text_files = []

    for root, dirs, files in os.walk(INPUT_FOLDER):

        for file in files:

            if file.endswith(".txt"):

                file_path = os.path.join(root, file)

                text_files.append(file_path)

    return text_files


def read_text_file(file_path):

    with open(file_path, "r", encoding="utf-8") as file:

        return file.read()


def split_by_legal_sections(text):

    pattern = r"(Article\s+\d+|Section\s+\d+|CHAPTER\s+[A-Z]+)"

    sections = re.split(pattern, text)

    legal_chunks = []

    current_chunk = ""

    for section in sections:

        if section.strip() == "":
            continue

        if re.match(pattern, section):

            if current_chunk:

                legal_chunks.append(current_chunk.strip())

            current_chunk = section

        else:

            current_chunk += " " + section

    if current_chunk:

        legal_chunks.append(current_chunk.strip())

    return legal_chunks


def recursive_chunking(legal_chunks):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,
        chunk_overlap=100
    )

    final_chunks = []

    for chunk in legal_chunks:

        if len(chunk) > 1500:

            smaller_chunks = splitter.split_text(chunk)

            final_chunks.extend(smaller_chunks)

        else:

            final_chunks.append(chunk)

    return final_chunks


def extract_metadata(chunk_text, source_file):

    article_match = re.search(
        r"(Article|Art\.?)\s+(\d+)",
        chunk_text,
        re.IGNORECASE
    )

    section_match = re.search(
        r"(Section\s+(\d+[A-Z]?))|(\b\d+[A-Z]?\.)",
        chunk_text,
        re.IGNORECASE
    )

    article_number = None
    section_number = None

    if article_match:

        article_number = article_match.group(2)

    if section_match:

        if section_match.group(2):

            section_number = section_match.group(2)

        else:

            section_number = section_match.group(3).replace(".", "")

    metadata = {

        "source_file": source_file,

        "document_type": source_file.replace(".txt", "").upper(),

        "article_number": article_number,

        "section_number": section_number
    }

    return metadata



def generate_chunk_objects(chunks, source_file):

    chunk_objects = []

    last_article = None
    last_section = None

    for index, chunk_text in enumerate(chunks):

        metadata = extract_metadata(chunk_text, source_file)

        # inherit previous article
        if metadata["article_number"]:

            last_article = metadata["article_number"]

        else:

            metadata["article_number"] = last_article

        # inherit previous section
        if metadata["section_number"]:

            last_section = metadata["section_number"]

        else:

            metadata["section_number"] = last_section

        chunk_data = {

            "chunk_id": f"{source_file}_{index + 1}",

            "text": chunk_text,

            "metadata": metadata
        }

        chunk_objects.append(chunk_data)

    return chunk_objects


    chunk_objects = []

    for index, chunk_text in enumerate(chunks):

        metadata = extract_metadata(chunk_text, source_file)

        chunk_data = {

            "chunk_id": f"{source_file}_{index + 1}",

            "text": chunk_text,

            "metadata": metadata
        }

        chunk_objects.append(chunk_data)

    return chunk_objects


def save_chunks(chunk_objects, source_file):

    os.makedirs(CHUNK_OUTPUT_FOLDER, exist_ok=True)

    output_file = source_file.replace(".txt", ".json")

    output_path = os.path.join(CHUNK_OUTPUT_FOLDER, output_file)

    with open(output_path, "w", encoding="utf-8") as file:

        json.dump(chunk_objects, file, indent=4, ensure_ascii=False)


def process_single_text_file(file_path):

    try:

        print(f"\nChunking: {file_path}")

        text = read_text_file(file_path)

        legal_chunks = split_by_legal_sections(text)

        final_chunks = recursive_chunking(legal_chunks)

        source_file = os.path.basename(file_path)

        chunk_objects = generate_chunk_objects(final_chunks, source_file)

        print(f"\nTotal Chunks: {len(chunk_objects)}")

        print("\n===== CHUNK PREVIEW =====\n")

        # print(chunk_objects[0]["text"][:1000])

        save_chunks(chunk_objects, source_file)

        # print(f"\nSaved: chunks/{source_file.replace('.txt', '.json')}")

    except Exception as error:

        print(f"\nError Processing: {file_path}")

        print(error)


def process_all_text_files():

    text_files = get_all_text_files()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        executor.map(process_single_text_file, text_files)


if __name__ == "__main__":

    process_all_text_files()