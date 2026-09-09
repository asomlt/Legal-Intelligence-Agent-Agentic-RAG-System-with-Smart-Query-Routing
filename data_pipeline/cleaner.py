import os
import re

from concurrent.futures import ThreadPoolExecutor


INPUT_FOLDER = "output"
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


def clean_legal_text(text):

    # Remove weird unicode/symbols
    text = re.sub(r"[^\w\s\.\,\:\;\-\(\)\[\]\/]", " ", text)

    # Remove extra spaces/newlines
    text = re.sub(r"\s+", " ", text)

    # Restore page markers
    text = text.replace("--- PAGE", "\n\n--- PAGE")

    return text.strip()


def save_clean_text(cleaned_text, file_path):

    with open(file_path, "w", encoding="utf-8") as file:

        file.write(cleaned_text)


def process_single_file(file_path):

    try:

        print(f"\nCleaning: {file_path}")

        raw_text = read_text_file(file_path)

        cleaned_text = clean_legal_text(raw_text)

        print("\n===== CLEAN TEXT PREVIEW =====\n")

        print(cleaned_text[:1500])

        # overwrite same file
        save_clean_text(cleaned_text, file_path)

        print(f"\nUpdated: {file_path}")

    except Exception as error:

        print(f"\nError cleaning: {file_path}")

        print(error)


def process_all_text_files():

    text_files = get_all_text_files()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        executor.map(process_single_file, text_files)


if __name__ == "__main__":

    process_all_text_files()