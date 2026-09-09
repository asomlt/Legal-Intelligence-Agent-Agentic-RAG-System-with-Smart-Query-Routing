import fitz
import os

from concurrent.futures import ThreadPoolExecutor


DATA_FOLDER = "files"
OUTPUT_FOLDER = "output/raw_text"

MAX_WORKERS = 4


def get_all_pdf_files():

    pdf_files = []

    for root, dirs, files in os.walk(DATA_FOLDER):

        for file in files:

            if file.endswith(".pdf"):

                pdf_path = os.path.join(root, file)

                pdf_files.append(pdf_path)

    return pdf_files


def extract_text_from_pdf(pdf_path):

    document = fitz.open(pdf_path)

    full_text = ""

    for page_number in range(len(document)):

        page = document.load_page(page_number)

        page_text = page.get_text()

        full_text += f"\n\n--- PAGE {page_number + 1} ---\n\n"

        full_text += str(page_text)

    return full_text


def generate_output_path(pdf_path):

    relative_path = os.path.relpath(pdf_path, DATA_FOLDER)

    output_file = relative_path.replace(".pdf", ".txt")

    output_path = os.path.join(OUTPUT_FOLDER, output_file)

    return output_path


def save_extracted_text(text, output_path):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:

        file.write(text)


def process_single_pdf(pdf_path):

    try:

        print(f"\nProcessing: {pdf_path}")

        extracted_text = extract_text_from_pdf(pdf_path)

        # print("\n===== TEXT PREVIEW =====\n")

        # print(extracted_text[:1500])

        output_path = generate_output_path(pdf_path)

        save_extracted_text(extracted_text, output_path)

        # print(f"\nSaved: {output_path}")

    except Exception as error:

        print(f"\nError processing {pdf_path}")

        print(error)


def process_all_pdfs():

    pdf_files = get_all_pdf_files()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        executor.map(process_single_pdf, pdf_files)


if __name__ == "__main__":

    process_all_pdfs()