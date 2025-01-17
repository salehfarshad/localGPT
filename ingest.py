import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import click
import torch
from langchain.docstore.document import Document
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import Chroma
from utils import get_embeddings

from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)


def file_log(logentry):
    file1 = open("file_ingest.log", "a")
    file1.write(logentry + "\n")
    file1.close()
    print(logentry + "\n")


def clean_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespaces
    text = ' '.join(text.split())
    return text


from bidi.algorithm import get_display
from langdetect import detect


def preprocess_text(text):
    # Detect the language of the text
    language = detect(text)

    # Handle RTL/LTR direction based on the detected language
    if language == 'fa' or language == 'fa-en':  # Persian language
        # Reverse the text to RTL direction
        text = get_display(text)
    else:
        # Leave the text as-is (LTR direction for non-Persian languages)
        pass

    return text


def load_single_document(file_path: str) -> Document:
    # Loads a single document from a
    # file path
    try:
        file_extension = os.path.splitext(file_path)[1]
        loader_class = DOCUMENT_MAP.get(file_extension)
        if loader_class:
            file_log(file_path + " loaded.")
            # if file_extension == ".pdf":
            # text = extract_text_from_pdf(file_path)
            # sentences = split_sentences(text)
            # return Document(page_content=text, metadata={"source": file_path})
            loader = loader_class(file_path)
            content = loader.load()[0].page_content
            # content = clean_text(content)
            return Document(page_content=preprocess_text(content), metadata={"source": file_path})
            # return loader.load()[0]
        else:
            file_log(file_path + " document type is undefined.")
            raise ValueError("Document type is undefined")
    except Exception as ex:
        file_log("%s loading error: \n%s" % (file_path, ex))
        return None


# Rest of the code remains unchanged
def load_document_batch(filepaths):
    logging.info("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        # collect data
        if futures is None:
            file_log(name + " failed to submit")
            return None
        else:
            data_list = [future.result() for future in futures]
            # return data and file paths
            return (data_list, filepaths)


def load_documents(source_dir: str) -> list[Document]:
    # Loads all documents from the source documents directory, including nested folders
    paths = []
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            print("Importing: " + file_name)
            file_extension = os.path.splitext(file_name)[1]
            source_file_path = os.path.join(root, file_name)
            if file_extension in DOCUMENT_MAP.keys():
                paths.append(source_file_path)

    # Have at least one worker and at most INGEST_THREADS workers
    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = round(len(paths) / n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        for i in range(0, len(paths), chunksize):
            # select a chunk of filenames
            filepaths = paths[i: (i + chunksize)]
            # submit the task
            try:
                future = executor.submit(load_document_batch, filepaths)
            except Exception as ex:
                file_log("executor task failed: %s" % (ex))
                future = None
            if future is not None:
                futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            try:
                contents, _ = future.result()
                docs.extend(contents)
            except Exception as ex:
                file_log("Exception: %s" % (ex))

    return docs


import re


def reverse_sentences(text):
    # Split the text into sentences

    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Reverse the characters in each sentence
    reversed_sentences = []
    for sentence in sentences:
        reversed_sentence = ' '.join(word[::-1] for word in sentence.split())
        reversed_sentences.append(reversed_sentence)

    # Join the reversed sentences back into a single string
    reversed_text = ' '.join(reversed_sentences)

    return reversed_text


def reverse_text(text):
    # Step 1: Split text into words
    words = text.split()

    # Step 2: Reverse the characters within each word
    # reversed_words = [''.join(word[::-1]) for word in words]

    # Step 3: Reverse the order of words within each line
    reversed_lines = reversed(words)

    # Step 4: Join the lines together in reverse order
    reversed_text = '\n'.join(reversed_lines)

    return reversed_text


from langdetect import detect


def split_documents(documents: list[Document]) -> tuple[list[Document], list[Document]]:
    # Splits documents for correct Text Splitter
    text_docs, persian_docs, python_docs = [], [], []
    for doc in documents:
        # Extract the text from the 'Document' object
        # doc_text = doc.page_content # Replace 'text' with the actual method or property name
        # print(doc_text)
        if doc is not None:
            file_extension = os.path.splitext(doc.metadata["source"])[1]
            if file_extension == ".py":
                python_docs.append(doc)
            else:
                text_docs.append(doc)

    return text_docs, python_docs


import langdetect


@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
def main(device_type):
    # Load documents and split in chunks
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    text_documents, python_documents = split_documents(documents)
    persian_separators = ["\n\n", "؟", "!","."]
    # avg_doc_length = sum(len(text) for text in text_documents[0].page_content) / len(text_documents)
    chunk_size = 800  # int(avg_doc_length / 20) if avg_doc_length > 10000 else 500  # Adjust this value as needed
    chunk_overlap = 100  # int(chunk_size * 0.2)  # Adjust this value as needed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap
                                                   )
    # Open the file in write mode ('w')
    # with open('output.txt', 'w', encoding='utf-8') as file:
    #     Write the text variable to the file
    # file.write(text_documents[0].page_content)

    # text_documents[0].page_content = reverse_text(text_documents[0].page_content)
    texts = text_splitter.split_documents(text_documents)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=880, chunk_overlap=200)

    texts.extend(python_splitter.split_documents(python_documents))
    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")

    """
    (1) Chooses an appropriate langchain library based on the enbedding model name.  Matching code is contained within fun_localGPT.py.

    (2) Provides additional arguments for instructor and BGE models to improve results, pursuant to the instructions contained on
    their respective huggingface repository, project page or github repository.
    """
    embeddings = get_embeddings(device_type)

    logging.info(f"Loaded embeddings from {EMBEDDING_MODEL_NAME}")

    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
