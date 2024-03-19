# Required Libraries
import os
import glob
# import fitz  # PyMuPDF for PDF handling
import pandas as pd
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer

from langchain.docstore.document import Document
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader
from langchain.document_loaders import UnstructuredFileLoader, UnstructuredMarkdownLoader
from langchain.document_loaders import UnstructuredHTMLLoader

DOCUMENT_MAP = {
    ".html": UnstructuredHTMLLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".py": TextLoader,
    # ".pdf": PDFMinerLoader,
    ".pdf": UnstructuredFileLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}


# Function to load and read documents
def load_single_document(file_path: str):  # -> Document:
    # Loads a single document from a
    # file path
    try:
        documents = []
        file_extension = os.path.splitext(file_path)[1]
        loader_class = DOCUMENT_MAP.get(file_extension)
        if loader_class:
            loader = loader_class(file_path)
            # content = loader.load()[0].page_content
            documents.append(loader.load()[0].page_content)
            return documents
        else:
            raise ValueError("Document type is undefined")
    except Exception as ex:
        return None


# Function to embed and store document contents in a vector store
def embed_and_store(documents, tokenizer):
    document_embeddings = []
    # max_length = 512  # Set a maximum length for padding/truncation
    for doc in documents:
        encoding = tokenizer(doc, padding=True, truncation=True, return_tensors="pt")
        document_embeddings.append(encoding)
    return document_embeddings


# Main function
def main():
    # Define directory containing documents
    directory = r"./SOURCE_DOCUMENTS\es.docx"

    # Step 1: Load and read documents
    documents = load_single_document(directory)

    # Step 2: Initialize HooshvareLab's BERT model for Persian embedding and tokenization
    persian_tokenizer = pipeline('feature-extraction', model='HooshvareLab/bert-fa-base-uncased')

    # Step 3: Embed and store document contents in a vector store
    document_embeddings = embed_and_store(documents, persian_tokenizer.tokenizer)

    # Step 4: Initialize RagTokenForGeneration pipeline for LLM
    llm_pipeline = pipeline("text-generation", model="aidal/Persian-Mistral-7B", tokenizer="aidal/Persian-Mistral-7B",
                            document_embeddings=document_embeddings)

    # Step 5: Retrieve relevant documents based on user query
    query = input("Enter your query: ")
    relevant_documents = llm_pipeline.retrieve_documents(query, documents)

    # Step 6: Generate response using retrieved context and user query
    response = llm_pipeline.generate_response(query, relevant_documents)

    # Step 7: Display response
    print("Response:")
    print(response)

if __name__ == "__main__":
    main()

    # Step 4: Retrieve relevant documents based on user query
    query = input("Enter your query: ")
    relevant_documents = retrieve_documents(query, documents, rag_retriever, tokenizer)

    # Step 5: Generate response using retrieved context and user query
    response_prompt = "retrieve documents based on query: {}.".format(query)
    generated_response = llm_pipeline(response_prompt)[0]['generated_text']

    # Step 6: Display response
    print("Response:")
    print(generated_response)

if __name__ == "__main__":
    main()
