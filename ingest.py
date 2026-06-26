# ============================================================
# IMPORTS
# ============================================================

import os
import uuid

# Local embedding model used for semantic search
from sentence_transformers import SentenceTransformer

# Used to read PDF files
from PyPDF2 import PdfReader

# ChromaDB vector database
import chromadb

# Project configuration
from config import (
    CHROMA_PATH,
    COLLECTION_NAME
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================
# This model converts text into vectors (embeddings).
#
# IMPORTANT:
# The same model must be used during both:
# 1. Ingestion
# 2. Retrieval
#
# Otherwise vector similarity search will not work correctly.
# ============================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================
# Reads a PDF file and extracts text from all pages.
#
# Input:
#   PDF file path
#
# Output:
#   Single string containing the entire document text.
# ============================================================

def extract_pdf_text(pdf_path):

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ============================================================
# TEXT CHUNKING
# ============================================================
# Splits large documents into smaller overlapping chunks.
#
# Example:
#
# Chunk 1: characters 0-1000
# Chunk 2: characters 800-1800
# Chunk 3: characters 1600-2600
#
# Overlap preserves context between chunks and
# improves retrieval quality.
# ============================================================

def chunk_text(
    text,
    chunk_size=1000,
    overlap=200
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(
            text[start:end]
        )

        start += chunk_size - overlap

    return chunks


# ============================================================
# EMBEDDING GENERATION
# ============================================================
# Converts text into a numerical vector using the
# local SentenceTransformer model.
#
# Model:
# all-MiniLM-L6-v2
#
# These vectors are stored in ChromaDB and later used
# for semantic similarity search during retrieval.
# ============================================================

def get_embedding(text):

    return embedding_model.encode(
        text
    ).tolist()


# ============================================================
# CHROMADB INITIALIZATION
# ============================================================
# Creates or connects to the local persistent ChromaDB.
#
# Database files are stored inside:
# ./db
# ============================================================

db_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = db_client.get_or_create_collection(
    name=COLLECTION_NAME
)


# ============================================================
# MAIN INGESTION PIPELINE
# ============================================================
# Workflow:
#
# 1. Scan data folder for PDFs
# 2. Read each PDF
# 3. Extract text
# 4. Split text into chunks
# 5. Generate embeddings
# 6. Store chunks in ChromaDB
#
# Any PDF placed in the data folder will automatically
# be processed without changing the code.
# ============================================================

def process_pdf(pdf_path):

    pdf_file = os.path.basename(

        pdf_path

    )

    print(

        f"\nReading PDF: {pdf_file}"

    )

    # --------------------------------------------------------

    # Remove old version of document

    # --------------------------------------------------------

    # If the same PDF is uploaded again,

    # delete all previously stored chunks.

    # --------------------------------------------------------

    existing_docs = collection.get(

        where={

            "source": pdf_file

        }

    )

    if existing_docs["ids"]:

        collection.delete(

            ids=existing_docs["ids"]

        )

        print(

            f"Removed existing chunks for {pdf_file}"

        )

    # --------------------------------------------------------

    # Extract text

    # --------------------------------------------------------

    text = extract_pdf_text(

        pdf_path

    )

    if not text.strip():

        print(

            f"No text extracted from {pdf_file}"

        )

        return 0

    print(

        f"Extracted {len(text)} characters"

    )

    # --------------------------------------------------------

    # Create chunks

    # --------------------------------------------------------

    chunks = chunk_text(text)

    print(

        f"Created {len(chunks)} chunks"

    )

    # --------------------------------------------------------

    # Store chunks

    # --------------------------------------------------------

    stored_count = 0

    for idx, chunk in enumerate(chunks):

        # Skip empty chunks

        if not chunk.strip():

            print(

                f"Skipped empty chunk {idx + 1}"

            )

            continue

        embedding = get_embedding(

            chunk

        )

        collection.add(

            ids=[

                str(uuid.uuid4())

            ],

            documents=[

                chunk

            ],

            embeddings=[

                embedding

            ],

            metadatas=[

                {

                    "source": pdf_file,

                    "document_name": pdf_file,

                    "chunk_number": idx,

                    "chunk_length": len(chunk)

                }

            ]

        )

        stored_count += 1

        print(

            f"Indexed chunk {idx + 1}/{len(chunks)}"

        )

    print(

        f"Finished indexing {pdf_file}"

    )

    return stored_count


if __name__ == "__main__":

    # --------------------------------------------------------
    # Locate all PDF files
    # --------------------------------------------------------

    DATA_FOLDER = "data"

    pdf_files = [
        file
        for file in os.listdir(DATA_FOLDER)
        if file.lower().endswith(".pdf")
    ]

    # --------------------------------------------------------
    # Ensure at least one PDF exists
    # --------------------------------------------------------

    if not pdf_files:

        print(
            "No PDF files found in data folder."
        )

        exit()

    print(
        f"Found {len(pdf_files)} PDF(s) to process."
    )

    # --------------------------------------------------------
    # Track total stored chunks
    # --------------------------------------------------------

    stored_count = 0

    # --------------------------------------------------------
    # Process every PDF
    # --------------------------------------------------------

    for pdf_file in pdf_files:

        pdf_path = os.path.join(
            DATA_FOLDER,
            pdf_file
        )

    stored_count += process_pdf(
        pdf_path
    )
    # --------------------------------------------------------
    # Final Summary
    # --------------------------------------------------------

    print(
        "\nAll PDFs indexed successfully."
    )

    print(
        f"Stored {stored_count} chunks in ChromaDB."
    )