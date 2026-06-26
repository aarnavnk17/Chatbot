# ============================================================
# IMPORTS
# ============================================================


import os

import streamlit as st

import chromadb
from config import CHROMA_PATH, COLLECTION_NAME

db_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = db_client.get_collection(
    COLLECTION_NAME
)

st.sidebar.write(
    f"Chunks in DB: {collection.count()}"
)

from rag_engine import (
    ask_question
)

from ingest import (
    process_pdf
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================
# Configures the Streamlit page.
# ============================================================

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# APPLICATION HEADER
# ============================================================
# Main title and description shown at the top of the app.
# ============================================================

st.title(
    "🤖 RAG Chatbot"
)

st.caption(
    "Ask questions about uploaded documents or general knowledge topics."
)


# ============================================================
# SIDEBAR - PDF UPLOAD
# ============================================================
# Users can upload PDFs directly from the UI.
#
# Uploaded PDFs are:
# 1. Saved into the data folder
# 2. Indexed into ChromaDB
# 3. Immediately available for querying
# ============================================================

st.sidebar.header(
    "📄 Document Upload"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


# ============================================================
# PROCESS NEW PDF UPLOAD
# ============================================================

if uploaded_file:

    os.makedirs(
        "data",
        exist_ok=True
    )

    save_path = os.path.join(
        "data",
        uploaded_file.name
    )

    # --------------------------------------------------------
    # Save uploaded file
    # --------------------------------------------------------

    with open(
        save_path,
        "wb"
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )

    # --------------------------------------------------------
    # Index uploaded document
    # --------------------------------------------------------

    try:

        chunk_count = process_pdf(
            save_path
        )

        st.sidebar.success(
            f"Indexed successfully ({chunk_count} chunks)"
        )

    except Exception as error:

        st.sidebar.error(
            f"Indexing failed: {error}"
        )


# ============================================================
# SESSION STATE
# ============================================================
# Stores chat history across reruns.
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================
# Renders all previous messages.
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# USER INPUT
# ============================================================
# Chat input box displayed at the bottom.
# ============================================================

question = st.chat_input(
    "Ask a question..."
)


# ============================================================
# PROCESS USER QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # Store User Message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )

    # --------------------------------------------------------
    # Generate Response
    # --------------------------------------------------------

    with st.spinner(
        "Searching knowledge base..."
    ):

        answer, sources = ask_question(
            question
        )

    # --------------------------------------------------------
    # Build Response Text
    # --------------------------------------------------------

    response_text = answer

    if sources:

        response_text += "\n\n### Sources\n"

        unique_sources = set()

        for source in sources:

            if source not in unique_sources:

                unique_sources.add(
                    source
                )

                response_text += (
                    f"- {source}\n"
                )

    # --------------------------------------------------------
    # Display Assistant Message
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        st.markdown(
            response_text
        )

    # --------------------------------------------------------
    # Save Assistant Message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text
        }
    )