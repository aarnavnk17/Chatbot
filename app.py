# ============================================================
# IMPORTS
# ============================================================

import os
import uuid                          # NEW: for session ID generation
from datetime import datetime

import streamlit as st

import chromadb
from config import CHROMA_PATH, COLLECTION_NAME

db_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = db_client.get_collection(
    COLLECTION_NAME
)

import rag_engine

from ingest import process_pdf

# --------------------------------------------------------
# NEW: session_memory and session_store for the clear
# session feature. Imported here so app.py can call
# clear() and clear_session() on user request.
# --------------------------------------------------------

import session_memory   # in-memory rolling buffer
import session_store    # ChromaDB session persistence


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
# SESSION STATE INITIALISATION
# ============================================================
# Streamlit reruns the whole script on every interaction.
# st.session_state persists values across reruns within the
# same browser tab.
#
# Keys managed here:
#   messages    — chat display history (list of role/content)
#   session_id  — NEW: unique UUID for this chat session,
#                 used to isolate memory and vector storage
#   turn_count  — NEW: tracks how many Q&A pairs have happened
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    # --------------------------------------------------------
    # NEW: Generate a stable UUID for this browser session.
    # This is created once per tab open and never changes
    # unless the user manually clears the session.
    # --------------------------------------------------------
    st.session_state.session_id = str(uuid.uuid4())

if "turn_count" not in st.session_state:
    # --------------------------------------------------------
    # NEW: Counter for how many turns have happened this
    # session. Used for display in the sidebar.
    # --------------------------------------------------------
    st.session_state.turn_count = 0

if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = st.session_state.session_id

if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []


def rebuild_session_memory(session_id):
    session_memory.clear(session_id)
    turns = session_store.get_session_turns(session_id)
    for turn in turns:
        user_msg = turn.get("user_msg", "")
        assistant_msg = turn.get("assistant_msg", "")
        if user_msg or assistant_msg:
            session_memory.add_turn(session_id, user_msg, assistant_msg)


def load_session_into_state(session_id):
    """Restore a stored session into the active Streamlit state."""
    turns = session_store.get_session_turns(session_id)
    messages = []

    rebuild_session_memory(session_id)

    for turn in turns:
        user_msg = turn.get("user_msg", "")
        assistant_msg = turn.get("assistant_display_msg", turn.get("assistant_msg", ""))
        assistant_sources = turn.get("assistant_sources", [])

        if user_msg:
            messages.append({"role": "user", "content": user_msg})

        if assistant_msg:
            messages.append({
                "role": "assistant",
                "content": assistant_msg,
                "sources": assistant_sources
            })

    st.session_state.messages = messages
    st.session_state.session_id = session_id
    st.session_state.active_session_id = session_id
    st.session_state.turn_count = len(turns)


def get_session_summary(session_id):
    summary_fn = getattr(session_store, "get_session_summary", None)
    if callable(summary_fn):
        summary = summary_fn(session_id)
    else:
        sessions = session_store.list_recent_sessions(limit=5)
        summary = next(
            (session for session in sessions if session["session_id"] == session_id),
            None
        )

    if not summary:
        summary = {
            "session_id": session_id,
            "turn_count": 0,
            "last_timestamp": 0,
            "first_timestamp": 0,
        }

    summary.setdefault("title", "Untitled Session")
    summary.setdefault("turn_count", 0)
    summary.setdefault("last_timestamp", 0)
    summary.setdefault("first_timestamp", 0)
    return summary


def get_memory_count():
    memory_count_fn = getattr(rag_engine, "get_memory_count", None)
    if callable(memory_count_fn):
        return memory_count_fn()
    return 0


def get_session_count():
    session_count_fn = getattr(session_store, "get_session_count", None)
    if callable(session_count_fn):
        return session_count_fn()

    recent_sessions = session_store.list_recent_sessions(limit=100)
    return len(recent_sessions)


def update_turn_display(session_id, turn_index, assistant_display_msg, assistant_sources):
    update_turn_display_fn = getattr(session_store, "update_turn_display", None)
    if callable(update_turn_display_fn):
        return update_turn_display_fn(
            session_id,
            turn_index,
            assistant_display_msg,
            assistant_sources
        )
    return False


def rename_session(session_id, new_title):
    rename_session_fn = getattr(session_store, "rename_session", None)
    if callable(rename_session_fn):
        return rename_session_fn(session_id, new_title)
    return False


def format_session_time(timestamp):
    if not timestamp:
        return "Just now"
    return datetime.fromtimestamp(timestamp).strftime("%b %d, %I:%M %p")


def get_uploaded_documents():
    if not os.path.isdir("data"):
        return []
    return sorted(
        file_name for file_name in os.listdir("data")
        if file_name.lower().endswith(".pdf")
    )


def render_sources(sources):
    if not sources:
        return

    chips = "".join(
        f'<span class="source-chip">PDF {source}</span>'
        for source in sources
    )
    st.markdown(
        f'<div class="source-chip-row">{chips}</div>',
        unsafe_allow_html=True
    )
    with st.expander("Source details"):
        for source in sources:
            st.markdown(f"- {source}")


def render_message(message):
    role = message["role"]
    bubble_class = "chat-bubble user-bubble" if role == "user" else "chat-bubble assistant-bubble"
    columns = st.columns([1, 5]) if role == "assistant" else st.columns([5, 1])
    target_column = columns[1] if role == "user" else columns[0]

    with target_column:
        st.markdown(f'<div class="{bubble_class}">', unsafe_allow_html=True)
        st.markdown(message["content"])
        if role == "assistant":
            render_sources(message.get("sources", []))
        st.markdown('</div>', unsafe_allow_html=True)


def start_new_chat():
    old_session_id = st.session_state.session_id
    session_memory.clear(old_session_id)
    st.session_state.messages = []
    st.session_state.turn_count = 0
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.active_session_id = st.session_state.session_id
    st.rerun()


def activate_session(session_id):
    if session_id and session_id != st.session_state.session_id:
        load_session_into_state(session_id)
        st.rerun()


def inject_brand_styles():
    st.markdown(
        """
        <style>
        :root {
            --nallas-primary: #2A3A86;
            --nallas-secondary: #4C63D2;
            --nallas-accent: #756CA1;
            --nallas-bg: #111827;
            --nallas-sidebar: #1A1F2E;
            --nallas-card: #242B3D;
            --nallas-text: #F8FAFC;
            --nallas-border: #394867;
        }

        .stApp {
            background: radial-gradient(circle at top, rgba(76,99,210,0.16), transparent 34%), var(--nallas-bg);
            color: var(--nallas-text);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1A1F2E 0%, #151B28 100%);
            border-right: 1px solid var(--nallas-border);
        }

        [data-testid="stSidebar"] * {
            color: var(--nallas-text);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        h1, h2, h3, h4, h5, h6, p, div, span, label {
            color: var(--nallas-text);
        }

        .nallas-brand {
            font-size: 0.85rem;
            letter-spacing: 0.24em;
            text-transform: uppercase;
            color: rgba(248,250,252,0.68);
            margin-bottom: 0.35rem;
        }

        .nallas-card {
            background: rgba(36,43,61,0.92);
            border: 1px solid rgba(57,72,103,0.95);
            border-radius: 18px;
            padding: 1rem 1.05rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.22);
        }

        .storage-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.8rem;
            margin-top: 0.8rem;
        }

        .nallas-metric {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--nallas-text);
        }

        .nallas-muted {
            color: rgba(248,250,252,0.68);
            font-size: 0.82rem;
        }

        .session-item {
            border: 1px solid rgba(57,72,103,0.55);
            background: rgba(36,43,61,0.55);
            border-radius: 14px;
            padding: 0.7rem 0.85rem;
            margin-bottom: 0.25rem;
            box-shadow: 0 8px 20px rgba(0,0,0,0.12);
            animation: fadeInUp 220ms ease-out;
        }

        .session-item.active {
            border-color: rgba(76,99,210,0.95);
            background: rgba(42,58,134,0.28);
        }

        .session-item-title {
            font-weight: 600;
            color: var(--nallas-text);
            margin-bottom: 0.2rem;
        }

        .session-item-meta {
            font-size: 0.76rem;
            color: rgba(248,250,252,0.64);
            line-height: 1.35;
        }

        .chat-bubble {
            padding: 1rem 1.1rem 0.9rem 1.1rem;
            border-radius: 20px;
            border: 1px solid rgba(57,72,103,0.88);
            box-shadow: 0 12px 28px rgba(0,0,0,0.18);
            margin-bottom: 0.85rem;
            animation: fadeInUp 240ms ease-out;
        }

        .assistant-bubble {
            background: linear-gradient(180deg, rgba(36,43,61,0.96), rgba(29,35,49,0.96));
        }

        .user-bubble {
            background: linear-gradient(135deg, rgba(42,58,134,0.94), rgba(76,99,210,0.94));
        }

        .chat-bubble [data-testid="stMarkdownContainer"] p {
            line-height: 1.65;
            margin-bottom: 0.4rem;
        }

        .stChatInput input, textarea {
            background: #242B3D !important;
            color: #F8FAFC !important;
            border: 1px solid #394867 !important;
            border-radius: 16px !important;
        }

        .stButton > button {
            background: linear-gradient(135deg, var(--nallas-primary), var(--nallas-secondary)) !important;
            color: white !important;
            border: 0 !important;
            border-radius: 14px !important;
            box-shadow: 0 8px 18px rgba(76,99,210,0.22);
            transition: transform 160ms ease, filter 160ms ease;
        }

        .stButton > button:hover {
            filter: brightness(1.08);
            transform: translateY(-1px);
        }

        .stFileUploader {
            background: rgba(36,43,61,0.7);
            border: 1px solid var(--nallas-border);
            border-radius: 16px;
            padding: 0.5rem;
        }

        .source-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.65rem;
        }

        .source-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background: rgba(17,24,39,0.46);
            border: 1px solid rgba(117,108,161,0.55);
            color: var(--nallas-text);
            border-radius: 999px;
            padding: 0.35rem 0.7rem;
            font-size: 0.8rem;
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(8px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        hr {
            border-color: rgba(57,72,103,0.7);
        }
        </style>
        """,
        unsafe_allow_html=True
    )


inject_brand_styles()


# ============================================================
# APPLICATION HEADER
# ============================================================

st.markdown('<div class="nallas-brand">Nallas Technologies</div>', unsafe_allow_html=True)
st.title("NALLAS AI")

st.caption(
    "Ask questions about uploaded documents or general knowledge topics."
)


# ============================================================
# SIDEBAR — SESSION INFO
# ============================================================
# NEW: Displays the current session ID and turn count so
# users can see that session memory is active.
# ============================================================

st.sidebar.markdown("## NALLAS AI")
if st.sidebar.button("+ New Chat", use_container_width=True):
    start_new_chat()

st.sidebar.markdown("---")

st.sidebar.markdown("### Recent Sessions")

current_summary = get_session_summary(st.session_state.session_id)
uploaded_documents = get_uploaded_documents()
st.session_state.uploaded_documents = uploaded_documents

recent_sessions = session_store.list_recent_sessions(limit=5)

if recent_sessions:
    current_session_id = st.session_state.session_id
    if current_summary["session_id"] not in [s["session_id"] for s in recent_sessions]:
        recent_sessions = [current_summary] + recent_sessions[:4]

    for session in recent_sessions[:5]:
        active = session["session_id"] == current_session_id
        item_class = "session-item active" if active else "session-item"
        st.sidebar.markdown(
            f"""
            <div class="{item_class}">
                <div class="session-item-title">{session.get('title', 'Untitled Session')}</div>
                <div class="session-item-meta">
                    {session.get('turn_count', 0)} turns · {format_session_time(session.get('last_timestamp', 0))}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.sidebar.button(
            "Continue",
            key=f"open_session_{session['session_id']}",
            use_container_width=True
        ):
            activate_session(session["session_id"])
else:
    st.sidebar.caption("No saved sessions yet.")

st.sidebar.markdown("---")


# ============================================================
# SIDEBAR — CLEAR SESSION BUTTON
# ============================================================
# NEW: Clicking this button starts a brand new active session
# without deleting any prior session data from ChromaDB.
# ============================================================

st.sidebar.markdown("### Storage Info")
if st.sidebar.button("Refresh storage", use_container_width=True):
    st.rerun()

st.sidebar.markdown(
    f"""
    <div class="nallas-card">
        <div class="nallas-muted">Session title</div>
        <div class="nallas-metric">{current_summary.get('title', 'Untitled Session')}</div>
        <div class="nallas-muted" style="margin-top:0.35rem;">{current_summary.get('session_id', '')[:8]}...</div>
        <div class="nallas-muted" style="margin-top:0.65rem;">Turns: {st.session_state.turn_count}</div>
        <div class="nallas-muted">Last updated: {format_session_time(current_summary.get('last_timestamp', 0))}</div>
        <div class="nallas-muted">Memory status: Active</div>
    </div>
    <div class="storage-grid">
        <div class="nallas-card">
            <div class="nallas-muted">Knowledge Base</div>
            <div class="nallas-metric">{collection.count()}</div>
        </div>
        <div class="nallas-card">
            <div class="nallas-muted">Long-Term Memory</div>
            <div class="nallas-metric">{get_memory_count()}</div>
        </div>
        <div class="nallas-card">
            <div class="nallas-muted">Session Store</div>
            <div class="nallas-metric">{get_session_count()}</div>
        </div>
        <div class="nallas-card">
            <div class="nallas-muted">Current Session</div>
            <div class="nallas-metric">{st.session_state.turn_count}</div>
        </div>
        <div class="nallas-card">
            <div class="nallas-muted">Embedding Model</div>
            <div class="nallas-metric">all-MiniLM-L6-v2</div>
        </div>
        <div class="nallas-card">
            <div class="nallas-muted">Uploaded Documents</div>
            <div class="nallas-metric">{len(uploaded_documents)}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

edited_title = st.sidebar.text_input(
    "Rename current session",
    value=current_summary.get("title", "Untitled Session"),
    key=f"rename_title_{st.session_state.session_id}"
)

if st.sidebar.button("Save session title", use_container_width=True):
    if rename_session(st.session_state.session_id, edited_title):
        st.rerun()

st.sidebar.markdown("---")

# ============================================================
# SIDEBAR — PDF UPLOAD
# ============================================================
# Users can upload PDFs directly from the UI.
#
# Uploaded PDFs are:
# 1. Saved into the data folder
# 2. Indexed into ChromaDB
# 3. Immediately available for querying
# ============================================================

st.sidebar.header("📄 Document Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


# ============================================================
# PROCESS NEW PDF UPLOAD
# ============================================================

if uploaded_file:

    os.makedirs("data", exist_ok=True)

    save_path = os.path.join("data", uploaded_file.name)

    # --------------------------------------------------------
    # Save uploaded file
    # --------------------------------------------------------

    with open(save_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    # --------------------------------------------------------
    # Index uploaded document
    # --------------------------------------------------------

    try:

        chunk_count = process_pdf(save_path)
        uploaded_documents = st.session_state.get("uploaded_documents", [])
        if uploaded_file.name not in uploaded_documents:
            uploaded_documents.append(uploaded_file.name)
        st.session_state.uploaded_documents = uploaded_documents

        st.sidebar.success(
            f"Indexed successfully ({chunk_count} chunks)"
        )

    except Exception as error:

        st.sidebar.error(
            f"Indexing failed: {error}"
        )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================
# Renders all previous messages for this Streamlit session.
# ============================================================

for message in st.session_state.messages:
    render_message(message)


# ============================================================
# USER INPUT
# ============================================================
# Chat input box displayed at the bottom.
# ============================================================

question = st.chat_input("Ask a question...")


# ============================================================
# PROCESS USER QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # Store and display user message
    # --------------------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    render_message({
        "role": "user",
        "content": question
    })

    # --------------------------------------------------------
    # Generate response
    # --------------------------------------------------------
    # NEW: Pass session_id to ask_question so that session
    # memory and vector storage are correctly scoped.
    # --------------------------------------------------------

    current_turn_index = st.session_state.turn_count

    with st.spinner("Thinking..."):

        answer, sources = rag_engine.ask_question(
            question,
            session_id=st.session_state.session_id   # NEW
        )

    # --------------------------------------------------------
    # NEW: Increment turn counter after each exchange
    # --------------------------------------------------------

    st.session_state.turn_count += 1

    # --------------------------------------------------------
    # Build response text with sources
    # --------------------------------------------------------

    response_text = answer

    unique_sources = []
    seen_sources = set()
    for source in sources:
        if source not in seen_sources:
            seen_sources.add(source)
            unique_sources.append(source)

    # --------------------------------------------------------
    # Display assistant message
    # --------------------------------------------------------

    render_message({
        "role": "assistant",
        "content": response_text,
        "sources": unique_sources
    })

    # --------------------------------------------------------
    # Save assistant message to display history
    # --------------------------------------------------------

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "sources": unique_sources
    })

    update_turn_display(
        st.session_state.session_id,
        current_turn_index,
        response_text,
        unique_sources
    )

    # --------------------------------------------------------
    # NEW: Rerun so sidebar counters (turn count, vector
    # count) refresh immediately after each exchange.
    # --------------------------------------------------------

    st.rerun()
