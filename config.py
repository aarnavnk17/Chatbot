from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_PATH = "./db"

COLLECTION_NAME = "knowledge_base"

CLIP_MODEL_NAME = "laion/CLIP-ViT-B32-128-multistep"
CLIP_PROCESSOR_NAME = "laion/CLIP-ViT-B32-128-multistep"
CLIP_TOKENIZER_NAME = "laion/CLIP-ViT-B32-128-multistep"

CHROMADB_COLLECTION = {
    'name': COLLECTION_NAME,
    'vector_size': 128
}

# ============================================================
# SESSION MEMORY CONFIGURATION
# ============================================================
# Controls how short-term and vector-backed session memory
# behave during a chat session.
# ============================================================

# ChromaDB collection that stores per-session conversation
# turns as embeddings (separate from document knowledge_base).
SESSION_COLLECTION_NAME = "session_conversations"
SESSION_INDEX_COLLECTION_NAME = "session_index"

# Maximum number of recent conversation turns (user + assistant
# pairs) kept in the in-memory rolling buffer per session.
# Older turns are dropped when the buffer exceeds this limit.
SHORT_TERM_MEMORY_WINDOW = 10

# Number of semantically-similar past turns to retrieve from
# the session ChromaDB collection when building a prompt.
SESSION_RETRIEVAL_RESULTS = 3
