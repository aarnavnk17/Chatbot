# ============================================================
# TOOLS — DEPRECATED
# ============================================================
# The VectorDatabase class that was previously defined here
# used FAISS with embedding dimension 128, but the
# all-MiniLM-L6-v2 model produces 384-dimensional vectors.
# This caused a runtime crash on every call to
# store_session_vector().
#
# Session vector storage has been replaced by:
#   session_store.py  — ChromaDB-backed, correct 384-dim,
#                       persistent, and session-scoped.
#
# This file is kept as a placeholder to avoid import errors
# from any external scripts that may reference tools.py.
# ============================================================

# ============================================================
# MIGRATION NOTE
# ============================================================
# Old (broken) usage in rag_engine.py:
#
#   from tools import VectorDatabase
#   vector_db = VectorDatabase()        # dimension bug: 128
#   vector_db.store_session_vector(     # hardcoded session id
#       "session_123",
#       get_embedding(conversation)
#   )
#
# New (correct) usage in rag_engine.py:
#
#   from session_store import store_turn, retrieve_relevant_turns
#   store_turn(session_id, user_msg, assistant_msg, turn_index)
#   context = retrieve_relevant_turns(session_id, question)
# ============================================================