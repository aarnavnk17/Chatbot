# ============================================================
# TEST TOOLS
# ============================================================
# Unit tests for session_memory and session_store.
#
# Previously tested VectorDatabase from tools.py (now removed
# — it had a broken FAISS embedding dimension of 128 vs the
# correct 384 from all-MiniLM-L6-v2).
#
# Now tests the replacement modules:
#   session_memory  — in-memory rolling buffer
#   session_store   — ChromaDB-backed session vector storage
# ============================================================

import session_memory
import session_store

from grep_search import grep_search
from single_find_and_replace import single_find_and_replace

# ============================================================
# TEST: session_memory rolling buffer
# ============================================================
# Verifies that add_turn correctly stores turns and
# get_history_text formats them for prompt injection.
# ============================================================

def test_session_memory_add_and_get():
    """Add two turns and verify the formatted history text."""

    # Use a unique session ID to avoid interference
    sid = "test-memory-session"

    # Ensure clean state before test
    session_memory.clear(sid)

    # Add first turn
    session_memory.add_turn(sid, "What is Python?", "Python is a programming language.")

    # Add second turn
    session_memory.add_turn(sid, "Who created it?", "Guido van Rossum created Python.")

    text = session_memory.get_history_text(sid)

    assert "User: What is Python?" in text, f"Missing user turn. Got:\n{text}"
    assert "Assistant: Python is a programming language." in text, f"Missing assistant turn. Got:\n{text}"
    assert "User: Who created it?" in text, f"Missing second user turn. Got:\n{text}"

    # Clean up
    session_memory.clear(sid)

    print("PASS: test_session_memory_add_and_get")


# ============================================================
# TEST: session_memory rolling window trim
# ============================================================
# Verifies that the buffer is trimmed to SHORT_TERM_MEMORY_WINDOW
# pairs when it overflows.
# ============================================================

def test_session_memory_rolling_window():
    """Verify old turns are dropped when window overflows."""

    from config import SHORT_TERM_MEMORY_WINDOW

    sid = "test-rolling-window"
    session_memory.clear(sid)

    # Add more turns than the window allows
    total_turns = SHORT_TERM_MEMORY_WINDOW + 5

    for i in range(total_turns):
        session_memory.add_turn(sid, f"question {i}", f"answer {i}")

    # The buffer should contain at most SHORT_TERM_MEMORY_WINDOW pairs
    count = session_memory.get_turn_count(sid)
    max_messages = SHORT_TERM_MEMORY_WINDOW * 2

    assert count <= max_messages, (
        f"Buffer overflow: expected <= {max_messages} messages, got {count}"
    )

    session_memory.clear(sid)

    print(f"PASS: test_session_memory_rolling_window (kept {count}/{max_messages} messages)")


# ============================================================
# TEST: session_memory clear
# ============================================================

def test_session_memory_clear():
    """Verify that clear() empties the buffer completely."""

    sid = "test-clear-session"
    session_memory.add_turn(sid, "Hello", "Hi there!")

    session_memory.clear(sid)

    text = session_memory.get_history_text(sid)
    assert text == "", f"Buffer not cleared. Got: {repr(text)}"

    print("PASS: test_session_memory_clear")


# ============================================================
# TEST: session_store store and retrieve
# ============================================================
# Verifies that a stored turn can be semantically retrieved
# from ChromaDB within the same session.
# ============================================================

def test_session_store_store_and_retrieve():
    """Store a turn and verify semantic retrieval returns it."""

    sid = "test-store-retrieve"

    # Clear any leftover state from previous runs
    session_store.clear_session(sid)

    # Store one turn
    session_store.store_turn(
        sid,
        "What is machine learning?",
        "Machine learning is a subset of AI.",
        turn_index=0
    )

    # Retrieve with a semantically similar query
    result = session_store.retrieve_relevant_turns(sid, "Tell me about machine learning")

    assert "Machine learning" in result, (
        f"Retrieved turn not found in result. Got: {repr(result)}"
    )

    # Clean up
    session_store.clear_session(sid)

    print("PASS: test_session_store_store_and_retrieve")


# ============================================================
# TEST: session_store empty session returns empty string
# ============================================================

def test_session_store_empty_session():
    """Retrieving from a non-existent session returns empty string."""

    result = session_store.retrieve_relevant_turns(
        "completely-nonexistent-session-xyz",
        "any question"
    )

    assert result == "", f"Expected empty string. Got: {repr(result)}"

    print("PASS: test_session_store_empty_session")


# ============================================================
# TEST: session_store clear_session
# ============================================================

def test_session_store_clear():
    """Verify clear_session removes all turns for a session."""

    sid = "test-store-clear"
    session_store.clear_session(sid)

    session_store.store_turn(sid, "Q", "A", turn_index=0)

    count_before = session_store.get_session_turn_count(sid)
    assert count_before == 1, f"Expected 1 turn before clear. Got: {count_before}"

    session_store.clear_session(sid)

    count_after = session_store.get_session_turn_count(sid)
    assert count_after == 0, f"Expected 0 turns after clear. Got: {count_after}"

    print("PASS: test_session_store_clear")


# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":

    print("\nRunning test suite...\n")

    test_session_memory_add_and_get()
    test_session_memory_rolling_window()
    test_session_memory_clear()
    test_session_store_empty_session()
    test_session_store_store_and_retrieve()
    test_session_store_clear()

    print("\nAll tests passed.")