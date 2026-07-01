# ============================================================
# SESSION MEMORY — SHORT-TERM ROLLING BUFFER
# ============================================================
# Manages a per-session in-memory conversation history.
#
# Each session is identified by a unique session_id (UUID).
# Turns are stored as {"role": ..., "content": ...} dicts.
#
# The buffer is capped at SHORT_TERM_MEMORY_WINDOW turns
# (user + assistant messages counted separately).
# When the buffer is full, the oldest turns are dropped
# to keep memory fresh and token usage bounded.
#
# This module is purely in-memory — it does NOT persist
# across server restarts. For persistent retrieval, use
# session_store.py (ChromaDB-backed).
# ============================================================

from config import SHORT_TERM_MEMORY_WINDOW


# ============================================================
# SESSION REGISTRY
# ============================================================
# Maps session_id -> list of turn dicts.
# Kept at module level so all calls within the same Python
# process share the same registry.
# ============================================================

_session_buffers = {}


# ============================================================
# ADD TURN
# ============================================================
# Appends a user message and the corresponding assistant
# reply to the session buffer, then trims to window size.
#
# Arguments:
#   session_id    — unique identifier for this chat session
#   user_msg      — the question the user asked
#   assistant_msg — the answer the assistant produced
# ============================================================

def add_turn(session_id, user_msg, assistant_msg):

    # Initialise buffer for new sessions
    if session_id not in _session_buffers:
        _session_buffers[session_id] = []

    buffer = _session_buffers[session_id]

    # Append the user turn
    buffer.append({
        "role": "user",
        "content": user_msg
    })

    # Append the assistant turn
    buffer.append({
        "role": "assistant",
        "content": assistant_msg
    })

    # --------------------------------------------------------
    # TRIM TO ROLLING WINDOW
    # --------------------------------------------------------
    # SHORT_TERM_MEMORY_WINDOW is defined in pairs (turns),
    # so multiply by 2 to get the message count.
    # Example: window=10 -> keep last 20 messages (10 pairs).
    # --------------------------------------------------------

    max_messages = SHORT_TERM_MEMORY_WINDOW * 2

    if len(buffer) > max_messages:
        # Drop oldest messages from the front of the list
        _session_buffers[session_id] = buffer[-max_messages:]


# ============================================================
# GET HISTORY TEXT
# ============================================================
# Returns the session buffer as a formatted multi-line string
# ready to be injected into a prompt.
#
# Format:
#   User: <message>
#   Assistant: <message>
#   ...
#
# Returns an empty string if there is no history yet.
# ============================================================

def get_history_text(session_id):

    buffer = _session_buffers.get(session_id, [])

    if not buffer:
        return ""

    lines = []

    for turn in buffer:

        # Capitalise role label: "user" -> "User"
        role = turn["role"].capitalize()
        content = turn["content"]

        lines.append(f"{role}: {content}")

    # Join turns with a blank line separator for readability
    return "\n\n".join(lines)


# ============================================================
# GET TURNS (RAW)
# ============================================================
# Returns the raw list of turn dicts for a session.
# Useful when you need to pass history to the Ollama
# chat API directly (list of {role, content} dicts).
# ============================================================

def get_turns(session_id):
    return list(_session_buffers.get(session_id, []))


# ============================================================
# GET TURN COUNT
# ============================================================
# Returns the number of individual messages (not pairs)
# currently stored for the session.
# ============================================================

def get_turn_count(session_id):
    return len(_session_buffers.get(session_id, []))


# ============================================================
# CLEAR SESSION
# ============================================================
# Removes the entire buffer for a session.
# Call this when the user clicks "Clear Session" in the UI.
# ============================================================

def clear(session_id):
    if session_id in _session_buffers:
        del _session_buffers[session_id]
