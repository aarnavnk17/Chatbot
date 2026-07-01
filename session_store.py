"""Backward-compatible wrapper around the relational-style Mongo service."""

from database import (
    archive_session,
    clear_session,
    create_session,
    create_user,
    delete_session,
    ensure_session_title,
    generate_session_title,
    get_backend_status,
    get_message_count,
    get_recent_messages,
    get_recent_session_messages,
    get_session,
    get_session_count,
    get_session_messages,
    get_session_messages_text,
    get_session_turns,
    get_session_summary,
    get_session_turn_count,
    list_recent_sessions,
    migrate_legacy_sessions,
    rename_session,
    save_message,
    save_turn,
    retrieve_relevant_turns,
    update_session,
    update_session_summary,
)


def store_turn(session_id, user_msg, assistant_msg, turn_index, user_id="legacy-user"):
    return save_turn(session_id, user_id, user_msg, assistant_msg, turn_index * 2)


def update_turn_display(session_id, turn_index, assistant_display_msg, assistant_sources=None):
    """Compatibility no-op for the new per-message schema."""
    return True
