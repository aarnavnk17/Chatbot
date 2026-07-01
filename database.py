import hashlib
import math
import os
import time
import uuid
from datetime import datetime, timezone

from ollama import chat
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import ConfigurationError, OperationFailure, ServerSelectionTimeoutError
from sentence_transformers import SentenceTransformer

from config import SESSION_RETRIEVAL_RESULTS

MONGODB_URI = os.getenv("MONGODB_URI", "").strip()
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "rag_chatbot").strip()
TITLE_MODEL = os.getenv("SESSION_TITLE_MODEL", "qwen2.5-coder:7b")

_client = None
_db = None
_indexes_ready = False
_embedding_model = None
_embedding_model_error = None


def _utcnow():
    return datetime.now(timezone.utc)


def _normalize_id(value):
    return value or str(uuid.uuid4())


def _embed(text):
    model = _get_embedding_model()
    if model is not None:
        return model.encode(text).tolist()
    return _fallback_embed(text)


def _get_embedding_model():
    global _embedding_model, _embedding_model_error
    if _embedding_model is not None or _embedding_model_error is not None:
        return _embedding_model
    try:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as exc:
        _embedding_model_error = exc
    return _embedding_model


def _fallback_embed(text, dimensions=384):
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [((digest[i % len(digest)] / 255.0) * 2.0) - 1.0 for i in range(dimensions)]


def _connect():
    global _client, _db
    if _client is not None and _db is not None:
        return _db
    if not MONGODB_URI:
        raise ConfigurationError("MONGODB_URI is not set.")
    _client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000, socketTimeoutMS=5000)
    _client.admin.command("ping")
    _db = _client[MONGODB_DB_NAME]
    return _db


def _collections():
    db = _connect()
    return db["users"], db["sessions"], db["conversations"]


def ensure_indexes():
    global _indexes_ready
    if _indexes_ready:
        return
    users, sessions, conversations = _collections()
    users.create_index([("user_id", ASCENDING)], unique=True)
    sessions.create_index([("session_id", ASCENDING)], unique=True)
    sessions.create_index([("user_id", ASCENDING), ("last_activity", DESCENDING)])
    sessions.create_index([("last_activity", DESCENDING)])
    conversations.create_index([("conversation_id", ASCENDING)], unique=True)
    conversations.create_index([("session_id", ASCENDING), ("message_index", ASCENDING)])
    conversations.create_index([("user_id", ASCENDING), ("timestamp", DESCENDING)])
    conversations.create_index([("timestamp", DESCENDING)])
    _indexes_ready = True


def create_user(user_id=None, username="", email="", settings=None):
    """Create or update a user document."""
    ensure_indexes()
    users, _, _ = _collections()
    user_id = _normalize_id(user_id)
    now = _utcnow()
    doc = {
        "user_id": user_id,
        "username": username,
        "email": email,
        "created_at": now,
        "updated_at": now,
        "total_sessions": 0,
        "settings": settings or {},
    }
    users.update_one({"user_id": user_id}, {"$setOnInsert": {"created_at": now}, "$set": doc}, upsert=True)
    return get_user(user_id)


def get_user(user_id):
    ensure_indexes()
    users, _, _ = _collections()
    return users.find_one({"user_id": user_id})


def create_session(user_id="legacy-user", title="Untitled Session", summary="", session_id=None):
    """Create a session owned by a user."""
    ensure_indexes()
    users, sessions, _ = _collections()
    session_id = _normalize_id(session_id)
    now = _utcnow()
    users.update_one(
        {"user_id": user_id},
        {"$setOnInsert": {"user_id": user_id, "created_at": now, "updated_at": now, "total_sessions": 0, "settings": {}}},
        upsert=True,
    )
    sessions.update_one(
        {"session_id": session_id},
        {"$setOnInsert": {"created_at": now, "message_count": 0, "active": True, "archived": False, "deleted": False},
         "$set": {"session_id": session_id, "user_id": user_id, "title": title, "summary": summary, "updated_at": now, "last_activity": now}},
        upsert=True,
    )
    users.update_one({"user_id": user_id}, {"$inc": {"total_sessions": 1}, "$set": {"updated_at": now}})
    return get_session(session_id)


def get_session(session_id):
    ensure_indexes()
    _, sessions, _ = _collections()
    return sessions.find_one({"session_id": session_id})


def update_session(session_id, **fields):
    ensure_indexes()
    _, sessions, _ = _collections()
    fields["updated_at"] = _utcnow()
    sessions.update_one({"session_id": session_id}, {"$set": fields})
    return get_session(session_id)


def rename_session(session_id, title):
    return update_session(session_id, title=" ".join((title or "").split()))


def update_session_summary(session_id, summary):
    return update_session(session_id, summary=summary or "")


def list_recent_sessions(user_id="legacy-user", limit=5, search=None):
    ensure_indexes()
    _, sessions, conversations = _collections()
    query = {"user_id": user_id, "deleted": {"$ne": True}}
    if search:
        query["$or"] = [
            {"title": {"$regex": search, "$options": "i"}},
            {"summary": {"$regex": search, "$options": "i"}},
        ]
    rows = list(sessions.find(query).sort("last_activity", DESCENDING).limit(limit))
    return [_session_view(row) for row in rows]


def delete_session(session_id):
    return update_session(session_id, deleted=True, active=False)


def archive_session(session_id):
    return update_session(session_id, archived=True, active=False)


def save_message(session_id, user_id, role, content, message_index, embedding=None, retrieved_sources=None, used_rag=False, model=None, latency_ms=None):
    """Persist a single message document in conversations."""
    ensure_indexes()
    _, sessions, conversations = _collections()
    session = get_session(session_id)
    if not session:
        session = create_session(user_id=user_id, session_id=session_id)
    now = _utcnow()
    conversation_id = str(uuid.uuid4())
    doc = {
        "conversation_id": conversation_id,
        "session_id": session_id,
        "user_id": user_id,
        "role": role,
        "content": content,
        "timestamp": now,
        "message_index": message_index,
        "embedding": embedding,
        "retrieved_sources": retrieved_sources or [],
        "used_rag": used_rag,
        "model": model,
        "latency_ms": latency_ms,
    }
    conversations.insert_one(doc)
    sessions.update_one({"session_id": session_id}, {"$set": {"updated_at": now, "last_activity": now}, "$inc": {"message_count": 1}})
    return doc


def get_session_messages(session_id):
    ensure_indexes()
    _, _, conversations = _collections()
    return list(conversations.find({"session_id": session_id}).sort("message_index", ASCENDING))


def get_session_turns(session_id):
    """Compatibility helper returning paired user/assistant turns."""
    messages = get_session_messages(session_id)
    turns = []
    current = {}
    for message in messages:
        if message.get("role") == "user":
            if current:
                turns.append(current)
            current = {
                "turn_index": message.get("message_index", 0) // 2,
                "timestamp": message.get("timestamp"),
                "user_msg": message.get("content", ""),
                "assistant_msg": "",
                "assistant_display_msg": "",
                "assistant_sources": [],
            }
        elif message.get("role") == "assistant":
            current["assistant_msg"] = message.get("content", "")
            current["assistant_display_msg"] = message.get("content", "")
            current["assistant_sources"] = message.get("retrieved_sources", [])
    if current:
        turns.append(current)
    return turns


def get_recent_messages(session_id, limit=10):
    ensure_indexes()
    _, _, conversations = _collections()
    return list(conversations.find({"session_id": session_id}).sort("message_index", DESCENDING).limit(limit))


def get_session_count(user_id="legacy-user"):
    ensure_indexes()
    _, sessions, _ = _collections()
    query = {"user_id": user_id, "deleted": {"$ne": True}} if user_id is not None else {}
    return sessions.count_documents(query)


def get_message_count(session_id=None):
    ensure_indexes()
    _, _, conversations = _collections()
    query = {} if session_id is None else {"session_id": session_id}
    return conversations.count_documents(query)


def clear_session(session_id):
    ensure_indexes()
    _, sessions, conversations = _collections()
    conversations.delete_many({"session_id": session_id})
    sessions.delete_one({"session_id": session_id})


def rename_session_if_needed(session_id, title):
    return rename_session(session_id, title)


def _session_view(row):
    return {
        "session_id": row.get("session_id"),
        "title": row.get("title", "Untitled Session"),
        "summary": row.get("summary", ""),
        "message_count": row.get("message_count", 0),
        "last_activity": row.get("last_activity"),
        "updated_at": row.get("updated_at"),
        "created_at": row.get("created_at"),
        "active": row.get("active", True),
        "archived": row.get("archived", False),
        "deleted": row.get("deleted", False),
    }


def get_session_summary(session_id):
    row = get_session(session_id) or {}
    return _session_view(row) | {"session_id": session_id}


def get_session_messages_text(session_id):
    messages = get_session_messages(session_id)
    return "\n\n".join(f"{m.get('role')}: {m.get('content', '')}" for m in messages)


def get_session_turn_count(session_id):
    return get_message_count(session_id) // 2


def retrieve_relevant_turns(session_id, query):
    turns = get_session_turns(session_id)
    if not turns:
        return ""
    query_embedding = _embed(query)
    scored = []
    for turn in turns:
        text = f"User: {turn.get('user_msg', '')}\nAssistant: {turn.get('assistant_msg', '')}"
        turn_embedding = _embed(text)
        numerator = sum(a * b for a, b in zip(query_embedding, turn_embedding))
        query_norm = math.sqrt(sum(a * a for a in query_embedding))
        embed_norm = math.sqrt(sum(b * b for b in turn_embedding))
        score = numerator / (query_norm * embed_norm + 1e-9)
        scored.append((score, turn))
    scored.sort(key=lambda item: item[0], reverse=True)
    return "\n\n---\n\n".join(
        f"User: {turn.get('user_msg', '')}\nAssistant: {turn.get('assistant_msg', '')}"
        for score, turn in scored[:SESSION_RETRIEVAL_RESULTS]
        if score > 0
    )


def _meaningful_title_seed(messages):
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"]
    return ""


def generate_session_title(user_msg=None, assistant_msg=None, session_id=None):
    seed = " ".join(part for part in [user_msg or "", assistant_msg or ""] if part).strip()
    if not seed:
        return "Untitled Session"
    prompt = f"Create a concise title for this chat session. Return only 2 to 5 words, Title Case, no punctuation.\n\nConversation:\n{seed}"
    try:
        response = chat(model=TITLE_MODEL, messages=[{"role": "user", "content": prompt}])
        title = " ".join(response["message"]["content"].replace('"', "").replace("'", "").split())
        if 2 <= len(title.split()) <= 5:
            return title[:40]
    except Exception:
        pass
    words = [w.strip(".,:;!?()[]{}\"'").capitalize() for w in seed.split() if w.isalpha()]
    return " ".join(words[:4]) or "Untitled Session"


def ensure_session_title(session_id):
    session = get_session(session_id) or {}
    if session.get("title") and session["title"] != "Untitled Session":
        return session["title"]
    messages = get_session_messages(session_id)
    assistant = _meaningful_title_seed(messages)
    if not assistant:
        return "Untitled Session"
    title = generate_session_title(messages[0].get("content") if messages else "", assistant, session_id)
    rename_session(session_id, title)
    return title


def save_turn(session_id, user_id, user_msg, assistant_msg, message_index, assistant_sources=None, used_rag=False, model=None, latency_ms=None):
    save_message(session_id, user_id, "user", user_msg, message_index, model=model, latency_ms=latency_ms)
    assistant_doc = save_message(
        session_id,
        user_id,
        "assistant",
        assistant_msg,
        message_index + 1,
        retrieved_sources=assistant_sources,
        used_rag=used_rag,
        model=model,
        latency_ms=latency_ms,
    )
    if assistant_msg and ensure_session_title(session_id) == "Untitled Session":
        rename_session(session_id, generate_session_title(user_msg, assistant_msg, session_id))
    return assistant_doc


def get_recent_session_messages(session_id, limit=10):
    return get_recent_messages(session_id, limit=limit)


def migrate_legacy_sessions(legacy_turns, legacy_sessions):
    """Copy old turn-based session collections into the new schema."""
    ensure_indexes()
    users, sessions, conversations = _collections()
    migrated_sessions = 0
    migrated_conversations = 0
    session_rows = legacy_sessions.find({})
    for session_row in session_rows:
        sid = session_row.get("session_id")
        if not sid:
            continue
        user_id = session_row.get("user_id", "legacy-user")
        create_user(user_id=user_id)
        create_session(user_id=user_id, session_id=sid, title=session_row.get("title", "Untitled Session"), summary=session_row.get("summary", ""))
        turns = list(legacy_turns.find({"session_id": sid}).sort("turn_index", ASCENDING))
        for turn in turns:
            base_index = int(turn.get("turn_index", 0)) * 2
            save_message(sid, user_id, "user", turn.get("user_msg", ""), base_index, embedding=turn.get("embedding"))
            save_message(sid, user_id, "assistant", turn.get("assistant_msg", ""), base_index + 1, embedding=turn.get("embedding"), retrieved_sources=turn.get("assistant_sources", []))
            migrated_conversations += 2
        update_session(sid, message_count=len(turns) * 2, last_activity=session_row.get("last_activity"), updated_at=session_row.get("updated_at"))
        migrated_sessions += 1
    return {"migrated_sessions": migrated_sessions, "migrated_messages": migrated_conversations}


def get_backend_status():
    try:
        _connect()
        return {"backend": "mongo", "reason": "MongoDB ping succeeded.", "mongodb_uri_configured": bool(MONGODB_URI), "mongodb_db_name": MONGODB_DB_NAME}
    except Exception as exc:
        return {"backend": "legacy", "reason": str(exc), "mongodb_uri_configured": bool(MONGODB_URI), "mongodb_db_name": MONGODB_DB_NAME}
