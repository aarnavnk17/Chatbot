from pymongo.errors import ServerSelectionTimeoutError, ConfigurationError

from database import migrate_legacy_sessions
from pymongo import MongoClient
import os


if __name__ == "__main__":
    try:
        uri = os.getenv("MONGODB_URI", "").strip()
        db_name = os.getenv("MONGODB_DB_NAME", "rag_chatbot").strip()
        client = MongoClient(uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000, socketTimeoutMS=5000)
        client.admin.command("ping")
        db = client[db_name]
        result = migrate_legacy_sessions(db["session_turns"], db["session_metadata"])
    except (ServerSelectionTimeoutError, ConfigurationError) as exc:
        print("Migration could not reach MongoDB.")
        print("Check MONGODB_URI, Atlas network access, and your database user credentials.")
        print(f"Details: {exc}")
    else:
        print(
            f"Migration complete. Sessions: {result['migrated_sessions']}, "
            f"Messages: {result['migrated_messages']}"
        )
