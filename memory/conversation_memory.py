import os
import json
import glob
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import FileChatMessageHistory

HISTORY_DIR = "history"


def _session_history_path(session_id: str) -> str:
    """Return the history file path for a given session."""
    os.makedirs(HISTORY_DIR, exist_ok=True)
    # sanitise session_id so it's safe as a filename
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
    return os.path.join(HISTORY_DIR, f"{safe_id}.json")


def get_memory(session_id: str = "default"):
    """Return a ConversationBufferMemory backed by a per-session JSON file.
    Each session_id gets its own file so histories never mix.
    """
    history_path = _session_history_path(session_id)
    message_history = FileChatMessageHistory(file_path=history_path)

    return ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=message_history,
        return_messages=True
    )


def search_past_conversations(query: str, current_session_id: str,
                               max_context_messages: int = 6) -> str:
    """Search all past session history files (excluding current) for messages
    that are loosely relevant to the query.

    Returns a formatted string of the most relevant past exchange(s) to inject
    as context into the current session — like GPT's long-term memory feature.

    Strategy: simple keyword overlap scoring (no extra embedding call needed).
    """
    try:
        pattern = os.path.join(HISTORY_DIR, "*.json")
        history_files = glob.glob(pattern)
        current_file = _session_history_path(current_session_id)

        query_words = set(query.lower().split())
        best_score = 0
        best_context = ""

        for fpath in history_files:
            if os.path.abspath(fpath) == os.path.abspath(current_file):
                continue  # skip the current session's own history

            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            messages = data if isinstance(data, list) else data.get("messages", [])
            if not messages:
                continue

            # Score this session's history by keyword overlap
            combined_text = " ".join(
                m.get("data", {}).get("content", "") if isinstance(m, dict) else str(m)
                for m in messages
            ).lower()

            overlap = len(query_words & set(combined_text.split()))
            if overlap > best_score:
                best_score = overlap
                # Extract last N messages as context snippet
                recent = messages[-max_context_messages:]
                lines = []
                for m in recent:
                    if isinstance(m, dict):
                        role = m.get("type", m.get("data", {}).get("type", "unknown"))
                        content = m.get("data", {}).get("content", "")
                        lines.append(f"[{role.upper()}]: {content}")
                best_context = "\n".join(lines)

        if best_score == 0 or not best_context:
            return ""

        return (
            "📚 I found something relevant from a previous conversation:\n"
            + best_context
        )

    except Exception:
        return ""
