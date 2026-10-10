"""Read-only access to the Hermes state.db of one profile. Never writes."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .select import HostMessage

EXCLUDED_SOURCES = {"cron", "subagent", "kanban", "tool"}
HIDDEN_DISPLAY_KINDS = {"hidden", "internal_notification", "process_complete", "async_delegation_complete"}
LOCAL_SOURCES = {"cli", "tui", "desktop", "acp", "webui", "oneshot"}
JSON_PREFIX = "\x00json:"


@dataclass(frozen=True)
class HostSession:
    id: str
    source: str
    title: str | None
    started_at: float
    ended_at: float | None
    end_reason: str | None
    parent_session_id: str | None
    user_id: str | None
    chat_id: str | None
    chat_type: str | None
    thread_id: str | None
    session_key: str | None
    origin: dict
    hidden: bool
    delegated: bool


def _connect(path: Path) -> sqlite3.Connection:
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    try:
        from hermes_cli.sqlite_safe_read import connect_tracked
    except ImportError:
        conn = sqlite3.connect(uri, uri=True, timeout=5.0, isolation_level=None)
    else:
        conn = connect_tracked(uri, tracking_path=path, uri=True, timeout=5.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def open_state(path: Path):
    """Open state.db read-only with Hermes' bounded retry for transient checkpoint I/O errors."""
    last: Exception | None = None
    for attempt in range(3):
        conn = None
        try:
            conn = _connect(path)
            conn.execute("SELECT 1 FROM sessions LIMIT 1")
            break
        except sqlite3.OperationalError as exc:
            if conn is not None:
                conn.close()
            last = exc
            if "disk I/O" not in str(exc):
                raise
            time.sleep(0.05 * (attempt + 1))
    else:
        raise last  # type: ignore[misc]
    try:
        yield HostState(conn)
    finally:
        conn.close()


def decode_content(raw: str | None) -> str:
    if not raw:
        return ""
    if not raw.startswith(JSON_PREFIX):
        return raw
    try:
        parts = json.loads(raw[len(JSON_PREFIX):])
    except ValueError:
        return ""
    if isinstance(parts, str):
        return parts
    texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") in ("text", "input_text")]
    return "\n".join(t for t in texts if t)


class HostState:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        message_columns = {row["name"] for row in conn.execute("PRAGMA table_info(messages)")}
        self.session_columns = {row["name"] for row in conn.execute("PRAGMA table_info(sessions)")}
        self.has_message_uid = "message_uid" in message_columns
        self.message_columns = message_columns
        self.identity_mode = "message_uid" if self.has_message_uid else "role_timestamp"

    def store_instance(self) -> str | None:
        try:
            row = self.conn.execute("SELECT value FROM state_meta WHERE key='store_instance_id'").fetchone()
        except sqlite3.OperationalError:
            return None
        return row["value"] if row else None

    def _col(self, name: str, default: str = "NULL") -> str:
        return name if name in self.session_columns else f"{default} AS {name}"

    def sessions(self, after: str = "", limit: int = 1000, ids: list[str] | None = None) -> list[HostSession]:
        cols = ", ".join(
            [
                "id", "source", "title", "started_at", "ended_at", "end_reason", "parent_session_id",
                "user_id", "model_config", self._col("chat_id"), self._col("chat_type"),
                self._col("thread_id"), self._col("session_key"), self._col("origin_json"),
                self._col("hidden", "0"),
            ]
        )
        if ids is not None:
            marks = ",".join("?" * len(ids))
            rows = self.conn.execute(f"SELECT {cols} FROM sessions WHERE id IN ({marks}) ORDER BY id", ids)
        else:
            rows = self.conn.execute(f"SELECT {cols} FROM sessions WHERE id > ? ORDER BY id LIMIT ?", (after, limit))
        return [self._session(row) for row in rows]

    def session_ids(self) -> set[str]:
        return {row[0] for row in self.conn.execute("SELECT id FROM sessions")}

    @staticmethod
    def _session(row: sqlite3.Row) -> HostSession:
        try:
            model_config = json.loads(row["model_config"] or "{}")
        except ValueError:
            model_config = {}
        try:
            origin = json.loads(row["origin_json"] or "{}")
        except ValueError:
            origin = {}
        return HostSession(
            id=row["id"],
            source=row["source"],
            title=row["title"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            end_reason=row["end_reason"],
            parent_session_id=row["parent_session_id"],
            user_id=row["user_id"],
            chat_id=row["chat_id"],
            chat_type=row["chat_type"],
            thread_id=row["thread_id"],
            session_key=row["session_key"],
            origin=origin if isinstance(origin, dict) else {},
            hidden=bool(row["hidden"]),
            delegated=isinstance(model_config, dict) and model_config.get("_delegate_from") is not None,
        )

    def visible_messages(self, session_id: str) -> list[HostMessage]:
        """User/assistant rows Hermes itself still shows and searches: active or compacted.

        Rewound/undone rows (active=0, compacted=0) and synthetic compaction summaries are
        excluded. Compaction clones share (role, timestamp) with their original; the lowest
        id wins so references stay stable.
        """
        uid = "message_uid" if self.has_message_uid else "NULL AS message_uid"
        rows = self.conn.execute(
            f"""SELECT id, {uid}, role, content, timestamp, tool_calls, compacted, active,
                       _compressed_summary, observed, display_kind, display_metadata
                FROM messages
                WHERE session_id = ? AND role IN ('user', 'assistant')
                  AND (active = 1 OR compacted = 1)
                ORDER BY id""",
            (session_id,),
        ).fetchall()
        by_key: dict[str, HostMessage] = {}
        for row in rows:
            if row["display_kind"] in HIDDEN_DISPLAY_KINDS or (row["role"] == "user" and row["observed"]):
                continue
            if _model_only(row["display_metadata"]):
                continue
            ref = f"uid:{row['message_uid']}" if row["message_uid"] else f"{row['role']}@{row['timestamp']!r}"
            content = decode_content(row["content"])
            existing = by_key.get(ref)
            if existing is not None and not (row["active"] and existing.content != content):
                continue
            by_key[ref] = HostMessage(
                id=existing.id if existing else row["id"],
                ref=ref,
                role=row["role"],
                content=content,
                timestamp=row["timestamp"],
                has_tool_calls=bool(row["tool_calls"] and row["tool_calls"] not in ("[]", "null")),
                compacted=bool(row["compacted"]) or bool(existing and existing.compacted),
                summary=bool(row["_compressed_summary"]),
            )
        return list(by_key.values())

    def rewound_refs(self, session_id: str) -> set[str]:
        rows = self.conn.execute(
            "SELECT role, timestamp FROM messages WHERE session_id=? AND active=0 AND compacted=0", (session_id,)
        )
        return {f"{r['role']}@{r['timestamp']!r}" for r in rows}

    def message_exists(self, session_id: str, message_id: int) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM messages WHERE session_id=? AND id=?", (session_id, message_id)
        ).fetchone() is not None


def _model_only(raw: str | None) -> bool:
    if not raw:
        return False
    try:
        data = json.loads(raw)
    except ValueError:
        return False
    return isinstance(data, dict) and bool(data.get("model_only"))


def eligibility(session: HostSession) -> str | None:
    """Reason a session is skipped by default, or None when it is a real conversation."""
    if session.source in EXCLUDED_SOURCES:
        return f"routine source {session.source}"
    if session.delegated:
        return "delegated subagent session"
    if session.hidden:
        return "hidden session"
    return None


def session_scope(session: HostSession) -> str:
    """Authoritative retrieval scope of a session. Display names never count as identity."""
    chat_type = session.chat_type or session.origin.get("chat_type")
    user_id = session.user_id or session.origin.get("user_id")
    chat_id = session.chat_id or session.origin.get("chat_id")
    if session.source in LOCAL_SOURCES and not chat_type:
        return "owner"
    if chat_type == "dm" and user_id:
        return f"peer:{session.source}:{user_id}"
    if chat_type in ("group", "channel", "thread") and chat_id:
        place = f"chat:{session.source}:{chat_id}" + (f":{session.thread_id}" if session.thread_id else "")
        participants = [str(p) for p in (session.origin.get("user_id_alt"), user_id) if p]
        key = session.session_key or ""
        owner = next((p for p in participants if key.endswith(f":{p}")), None)
        if owner:
            return f"{place}/peer:{owner}"
        # Shared room, or a per-user key we cannot prove: confine to this session.
    return f"session:{session.id}"
