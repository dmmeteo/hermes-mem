"""Plugin-owned SQLite store: durable episode excerpts, chunks, FTS, vectors and curated records."""

from __future__ import annotations

import sqlite3
import time
from array import array
from pathlib import Path

# 1: episodes, chunks, vectors. 2: adds curated_records and curated_events (additive; ADR 0005).
SCHEMA_VERSION = 2
MIGRATES_FROM = {1}

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS sources (
    session_id TEXT PRIMARY KEY,
    source TEXT,
    scope TEXT NOT NULL,
    title TEXT,
    started_at REAL,
    state TEXT NOT NULL CHECK (state IN ('present', 'missing')),
    eligible INTEGER NOT NULL,
    skip_reason TEXT,
    fingerprint TEXT,
    checked_at REAL NOT NULL
);

-- embeddings.model_id holds the EmbeddingSpace.space_id (backend, model, revision, preprocessing).
-- One row per version of an episode. episode_key is stable across versions.
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY,
    episode_key TEXT NOT NULL,
    version INTEGER NOT NULL,
    session_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    identity_mode TEXT NOT NULL,
    user_ref TEXT,
    assistant_refs TEXT NOT NULL,
    first_message_id INTEGER NOT NULL,
    last_message_id INTEGER NOT NULL,
    user_text TEXT NOT NULL,
    assistant_text TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at REAL NOT NULL,
    compacted INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('current', 'superseded', 'retracted')),
    source_state TEXT NOT NULL CHECK (source_state IN ('present', 'missing')),
    superseded_by INTEGER REFERENCES episodes(id),
    status_reason TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE (episode_key, version)
);
CREATE UNIQUE INDEX IF NOT EXISTS episodes_one_current
    ON episodes(episode_key) WHERE status = 'current';
CREATE INDEX IF NOT EXISTS episodes_session ON episodes(session_id, status);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    episode_id INTEGER NOT NULL REFERENCES episodes(id),
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    UNIQUE (episode_id, chunk_index)
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content='chunks', content_rowid='id');
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts_trigram USING fts5(
    text, content='chunks', content_rowid='id', tokenize='trigram'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
    INSERT INTO chunks_fts_trigram(rowid, text) VALUES (new.id, new.text);
END;

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id INTEGER NOT NULL REFERENCES chunks(id),
    model_id TEXT NOT NULL,
    dim INTEGER NOT NULL,
    text_hash TEXT NOT NULL,
    vector BLOB NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (chunk_id, model_id)
);

CREATE TABLE IF NOT EXISTS embed_failures (
    chunk_id INTEGER NOT NULL,
    model_id TEXT NOT NULL,
    attempts INTEGER NOT NULL,
    last_error TEXT NOT NULL,
    retry_after REAL NOT NULL,
    PRIMARY KEY (chunk_id, model_id)
);

-- Curated records (ADR 0005). Content columns never change after insert; a revision inserts a
-- successor. Rows are never deleted, and curated_events is an append-only provenance log.
CREATE TABLE IF NOT EXISTS curated_records (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('peer', 'intent', 'observation', 'decision')),
    scope TEXT NOT NULL CHECK (scope != ''),
    subject TEXT NOT NULL,
    statement TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('proposed', 'accepted', 'superseded', 'retracted')),
    basis TEXT NOT NULL CHECK (basis IN ('declared', 'inferred')),
    confidence TEXT NOT NULL CHECK (confidence IN ('low', 'medium', 'high')),
    evidence TEXT NOT NULL DEFAULT '[]',
    author TEXT NOT NULL CHECK (author IN ('agent', 'operator')),
    supersedes TEXT REFERENCES curated_records(id) DEFERRABLE INITIALLY DEFERRED,
    superseded_by TEXT REFERENCES curated_records(id) DEFERRABLE INITIALLY DEFERRED,
    note TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    reviewed_at REAL,
    CHECK ((status = 'superseded') = (superseded_by IS NOT NULL))
);
CREATE UNIQUE INDEX IF NOT EXISTS curated_one_successor ON curated_records(supersedes) WHERE supersedes IS NOT NULL;
CREATE INDEX IF NOT EXISTS curated_scope ON curated_records(scope, kind, status);
CREATE TRIGGER IF NOT EXISTS curated_content_immutable
BEFORE UPDATE OF id, kind, scope, subject, statement, basis, confidence, evidence, author, supersedes, created_at
ON curated_records BEGIN
    SELECT RAISE(ABORT, 'curated record content is immutable; revise it to create a successor');
END;
CREATE TRIGGER IF NOT EXISTS curated_final_status BEFORE UPDATE OF status ON curated_records
WHEN OLD.status IN ('superseded', 'retracted') AND NEW.status != OLD.status BEGIN
    SELECT RAISE(ABORT, 'superseded and retracted curated records are final');
END;
CREATE TRIGGER IF NOT EXISTS curated_successor_final BEFORE UPDATE OF superseded_by ON curated_records
WHEN OLD.superseded_by IS NOT NULL AND NEW.superseded_by IS NOT OLD.superseded_by BEGIN
    SELECT RAISE(ABORT, 'superseded_by is final once set');
END;
CREATE TRIGGER IF NOT EXISTS curated_no_unaccept BEFORE UPDATE OF status ON curated_records
WHEN OLD.status = 'accepted' AND NEW.status = 'proposed' BEGIN
    SELECT RAISE(ABORT, 'an accepted curated record cannot return to proposed');
END;
CREATE TRIGGER IF NOT EXISTS curated_never_deleted BEFORE DELETE ON curated_records BEGIN
    SELECT RAISE(ABORT, 'curated records are never deleted');
END;

CREATE TABLE IF NOT EXISTS curated_events (
    id INTEGER PRIMARY KEY,
    record_id TEXT NOT NULL REFERENCES curated_records(id) DEFERRABLE INITIALLY DEFERRED,
    action TEXT NOT NULL CHECK (action IN ('propose', 'add', 'revise', 'supersede', 'accept', 'retract', 'import')),
    author TEXT NOT NULL CHECK (author IN ('agent', 'operator')),
    status TEXT NOT NULL,
    note TEXT,
    source TEXT,
    at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS curated_events_record ON curated_events(record_id, id);
CREATE TRIGGER IF NOT EXISTS curated_events_no_update BEFORE UPDATE ON curated_events BEGIN
    SELECT RAISE(ABORT, 'curated history is append-only');
END;
CREATE TRIGGER IF NOT EXISTS curated_events_no_delete BEFORE DELETE ON curated_events BEGIN
    SELECT RAISE(ABORT, 'curated history is append-only');
END;
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA foreign_keys=ON")
    if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'").fetchone():
        found = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if found is not None and int(found["value"]) not in MIGRATES_FROM | {SCHEMA_VERSION}:
            conn.close()  # refuse before any DDL can touch a newer store
            raise RuntimeError(f"hermes-mem store schema {found['value']} != supported {SCHEMA_VERSION}")
    conn.executescript(SCHEMA)
    found = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    if found is None:
        conn.execute("INSERT OR IGNORE INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
    elif int(found["value"]) in MIGRATES_FROM:
        # Every later table is CREATE ... IF NOT EXISTS above, so the upgrade is that DDL plus this mark.
        conn.execute("UPDATE meta SET value=? WHERE key='schema_version' AND value=?", (str(SCHEMA_VERSION), found["value"]))
    elif int(found["value"]) != SCHEMA_VERSION:
        raise RuntimeError(f"hermes-mem store schema {found['value']} != supported {SCHEMA_VERSION}")
    return conn


def get_meta(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def pending_chunks(conn: sqlite3.Connection, model_id: str, limit: int) -> list[sqlite3.Row]:
    """Chunks of current episodes without a fresh vector for model_id, skipping backed-off failures."""
    return conn.execute(
        """
        SELECT c.id, c.text, c.text_hash FROM chunks c
        JOIN episodes e ON e.id = c.episode_id AND e.status = 'current'
        LEFT JOIN embeddings v ON v.chunk_id = c.id AND v.model_id = ? AND v.text_hash = c.text_hash
        LEFT JOIN embed_failures f ON f.chunk_id = c.id AND f.model_id = ?
        WHERE v.chunk_id IS NULL AND (f.chunk_id IS NULL OR f.retry_after <= ?)
        ORDER BY c.id LIMIT ?
        """,
        (model_id, model_id, time.time(), limit),
    ).fetchall()


def save_vectors(conn: sqlite3.Connection, model_id: str, rows: list[sqlite3.Row], vectors: list[array]) -> None:
    now = time.time()
    conn.execute("BEGIN IMMEDIATE")
    try:
        for row, vector in zip(rows, vectors, strict=True):
            conn.execute(
                """INSERT INTO embeddings(chunk_id, model_id, dim, text_hash, vector, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(chunk_id, model_id) DO UPDATE SET
                     dim=excluded.dim, text_hash=excluded.text_hash,
                     vector=excluded.vector, created_at=excluded.created_at""",
                (row["id"], model_id, len(vector), row["text_hash"], array("f", vector).tobytes(), now),
            )
            conn.execute("DELETE FROM embed_failures WHERE chunk_id=? AND model_id=?", (row["id"], model_id))
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def record_failure(conn: sqlite3.Connection, model_id: str, rows: list[sqlite3.Row], error: str) -> None:
    now = time.time()
    for row in rows:
        conn.execute(
            """INSERT INTO embed_failures(chunk_id, model_id, attempts, last_error, retry_after)
               VALUES (?, ?, 1, ?, ?)
               ON CONFLICT(chunk_id, model_id) DO UPDATE SET
                 attempts = attempts + 1, last_error = excluded.last_error,
                 retry_after = ? + min(3600, 30 * (1 << min(attempts, 7)))""",
            (row["id"], model_id, error[:500], now + 30, now),
        )


def space_coverage(conn: sqlite3.Connection) -> list[dict]:
    """Every embedding space with stored vectors, and how many current chunks each still covers."""
    rows = conn.execute(
        """SELECT v.model_id, MIN(v.dim) AS dim, MAX(v.dim) AS max_dim,
                  SUM(CASE WHEN e.status='current' AND v.text_hash=c.text_hash THEN 1 ELSE 0 END) AS current_vectors,
                  COUNT(*) AS stored_vectors
           FROM embeddings v JOIN chunks c ON c.id=v.chunk_id JOIN episodes e ON e.id=c.episode_id
           GROUP BY v.model_id ORDER BY v.model_id"""
    ).fetchall()
    return [dict(r) for r in rows]


def current_chunk_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM chunks c JOIN episodes e ON e.id=c.episode_id AND e.status='current'").fetchone()[0]


def failure_summary(conn: sqlite3.Connection) -> dict[str, dict]:
    """Per space: current chunks whose embedding failed, how many are still backing off, and the latest error."""
    rows = conn.execute(
        """SELECT f.model_id, COUNT(*) AS failed, SUM(f.retry_after > ?) AS backing_off,
                  (SELECT last_error FROM embed_failures g WHERE g.model_id = f.model_id
                   ORDER BY g.retry_after DESC LIMIT 1) AS last_error
           FROM embed_failures f JOIN chunks c ON c.id = f.chunk_id
           JOIN episodes e ON e.id = c.episode_id AND e.status = 'current'
           GROUP BY f.model_id""",
        (time.time(),),
    ).fetchall()
    return {r["model_id"]: {"failed": r["failed"], "backing_off": r["backing_off"], "last_error": r["last_error"]} for r in rows}
