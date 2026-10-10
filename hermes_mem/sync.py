"""Idempotent ingestion and reconciliation from state.db into the plugin store."""

from __future__ import annotations

import fcntl
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import store
from .embed import Embedder, EmbeddingUnavailable
from .host import HostSession, HostState, eligibility, open_state, session_scope
from .select import CHUNK_CHARS, Episode, ProcessingPolicy, build_episodes, sha256

log = logging.getLogger("hermes_mem")


@dataclass
class SyncReport:
    sessions: int = 0
    skipped: int = 0
    unchanged: int = 0
    inserted: int = 0
    revised: int = 0
    rechunked: int = 0
    retracted: int = 0
    source_missing: int = 0
    identity_mode: str = ""
    processing: dict = field(default_factory=dict)
    complete: bool = False
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _fingerprint(scope: str, episodes: list[Episode], chunk_chars: int) -> str:
    # The default chunk size keeps the MVP fingerprint, so upgrading does not revisit every session.
    head = [scope] if chunk_chars == CHUNK_CHARS else [scope, chunk_chars]
    return sha256(json.dumps(head + [(e.key, e.content_hash, e.first_message_id) for e in episodes]))


def _chunks_differ(conn: sqlite3.Connection, episode_id: int, ep: Episode) -> bool:
    stored = [r[0] for r in conn.execute("SELECT text_hash FROM chunks WHERE episode_id=? ORDER BY chunk_index", (episode_id,))]
    return stored != [sha256(text) for text in ep.chunks]


def _insert_episode(conn: sqlite3.Connection, session: HostSession, scope: str, mode: str, ep: Episode, version: int) -> int:
    now = time.time()
    cur = conn.execute(
        """INSERT INTO episodes(episode_key, version, session_id, scope, identity_mode, user_ref, assistant_refs,
               first_message_id, last_message_id, user_text, assistant_text, started_at, ended_at, compacted,
               content_hash, status, source_state, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'current', 'present', ?, ?)""",
        (ep.key, version, session.id, scope, mode, ep.user_ref, json.dumps(ep.assistant_refs),
         ep.first_message_id, ep.last_message_id, ep.user_text, ep.assistant_text, ep.started_at,
         ep.ended_at, int(ep.compacted), ep.content_hash, now, now),
    )
    episode_id = cur.lastrowid
    for index, text in enumerate(ep.chunks):
        chunk_id = conn.execute(
            "INSERT INTO chunks(episode_id, chunk_index, text, text_hash) VALUES (?, ?, ?, ?)",
            (episode_id, index, text, sha256(text)),
        ).lastrowid
        # A vector depends only on (space, text): reuse ones an earlier version of this episode already has.
        conn.execute(
            """INSERT OR IGNORE INTO embeddings(chunk_id, model_id, dim, text_hash, vector, created_at)
               SELECT ?, v.model_id, v.dim, v.text_hash, v.vector, v.created_at
               FROM embeddings v JOIN chunks c ON c.id = v.chunk_id JOIN episodes e ON e.id = c.episode_id
               WHERE e.episode_key = ? AND e.id != ? AND c.text_hash = ? AND v.text_hash = c.text_hash
               GROUP BY v.model_id""",
            (chunk_id, ep.key, episode_id, sha256(text)),
        )
    return episode_id


def reconcile_session(
    conn: sqlite3.Connection, host: HostState, session: HostSession, report: SyncReport, policy: ProcessingPolicy
) -> None:
    now = time.time()
    skip = eligibility(session)
    scope = session_scope(session)
    if skip:
        report.skipped += 1
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                """INSERT INTO sources(session_id, source, scope, title, started_at, state, eligible, skip_reason, checked_at)
                   VALUES (?, ?, ?, ?, ?, 'present', 0, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET state='present', eligible=0, skip_reason=excluded.skip_reason,
                     fingerprint=NULL, checked_at=excluded.checked_at""",
                (session.id, session.source, scope, session.title, session.started_at, skip, now),
            )
            report.retracted += conn.execute(
                """UPDATE episodes SET status='retracted', status_reason=?, updated_at=?
                   WHERE session_id=? AND status='current'""",
                (f"session no longer eligible: {skip}", now, session.id),
            ).rowcount
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        return

    # Keys are scope + message identity, not session ids: compaction clones, rotated copies and
    # sessions whose parent was pruned all map to the same episode.
    episodes = policy.apply(build_episodes(scope, host.visible_messages(session.id)))
    fingerprint = _fingerprint(scope, episodes, policy.chunk_chars)
    known = conn.execute("SELECT fingerprint, state FROM sources WHERE session_id=?", (session.id,)).fetchone()
    if known and known["fingerprint"] == fingerprint and known["state"] == "present":
        report.unchanged += 1
        return

    conn.execute("BEGIN IMMEDIATE")
    try:
        derived = {ep.key: ep for ep in episodes}
        for ep in episodes:
            current = conn.execute(
                "SELECT id, content_hash, session_id, source_state, scope FROM episodes WHERE episode_key=? AND status='current'",
                (ep.key,),
            ).fetchone()
            if current is None:
                version = conn.execute(
                    "SELECT COALESCE(MAX(version), 0) + 1 FROM episodes WHERE episode_key=?", (ep.key,)
                ).fetchone()[0]
                _insert_episode(conn, session, scope, host.identity_mode, ep, version)
                report.inserted += 1
            elif current["content_hash"] != ep.content_hash or _chunks_differ(conn, current["id"], ep):
                content_changed = current["content_hash"] != ep.content_hash
                reason = "source content changed" if content_changed else f"re-chunked at sync.chunk_chars={policy.chunk_chars}"
                conn.execute(
                    "UPDATE episodes SET status='superseded', status_reason=?, updated_at=? WHERE id=?",
                    (reason, now, current["id"]),
                )
                version = conn.execute("SELECT MAX(version) + 1 FROM episodes WHERE episode_key=?", (ep.key,)).fetchone()[0]
                new_id = _insert_episode(conn, session, scope, host.identity_mode, ep, version)
                conn.execute("UPDATE episodes SET superseded_by=? WHERE id=?", (new_id, current["id"]))
                if content_changed:
                    report.revised += 1
                else:
                    report.rechunked += 1
            elif current["session_id"] != session.id and current["source_state"] == "missing":
                conn.execute(
                    """UPDATE episodes SET session_id=?, first_message_id=?, last_message_id=?, source_state='present',
                       updated_at=? WHERE id=?""",
                    (session.id, ep.first_message_id, ep.last_message_id, now, current["id"]),
                )
            elif current["session_id"] == session.id and (
                current["source_state"] != "present"
            ):
                conn.execute(
                    "UPDATE episodes SET source_state='present', first_message_id=?, last_message_id=?, updated_at=? WHERE id=?",
                    (ep.first_message_id, ep.last_message_id, now, current["id"]),
                )
        rewound = host.rewound_refs(session.id)
        for row in conn.execute(
            "SELECT id, episode_key, user_ref, assistant_refs, first_message_id FROM episodes WHERE session_id=? AND status='current'",
            (session.id,),
        ).fetchall():
            if row["episode_key"] in derived:
                continue
            refs = {row["user_ref"], *json.loads(row["assistant_refs"])} - {None}
            if refs & rewound:
                conn.execute(
                    "UPDATE episodes SET status='retracted', status_reason='rewound in source session', updated_at=? WHERE id=?",
                    (now, row["id"]),
                )
                report.retracted += 1
            elif not host.message_exists(session.id, row["first_message_id"]):
                changed = conn.execute(
                    "UPDATE episodes SET source_state='missing', updated_at=? WHERE id=? AND source_state='present'",
                    (now, row["id"]),
                ).rowcount
                report.source_missing += changed
            else:
                conn.execute(
                    "UPDATE episodes SET status='retracted', status_reason='no longer a visible exchange in source', updated_at=? WHERE id=?",
                    (now, row["id"]),
                )
                report.retracted += 1
        conn.execute(
            """INSERT INTO sources(session_id, source, scope, title, started_at, state, eligible, fingerprint, checked_at)
               VALUES (?, ?, ?, ?, ?, 'present', 1, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET source=excluded.source, scope=excluded.scope, title=excluded.title,
                 state='present', eligible=1, skip_reason=NULL, fingerprint=excluded.fingerprint, checked_at=excluded.checked_at""",
            (session.id, session.source, scope, session.title, session.started_at, fingerprint, now),
        )
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def mark_missing_sessions(conn: sqlite3.Connection, host: HostState, report: SyncReport) -> None:
    present = host.session_ids()
    now = time.time()
    gone = [
        row["session_id"]
        for row in conn.execute("SELECT session_id FROM sources WHERE state='present'")
        if row["session_id"] not in present
    ]
    if not gone:
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        for session_id in gone:
            conn.execute("UPDATE sources SET state='missing', fingerprint=NULL, checked_at=? WHERE session_id=?", (now, session_id))
            report.source_missing += conn.execute(
                "UPDATE episodes SET source_state='missing', updated_at=? WHERE session_id=? AND source_state='present'",
                (now, session_id),
            ).rowcount
        # A surviving copy (e.g. a compression child) may now be the only source: re-reconcile all.
        conn.execute("UPDATE sources SET fingerprint=NULL WHERE state='present'")
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def sync(
    conn: sqlite3.Connection,
    state_db: Path,
    *,
    policy: ProcessingPolicy,
    session_ids: list[str] | None = None,
    max_sessions: int = 200,
) -> SyncReport:
    """Reconcile a bounded batch of sessions. Repeating it converges; it never deletes excerpts.

    With session_ids, only those sessions are reconciled. Otherwise the backfill cursor
    advances through all sessions; reaching the end also marks vanished sessions missing
    and restarts the cursor so the next pass replays everything (catching missed hooks).
    """
    report = SyncReport(processing=policy.describe())
    if not state_db.exists():
        report.warnings.append(f"state.db not found at {state_db}")
        return report
    with open_state(state_db) as host:
        report.identity_mode = host.identity_mode
        instance = host.store_instance()
        recorded = store.get_meta(conn, "host_store_instance")
        if recorded and instance and recorded != instance:
            report.warnings.append("state.db store_instance_id changed: message ids may refer to a restored or replaced file")
        if instance:
            store.set_meta(conn, "host_store_instance", instance)
        if session_ids is not None:
            sessions = host.sessions(ids=session_ids)
            for session in sessions:
                reconcile_session(conn, host, session, report, policy)
                report.sessions += 1
            missing = set(session_ids) - {s.id for s in sessions}
            if missing:
                mark_missing_sessions(conn, host, report)
            return report
        cursor = store.get_meta(conn, "backfill_cursor", "") or ""
        batch = host.sessions(after=cursor, limit=max_sessions)
        for session in batch:
            reconcile_session(conn, host, session, report, policy)
            report.sessions += 1
            store.set_meta(conn, "backfill_cursor", session.id)
        if len(batch) < max_sessions:
            mark_missing_sessions(conn, host, report)
            store.set_meta(conn, "backfill_cursor", "")
            store.set_meta(conn, "last_full_pass_at", str(time.time()))
            report.complete = True
    return report


def embed_pending(conn: sqlite3.Connection, embedder: Embedder | None, *, batch_size: int = 32, max_batches: int = 50) -> dict:
    if embedder is None:
        return {"embedded": 0, "status": "disabled"}
    embedded = 0
    for _ in range(max_batches):
        rows = store.pending_chunks(conn, embedder.model_id, batch_size)
        if not rows:
            return {"embedded": embedded, "status": "complete"}
        try:
            vectors = embedder.embed([row["text"] for row in rows], kind="passage")
        except EmbeddingUnavailable as exc:
            if len(rows) == 1:
                store.record_failure(conn, embedder.model_id, rows, str(exc))
                return {"embedded": embedded, "status": "failed", "error": str(exc)}
            # Isolate the failing chunks so one bad text cannot block its whole batch.
            failed = None
            for row in rows:
                try:
                    store.save_vectors(conn, embedder.model_id, [row], embedder.embed([row["text"]], kind="passage"))
                    embedded += 1
                except EmbeddingUnavailable as single:
                    store.record_failure(conn, embedder.model_id, [row], str(single))
                    failed = str(single)
            if failed and embedded == 0:
                return {"embedded": 0, "status": "failed", "error": failed}
            continue
        store.save_vectors(conn, embedder.model_id, rows, vectors)
        embedded += len(rows)
    return {"embedded": embedded, "status": "partial"}


class BackgroundSync:
    """Runs per-session sync and embedding off the response path.

    Requests coalesce; a non-blocking file lock keeps a second process from
    duplicating work (a skipped run is repaired by the next reconcile pass).
    """

    def __init__(self, open_store, state_db: Path, policy: ProcessingPolicy, embedder_factory, lock_path: Path):
        self._open_store = open_store
        self._state_db = state_db
        self._policy = policy
        self._embedder_factory = embedder_factory
        self._lock_path = lock_path
        self._pending: set[str] = set()
        self._cond = threading.Condition()
        self._thread: threading.Thread | None = None
        self.backfill_step = 25

    def request(self, session_id: str | None) -> None:
        if not session_id:
            return
        with self._cond:
            self._pending.add(session_id)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="hermes-mem-sync", daemon=True)
                self._thread.start()
            self._cond.notify()

    def wait_idle(self, timeout: float = 30.0) -> bool:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
            return not thread.is_alive()
        return True

    def _run(self) -> None:
        while True:
            with self._cond:
                if not self._pending:
                    self._thread = None
                    return
                batch, self._pending = sorted(self._pending), set()
            try:
                self._lock_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._lock_path, "w") as handle:
                    for _ in range(20):
                        try:
                            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            time.sleep(0.5)
                    else:
                        log.info("hermes-mem: sync lock busy; %s left for the next reconcile pass", batch)
                        continue
                    conn = self._open_store()
                    try:
                        sync(conn, self._state_db, policy=self._policy, session_ids=batch)
                        # One bounded backfill step per turn replays the whole history over time,
                        # repairing turns whose hooks never fired.
                        sync(conn, self._state_db, policy=self._policy, max_sessions=self.backfill_step)
                        embed_pending(conn, self._embedder_factory(), max_batches=10)
                    finally:
                        conn.close()
            except Exception:
                log.exception("hermes-mem background sync failed for %s", batch)
