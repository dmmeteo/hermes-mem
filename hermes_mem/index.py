"""Operator indexing: preview selection read-only, then select -> episodes -> embeddings in bounded batches.

Reuses sync.reconcile/embed_pending; bounds only narrow which sessions are visited. Eligibility,
visibility and authoritative scope rules are applied exactly as the background sync does.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .host import HostSession, eligibility, open_state, session_scope
from .select import ProcessingPolicy, build_episodes, visible_text


def parse_date(text: str | None) -> float | None:
    if not text:
        return None
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        raise ValueError(f"dates are ISO 8601 (2026-10-01 or 2026-10-01T12:00), got {text!r}") from None
    return (moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)).timestamp()


@dataclass
class Bounds:
    sessions: list[str] | None = None
    sources: list[str] | None = None
    since: float | None = None
    until: float | None = None

    def admits(self, s: HostSession) -> bool:
        if self.sources and s.source not in self.sources:
            return False
        started = s.started_at or 0.0
        if self.since is not None and started < self.since:
            return False
        return self.until is None or started < self.until


def bounded_sessions(state_db: Path, bounds: Bounds) -> list[HostSession]:
    with open_state(state_db) as host:
        found = host.sessions(ids=bounds.sessions) if bounds.sessions else host.sessions(limit=10**9)
    return [s for s in found if bounds.admits(s)]


@dataclass
class Preview:
    sessions_considered: int = 0
    sessions_selected: int = 0
    sessions_excluded: Counter = field(default_factory=Counter)
    episodes: int = 0
    chunks: int = 0
    messages_excluded: Counter = field(default_factory=Counter)
    scopes: Counter = field(default_factory=Counter)
    samples: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "dry_run": True,
            "sessions_considered": self.sessions_considered,
            "sessions_selected": self.sessions_selected,
            "sessions_excluded": dict(self.sessions_excluded),
            "episodes": self.episodes,
            "chunks": self.chunks,
            "messages_excluded": dict(self.messages_excluded),
            "scopes": dict(self.scopes),
            **({"samples": self.samples} if self.samples else {}),
        }


def preview(state_db: Path, bounds: Bounds, policy: ProcessingPolicy, *, excerpts: int = 0, excerpt_chars: int = 160) -> dict:
    """Read-only: opens state.db with mode=ro and writes nothing anywhere."""
    out = Preview()
    if not state_db.exists():
        return {"dry_run": True, "error": f"state.db not found at {state_db}"}
    with open_state(state_db) as host:
        sessions = host.sessions(ids=bounds.sessions) if bounds.sessions else host.sessions(limit=10**9)
        for session in sessions:
            if not bounds.admits(session):
                continue
            out.sessions_considered += 1
            skip = eligibility(session)
            if skip:
                out.sessions_excluded[skip] += 1
                continue
            out.sessions_selected += 1
            scope = session_scope(session)
            messages = host.visible_messages(session.id)
            for m in messages:
                if visible_text(m) is None:
                    reason = ("compaction summary" if m.summary else "assistant tool call" if m.has_tool_calls
                              else "empty" if not (m.content or "").strip() else "log/trace dump")
                    out.messages_excluded[reason] += 1
            episodes = policy.apply(build_episodes(scope, messages))
            out.episodes += len(episodes)
            out.chunks += sum(len(e.chunks) for e in episodes)
            out.scopes[scope.split(":", 1)[0]] += len(episodes)
            for ep in episodes[: max(0, excerpts - len(out.samples))]:
                out.samples.append({
                    "session_id": session.id, "scope": scope, "message_ids": [ep.first_message_id, ep.last_message_id],
                    "user": ep.user_text[:excerpt_chars], "assistant": ep.assistant_text[:excerpt_chars],
                })
    result = out.as_dict()
    result["note"] = "Host rows outside (active=1 OR compacted=1), tool/system roles and observed chatter are never read."
    return result


def run(conn, state_db: Path, bounds: Bounds, policy: ProcessingPolicy, embedder, *, batch_sessions: int,
        embed_batch: int, max_batches: int | None, progress=None) -> dict:
    """Reconcile bounded sessions in batches, then embed pending chunks. Safe to interrupt and repeat."""
    from .sync import SyncReport, embed_pending, sync

    ids = [s.id for s in bounded_sessions(state_db, bounds)] if state_db.exists() else []
    totals = SyncReport(processing=policy.describe())
    started = time.monotonic()
    batches = 0
    interrupted = False
    try:
        for i in range(0, len(ids), batch_sessions):
            if max_batches is not None and batches >= max_batches:
                break
            report = sync(conn, state_db, policy=policy, session_ids=ids[i:i + batch_sessions])
            for key in ("sessions", "skipped", "unchanged", "inserted", "revised", "rechunked", "retracted", "source_missing"):
                setattr(totals, key, getattr(totals, key) + getattr(report, key))
            totals.warnings += report.warnings
            batches += 1
            if progress:
                progress({"stage": "select", "sessions_done": min(i + batch_sessions, len(ids)), "sessions_total": len(ids)})
        embedding = {"embedded": 0, "status": "disabled"}
        if embedder is not None:
            embedding = embed_pending(conn, embedder, batch_size=embed_batch, max_batches=10**6 if max_batches is None else max_batches)
            if progress:
                progress({"stage": "embed", **embedding})
    except KeyboardInterrupt:
        interrupted = True
        embedding = {"status": "interrupted"}
    remaining = len(ids) - min(len(ids), batches * batch_sessions)
    return {
        "sessions_in_bounds": len(ids),
        "batches": batches,
        "sessions_remaining": remaining,
        "selection": {k: v for k, v in totals.as_dict().items() if k not in ("complete", "identity_mode")},
        "embedding": embedding,
        "space": embedder.model_id if embedder is not None else None,
        "seconds": round(time.monotonic() - started, 2),
        "interrupted": interrupted,
        "resume": "re-run the same command; unchanged sessions and existing vectors are skipped"
        if interrupted or remaining or embedding.get("status") in ("partial", "failed") else None,
    }
