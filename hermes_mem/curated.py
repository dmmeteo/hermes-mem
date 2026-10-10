"""Curated records: one shape for peer cards, intents, observations and decisions, stored in mem.db.

A record's content never changes after insert (store.SCHEMA enforces it with triggers). A revision
inserts a successor and supersedes the old record; accept and retract change only the status. Every
change appends a `curated_events` row in the same transaction, so the event log is the provenance.
Markdown is only an explicit import/export format (curated_md.py), never a second store.
"""

from __future__ import annotations

import json
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterable

KINDS = ("peer", "intent", "observation", "decision")
STATUSES = ("proposed", "accepted", "superseded", "retracted")
ACTIVE_STATUSES = ("proposed", "accepted")
BASES = ("declared", "inferred")
CONFIDENCES = ("low", "medium", "high")
AUTHORS = ("agent", "operator")
EVIDENCE_RE = re.compile(r"^(episode:\d+|session:[^#\s]+#msg:\d+)$")
ID_RE = re.compile(r"^(peer|intent|observation|decision)-\d{8}-[0-9a-f]{6}$")
MAX_STATEMENT_CHARS = 4000
MAX_FIELD_CHARS = {"subject": 200, "note": 1000}
MAX_EVIDENCE = 20
# What a record says and who said it. Two records with equal content are the same statement.
CONTENT_FIELDS = ("id", "kind", "scope", "subject", "statement", "basis", "confidence", "evidence", "author",
                  "supersedes", "created_at")
REVIEW_TARGETS = {"accept": "accepted", "retract": "retracted"}
# Statuses a record can reach from each status through the store's own operations.
LATER_STATUSES = {"proposed": {"accepted", "retracted", "superseded"}, "accepted": {"retracted", "superseded"},
                  "superseded": set(), "retracted": set()}


class CuratedError(ValueError):
    pass


@dataclass
class Record:
    id: str
    kind: str
    scope: str
    subject: str
    statement: str
    status: str = "proposed"
    basis: str = "inferred"
    confidence: str = "low"
    evidence: list[str] = field(default_factory=list)
    author: str = "agent"
    supersedes: str | None = None
    superseded_by: str | None = None
    note: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0
    reviewed_at: float | None = None

    def content(self) -> tuple:
        return tuple(getattr(self, name) for name in CONTENT_FIELDS)

    def validate(self) -> None:
        checks = [
            (self.kind in KINDS, f"kind must be one of {KINDS}"),
            (self.status in STATUSES, f"status must be one of {STATUSES}"),
            (self.basis in BASES, f"basis must be one of {BASES}"),
            (self.confidence in CONFIDENCES, f"confidence must be one of {CONFIDENCES}"),
            (self.author in AUTHORS, f"author must be one of {AUTHORS}"),
            (bool(self.scope), "scope is required"),
            (bool(self.subject.strip()), "subject is required"),
            (bool(self.statement.strip()), "statement is required"),
            (len(self.statement) <= MAX_STATEMENT_CHARS, f"statement exceeds {MAX_STATEMENT_CHARS} chars"),
            (len(self.evidence) <= MAX_EVIDENCE, f"at most {MAX_EVIDENCE} evidence refs"),
            (ID_RE.match(self.id) is not None, "malformed record id"),
            (self.id.split("-", 1)[0] == self.kind, "record id does not match its kind"),
            ((self.status == "superseded") == (self.superseded_by is not None), "superseded_by is set exactly when superseded"),
        ] + [
            (len(getattr(self, name) or "") <= limit, f"{name} exceeds {limit} chars")
            for name, limit in MAX_FIELD_CHARS.items()
        ]
        for ok, message in checks:
            if not ok:
                raise CuratedError(f"{self.id}: {message}")
        for ref in self.evidence:
            if not EVIDENCE_RE.match(ref):
                raise CuratedError(f"{self.id}: malformed evidence ref {ref!r}")
        if self.basis == "declared" and not self.evidence:
            raise CuratedError(f"{self.id}: a declared statement needs evidence of the declaration")


COLUMNS = tuple(Record.__dataclass_fields__)


def _from_row(row: sqlite3.Row) -> Record:
    data = dict(row)
    data["evidence"] = json.loads(data["evidence"])
    return Record(**data)


EvidenceCheck = Callable[[str], "str | None"]
"""Returns the author role of an evidence ref ("user"/"assistant") or None when unknown."""


class CuratedStore:
    """Curated records in an open mem.db connection (autocommit; each change is one IMMEDIATE transaction)."""

    def __init__(self, conn: sqlite3.Connection, evidence_check: EvidenceCheck | None = None):
        self.conn = conn
        self.evidence_check = evidence_check

    @contextmanager
    def transaction(self):
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.conn.execute("COMMIT")  # deferred foreign keys are checked here
        except BaseException as exc:
            self.conn.execute("ROLLBACK")
            if isinstance(exc, sqlite3.IntegrityError):
                raise CuratedError(f"curated store rejected the change: {exc}") from None
            raise

    def insert(self, record: Record, *, action: str, actor: str, source: str | None = None) -> None:
        """Insert a validated record and its first event. Callers hold a transaction."""
        record.validate()
        values = [json.dumps(record.evidence) if name == "evidence" else getattr(record, name) for name in COLUMNS]
        try:
            self.conn.execute(
                f"INSERT INTO curated_records({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})", values
            )
        except sqlite3.IntegrityError as exc:
            raise CuratedError(f"{record.id}: {exc}") from None
        self._event(record.id, action, actor, record.status, record.note, source)

    def _event(self, record_id: str, action: str, author: str, status: str, note: str | None = None,
               source: str | None = None) -> None:
        self.conn.execute(
            "INSERT INTO curated_events(record_id, action, author, status, note, source, at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (record_id, action, author, status, note, source, time.time()),
        )

    def check_evidence(self, basis: str, evidence: Iterable[str]) -> None:
        evidence = list(evidence)
        for ref in evidence:
            if not EVIDENCE_RE.match(str(ref)):
                raise CuratedError(f"malformed evidence ref {ref!r}")
        if self.evidence_check is None:
            return
        roles = {}
        for ref in evidence:
            role = self.evidence_check(ref)
            if role is None:
                raise CuratedError(f"evidence {ref!r} does not exist in the episode store")
            roles[ref] = role
        if basis == "declared" and "user" not in roles.values():
            raise CuratedError("a declared statement needs evidence from a user-authored message")

    def find(self, record_id: str) -> Record | None:
        row = self.conn.execute("SELECT * FROM curated_records WHERE id=?", (record_id,)).fetchone()
        return _from_row(row) if row else None

    def get(self, record_id: str) -> Record:
        if not ID_RE.match(record_id or ""):
            raise CuratedError(f"malformed record id {record_id!r}")
        record = self.find(record_id)
        if record is None:
            raise CuratedError(f"no curated record {record_id!r}")
        return record

    def all(self) -> list[Record]:
        return [_from_row(r) for r in self.conn.execute("SELECT * FROM curated_records ORDER BY created_at, id")]

    def counts(self) -> dict[str, int]:
        return {r[0]: r[1] for r in self.conn.execute("SELECT status, COUNT(*) FROM curated_records GROUP BY status")}

    def list(
        self,
        *,
        scopes: Iterable[str] | None,
        kind: str | None = None,
        subject: str | None = None,
        include_inactive: bool = False,
        limit: int = 20,
    ) -> list[Record]:
        where, args = [], []
        if scopes is not None:
            scopes = list(scopes)
            where.append(f"scope IN ({', '.join('?' * len(scopes))})" if scopes else "0")
            args += scopes
        for column, value in (("kind", kind), ("subject", subject)):
            if value is not None:
                where.append(f"{column}=?")
                args.append(value)
        if not include_inactive:
            where.append("status IN ('proposed', 'accepted')")
        rows = self.conn.execute(
            f"SELECT * FROM curated_records {'WHERE ' + ' AND '.join(where) if where else ''} "
            "ORDER BY status != 'accepted', updated_at DESC, id LIMIT ?",
            [*args, max(1, min(limit, 100))],
        )
        return [_from_row(r) for r in rows]

    def history(self, record_id: str) -> dict:
        """The record's whole revision chain, oldest first, and every event on it."""
        record, seen = self.get(record_id), {record_id}
        while record.supersedes and record.supersedes not in seen:
            seen.add(record.supersedes)
            record = self.get(record.supersedes)
        chain = [record]
        while chain[-1].superseded_by and chain[-1].superseded_by not in {r.id for r in chain}:
            chain.append(self.get(chain[-1].superseded_by))
        ids = [r.id for r in chain]
        events = self.conn.execute(
            f"SELECT record_id, action, author, status, note, source, at FROM curated_events "
            f"WHERE record_id IN ({', '.join('?' * len(ids))}) ORDER BY id",
            ids,
        ).fetchall()
        return {"chain": chain, "events": [dict(e) for e in events]}

    def _build(
        self,
        *,
        kind: str,
        scope: str,
        subject: str,
        statement: str,
        basis: str = "inferred",
        confidence: str = "low",
        evidence: Iterable[str] = (),
        author: str = "agent",
        note: str | None = None,
        supersedes: str | None = None,
    ) -> Record:
        if kind not in KINDS:
            raise CuratedError(f"kind must be one of {KINDS}")
        evidence = list(dict.fromkeys(evidence))
        self.check_evidence(basis, evidence)
        now = time.time()
        record = Record(
            id=f"{kind}-{time.strftime('%Y%m%d', time.gmtime(now))}-{secrets.token_hex(3)}",
            kind=kind,
            scope=scope,
            subject=subject,
            statement=statement.strip(),
            status="accepted" if author == "operator" else "proposed",
            basis=basis,
            confidence=confidence,
            evidence=evidence,
            author=author,
            note=note,
            supersedes=supersedes,
            created_at=now,
            updated_at=now,
            reviewed_at=now if author == "operator" else None,
        )
        record.validate()
        return record

    def propose(self, **fields) -> Record:
        record = self._build(**fields)
        with self.transaction():
            self.insert(record, action="add" if record.author == "operator" else "propose", actor=record.author)
        return record

    def revise(self, record_id: str, *, statement: str, author: str, **changes) -> Record:
        """Insert a successor and supersede the old record in one transaction (at most one successor)."""
        with self.transaction():
            old = self.get(record_id)
            self._check_mutable(old, author)
            new = self._build(
                kind=old.kind,
                scope=old.scope,
                subject=old.subject,
                statement=statement,
                basis=changes.get("basis") or old.basis,
                confidence=changes.get("confidence") or old.confidence,
                evidence=changes.get("evidence") if changes.get("evidence") is not None else old.evidence,
                author=author,
                note=changes.get("note"),
                supersedes=old.id,
            )
            self.insert(new, action="revise", actor=author)
            self.conn.execute(
                "UPDATE curated_records SET status='superseded', superseded_by=?, updated_at=? WHERE id=?",
                (new.id, new.created_at, old.id),
            )
            self._event(old.id, "supersede", author, "superseded", f"superseded by {new.id}")
        return new

    @staticmethod
    def _check_mutable(record: Record, author: str) -> None:
        if record.status not in ACTIVE_STATUSES:
            raise CuratedError(f"{record.id} is {record.status}; revise its successor instead")
        if author != "operator" and record.status == "accepted":
            raise CuratedError(f"{record.id} was accepted by the operator; propose a new statement instead")

    def review(self, record_id: str, *, action: str, author: str, note: str | None = None) -> Record:
        if action == "accept" and author != "operator":
            raise CuratedError("only the operator can accept a curated statement")
        target = REVIEW_TARGETS.get(action)
        if target is None:
            raise CuratedError("review action must be accept or retract")
        with self.transaction():
            record = self.get(record_id)
            self._check_mutable(record, author)
            now = time.time()
            record.status, record.updated_at, record.reviewed_at = target, now, now
            record.note = note or record.note
            self.conn.execute(
                "UPDATE curated_records SET status=?, updated_at=?, reviewed_at=?, note=? WHERE id=?",
                (target, now, now, record.note, record.id),
            )
            self._event(record.id, action, author, target, note)
        return record
