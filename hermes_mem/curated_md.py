"""Explicit Markdown import and export of curated records (the pre-SQLite on-disk format).

Layout: <root>/<kind>/<record_id>.md with YAML front matter. Nothing reads this format implicitly;
mem.db is the only authoritative store. An import is planned and applied under the store's write
lock, publishes all records in one transaction or none, is repeatable, and never modifies or
deletes the files it reads. The stored state always wins: a file can add records, never change them.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .curated import KINDS, LATER_STATUSES, CuratedError, CuratedStore, Record

TEXT_FIELDS = ("id", "kind", "scope", "subject", "status", "basis", "confidence", "author")
OPTIONAL_TEXT_FIELDS = ("supersedes", "superseded_by", "note")
TIME_FIELDS = ("created_at", "updated_at")


def dump(record: Record) -> str:
    meta = {k: v for k, v in record.__dict__.items() if k != "statement" and v not in (None, [])}
    head = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False).strip()
    return f"---\n{head}\n---\n{record.statement.strip()}\n"


def _check_types(meta: dict) -> None:
    for name in TEXT_FIELDS + OPTIONAL_TEXT_FIELDS:
        value = meta.get(name)
        if value is not None and not isinstance(value, str):
            raise CuratedError(f"{name} must be a string, got {value!r}")
    for name in (*TIME_FIELDS, "reviewed_at"):
        value = meta.get(name)
        if (value is not None or name in TIME_FIELDS) and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise CuratedError(f"{name} must be a number, got {value!r}")
        if value is not None:
            meta[name] = float(value)
    evidence = meta.get("evidence") or []
    if not isinstance(evidence, list) or not all(isinstance(ref, str) for ref in evidence):
        raise CuratedError(f"evidence must be a list of strings, got {evidence!r}")
    meta["evidence"] = evidence


def parse(path: Path, text: str) -> Record:
    match = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.S)
    if not match:
        raise CuratedError("missing front matter")
    meta = yaml.safe_load(match.group(1)) or {}
    if not isinstance(meta, dict):
        raise CuratedError("front matter is not a mapping")
    unknown = set(meta) - (set(Record.__dataclass_fields__) - {"statement"})
    if unknown:
        raise CuratedError(f"unknown fields {sorted(map(str, unknown))}")
    meta.setdefault("id", path.stem)
    meta.setdefault("kind", path.parent.name)
    _check_types(meta)
    record = Record(statement=match.group(2).strip(), **meta)
    if record.id != path.stem:
        raise CuratedError(f"id {record.id!r} does not match the file name")
    if record.kind != path.parent.name:
        raise CuratedError(f"kind {record.kind!r} does not match its directory")
    record.validate()
    return record


def _summary(record: Record) -> dict:
    """What the operator should review before publishing: who claims what, with which authority."""
    return {"id": record.id, "kind": record.kind, "scope": record.scope, "subject": record.subject,
            "status": record.status, "author": record.author, "basis": record.basis, "evidence": record.evidence}


@dataclass
class ImportPlan:
    source: Path
    new: list[tuple[Record, str]] = field(default_factory=list)  # (record, provenance)
    unchanged: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.conflicts and not self.invalid

    def as_dict(self) -> dict:
        return {
            "source": str(self.source),
            "ok": self.ok,
            "new": [_summary(r) for r, _ in self.new],
            "unchanged": self.unchanged,
            "conflicts": self.conflicts,
            "invalid": self.invalid,
            "ignored": self.ignored,
        }


def _compare(existing: Record, record: Record) -> str | None:
    """None when the stored record is this file's record, as written or moved on by the store; else why not."""
    if existing.content() != record.content():
        return f"record {record.id} is already stored with different content"
    if existing.status == record.status:
        if existing.superseded_by != record.superseded_by:
            return f"stored {record.id} is superseded by {existing.superseded_by}, the file says {record.superseded_by}"
        return None
    if existing.status in LATER_STATUSES[record.status]:
        return None  # the store accepted, retracted or revised it after the file was written
    return f"stored {record.id} is {existing.status}, the file says {record.status}"


def plan_import(store: CuratedStore, root: Path) -> ImportPlan:
    """Read-only: classify every file as new, unchanged, conflict or invalid."""
    plan = ImportPlan(source=root)
    if not root.is_dir():
        plan.invalid.append(f"{root}: not a directory")
        return plan
    found: dict[str, tuple[Record, str]] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_dir() or rel == ".lock" or rel.endswith(".md.tmp"):
            continue
        if path.suffix != ".md" or path.parent.parent != root or path.parent.name not in KINDS:
            plan.ignored.append(rel)
            continue
        try:
            raw = path.read_bytes()
            record = parse(path, raw.decode("utf-8"))
        except (CuratedError, yaml.YAMLError, UnicodeDecodeError, OSError, TypeError, ValueError) as exc:
            plan.invalid.append(f"{rel}: {exc}")
            continue
        found[record.id] = (record, rel)
        existing = store.find(record.id)
        if existing is None:
            try:
                store.check_evidence(record.basis, record.evidence)
            except CuratedError as exc:
                plan.invalid.append(f"{rel}: {exc}")
                continue
            plan.new.append((record, f"{rel}#sha256={hashlib.sha256(raw).hexdigest()}"))
        else:
            reason = _compare(existing, record)
            if reason:
                plan.conflicts.append(f"{rel}: {reason}")
            else:
                plan.unchanged.append(record.id)

    def view(record_id: str) -> Record | None:
        """The stored row wins over a file: links are checked against what the store will hold."""
        return store.find(record_id) or (found[record_id][0] if record_id in found else None)

    for record, rel in found.values():
        for name, link, back in (("supersedes", record.supersedes, "superseded_by"),
                                 ("superseded_by", record.superseded_by, "supersedes")):
            if link is None:
                continue
            if link == record.id:
                plan.invalid.append(f"{rel}: {record.id} {name} itself")
                continue
            target = view(link)
            if target is None:
                plan.invalid.append(f"{rel}: {name} {link} is neither imported nor stored")
            elif getattr(target, back) != record.id:
                plan.invalid.append(f"{rel}: {name} {link}, but {link} has {back} {getattr(target, back)}")
        seen, step = {record.id}, view(record.id)
        while step is not None and step.supersedes and step.supersedes != step.id:
            if step.supersedes in seen:
                plan.invalid.append(f"{rel}: {record.id} is part of a supersession cycle")
                break
            seen.add(step.supersedes)
            step = view(step.supersedes)
    plan.invalid = sorted(set(plan.invalid))
    return plan


class _Blocked(Exception):
    def __init__(self, plan: ImportPlan):
        self.plan = plan


def apply_import(store: CuratedStore, root: Path) -> dict:
    """Plan again under the write lock and publish every new record in one transaction, or nothing."""
    try:
        with store.transaction():
            plan = plan_import(store, root)
            if not plan.ok:
                raise _Blocked(plan)
            for record, provenance in plan.new:
                store.insert(record, action="import", actor="operator", source=provenance)
    except _Blocked as blocked:
        raise CuratedError(f"import blocked: {len(blocked.plan.conflicts)} conflict(s), {len(blocked.plan.invalid)} "
                           f"invalid file(s); nothing was imported. Run with --dry-run to see them.") from None
    except CuratedError as exc:
        raise CuratedError(f"import rolled back, nothing was imported: {exc}") from None
    return {**plan.as_dict(), "imported": len(plan.new)}


def export(store: CuratedStore, dest: Path) -> dict:
    """Write every record (all statuses) as Markdown into a new or empty directory, all at once."""
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise CuratedError(f"{dest} exists and is not an empty directory")
    dest.parent.mkdir(parents=True, exist_ok=True)
    records = store.all()
    staging = Path(tempfile.mkdtemp(prefix=f".{dest.name}.export-", dir=dest.parent))
    try:
        for record in records:
            path = staging / record.kind / f"{record.id}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(dump(record), encoding="utf-8")
        if dest.exists():
            dest.rmdir()
        os.replace(staging, dest)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"exported": len(records), "to": str(dest)}
