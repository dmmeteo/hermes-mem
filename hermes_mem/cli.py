"""Operator CLI: `hermes hermes-mem ...` inside Hermes, or `python -m hermes_mem --home PATH ...`."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from . import curated_md, index, store
from .app import OWNER, ConfigError, MemApp
from .config import space_of
from .curated import BASES, CONFIDENCES, KINDS, CuratedError
from .runtime import setup_runtime
from .search import search
from .sync import embed_pending, sync


def setup(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="mem_command", required=True)

    p = sub.add_parser("sync", help="Reconcile episodes from state.db (bounded, resumable, idempotent)")
    p.add_argument("--max-sessions", type=int, default=None)
    p.add_argument("--session", action="append", help="Only these session ids")
    p.add_argument("--all", action="store_true", help="Repeat bounded batches until a full pass completes")
    p.add_argument("--embed", action="store_true", help="Also embed pending chunks")

    p = sub.add_parser("embed", help="Embed pending chunks with the active (or given) model profile")
    p.add_argument("--max-batches", type=int, default=50)
    p.add_argument("--profile", help="Embedding profile; default embedding.active")

    p = sub.add_parser("setup", help="Provision the locked external embedding engine runtime (idempotent)")
    p.add_argument("--python", help="Interpreter to build the engine venv from (default: the host's base Python)")
    p.add_argument("--force", action="store_true", help="Rebuild; the old runtime stays until the new one works")

    p = sub.add_parser("index", help="Select history -> durable episodes -> embeddings (bounded, resumable)")
    p.add_argument("--dry-run", action="store_true", help="Read-only preview of selected/excluded counts and reasons")
    p.add_argument("--excerpts", type=int, default=0, help="With --dry-run, show up to N bounded sample episodes")
    p.add_argument("--session", action="append", help="Only these session ids")
    p.add_argument("--source", action="append", help="Only sessions from these sources (cli, telegram, ...)")
    p.add_argument("--since", help="Sessions started at/after this ISO date")
    p.add_argument("--until", help="Sessions started before this ISO date")
    p.add_argument("--batch-sessions", type=int, default=50)
    p.add_argument("--max-batches", type=int, help="Stop after N batches (re-run to resume)")
    p.add_argument("--profile", help="Embedding profile; default embedding.active")
    p.add_argument("--no-embed", action="store_true", help="Only select and store episodes")

    p = sub.add_parser("search", help="Search episodes as the profile owner")
    p.add_argument("query")
    p.add_argument("--mode", choices=["hybrid", "keyword", "semantic"], default="hybrid")
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--scope", help="Restrict to one scope, e.g. peer:telegram:123")
    p.add_argument("--conversation", help="Recent turns for a follow-up question")
    p.add_argument("--variant", action="append", default=[], help="Semantic variant (translation/transliteration); repeatable")
    p.add_argument("--profile", help="Embedding profile; default embedding.active")

    sub.add_parser("status", help="Index, embedding and source status")

    p = sub.add_parser("history", help="All versions of an episode")
    p.add_argument("episode_id", type=int)

    cur = sub.add_parser("curated", help="Curated peer cards, intents, observations, decisions")
    csub = cur.add_subparsers(dest="curated_command", required=True)
    c = csub.add_parser("list")
    c.add_argument("--kind", choices=KINDS)
    c.add_argument("--subject")
    c.add_argument("--all", action="store_true", help="Include superseded and retracted")
    c = csub.add_parser("show")
    c.add_argument("id")
    c = csub.add_parser("add", help="Add an operator-declared (accepted) statement")
    c.add_argument("kind", choices=KINDS)
    c.add_argument("subject")
    c.add_argument("statement")
    c.add_argument("--scope", default="owner")
    c.add_argument("--basis", choices=BASES, default="declared")
    c.add_argument("--confidence", choices=CONFIDENCES, default="high")
    c.add_argument("--evidence", action="append", default=[])
    c = csub.add_parser("revise")
    c.add_argument("id")
    c.add_argument("statement")
    c.add_argument("--evidence", action="append")
    c.add_argument("--confidence", choices=CONFIDENCES)
    c = csub.add_parser("review", help="Accept or retract a record")
    c.add_argument("id")
    group = c.add_mutually_exclusive_group(required=True)
    group.add_argument("--accept", action="store_true")
    group.add_argument("--retract", action="store_true")
    c.add_argument("--note")
    c = csub.add_parser("history", help="A record's revision chain and every change to it")
    c.add_argument("id")
    c = csub.add_parser("import", help="Import Markdown records (previous on-disk format); all or nothing, repeatable")
    c.add_argument("--from", dest="source", type=Path, help="Directory of <kind>/<id>.md files (default: <data dir>/curated)")
    c.add_argument("--dry-run", action="store_true", help="Preview new/unchanged/conflicting/invalid files; write nothing")
    c = csub.add_parser("export", help="Write all records as Markdown into a new or empty directory")
    c.add_argument("--to", dest="dest", type=Path, required=True)


def _print(data) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def run(args: argparse.Namespace, app: MemApp | None = None) -> int:
    try:
        app = app or MemApp(getattr(args, "home", None))
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    command = args.mem_command
    if command == "setup":
        result = setup_runtime(app.runtime, python=args.python, force=args.force)
        _print(result)
        return 0 if result["status"] != "failed" else 1
    if command == "index" and args.dry_run:
        try:
            bounds = _bounds(args)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        _print(index.preview(app.state_db, bounds, app.policy, excerpts=max(0, min(args.excerpts, 20))))
        return 0
    conn = app.open_store()
    try:
        command = args.mem_command
        if command == "sync":
            max_sessions = args.max_sessions or app.config["sync"]["max_sessions"]
            reports = []
            while True:
                report = sync(conn, app.state_db, policy=app.policy, session_ids=args.session, max_sessions=max_sessions)
                reports.append(report.as_dict())
                if args.session or not args.all or report.complete:
                    break
            out = {"passes": reports}
            if args.embed:
                out["embedding"] = embed_pending(conn, app.embedder())
            _print(out)
            return 0
        if command == "embed":
            embedder = app.embedder(args.profile)
            result = embed_pending(conn, embedder, batch_size=app.config["embedding"]["batch_size"], max_batches=args.max_batches)
            result["space"] = embedder.model_id if embedder else None
            _print(result)
            return 0 if result["status"] != "failed" else 1
        if command == "index":
            try:
                bounds = _bounds(args)
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            result = index.run(
                conn, app.state_db, bounds, app.policy, None if args.no_embed else app.embedder(args.profile),
                batch_sessions=max(1, args.batch_sessions), embed_batch=app.config["embedding"]["batch_size"],
                max_batches=args.max_batches,
                progress=lambda event: print(json.dumps(event, ensure_ascii=False), file=sys.stderr),
            )
            _print(result)
            if result["interrupted"]:
                return 130
            return 1 if result["embedding"].get("status") == "failed" else 0
        if command == "search":
            params = app.search_params(args.query, [args.scope] if args.scope else None, mode=args.mode,
                                       limit=args.limit, conversation=args.conversation, variants=args.variant)
            _print(search(conn, app.state_db, app.embedder(args.profile), params))
            return 0
        if command == "status":
            _print(status(conn, app))
            return 0
        if command == "history":
            key = conn.execute("SELECT episode_key FROM episodes WHERE id=?", (args.episode_id,)).fetchone()
            if key is None:
                print(f"no episode {args.episode_id}", file=sys.stderr)
                return 1
            rows = conn.execute(
                """SELECT id, version, status, status_reason, source_state, session_id, superseded_by, content_hash,
                          substr(user_text, 1, 200) AS user, substr(assistant_text, 1, 200) AS assistant
                   FROM episodes WHERE episode_key=? ORDER BY version""",
                (key[0],),
            ).fetchall()
            _print([dict(r) for r in rows])
            return 0
        if command == "curated":
            return _curated(args, app, conn)
        raise AssertionError(command)
    except (CuratedError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        conn.close()
        app.close()


def _bounds(args) -> index.Bounds:
    return index.Bounds(sessions=args.session, sources=args.source,
                        since=index.parse_date(args.since), until=index.parse_date(args.until))


def _curated(args, app: MemApp, conn) -> int:
    curated = app.curated(conn, OWNER)
    sub = args.curated_command
    if sub == "list":
        records = curated.list(scopes=None, kind=args.kind, subject=args.subject, include_inactive=args.all, limit=100)
        _print({"records": [asdict(r) for r in records]})
    elif sub == "show":
        _print(asdict(curated.get(args.id)))
    elif sub == "add":
        record = curated.propose(
            kind=args.kind, scope=args.scope, subject=args.subject, statement=args.statement,
            basis=args.basis, confidence=args.confidence, evidence=args.evidence, author="operator",
        )
        _print(asdict(record))
    elif sub == "revise":
        _print(asdict(curated.revise(args.id, statement=args.statement, author="operator",
                                     evidence=args.evidence, confidence=args.confidence)))
    elif sub == "review":
        action = "accept" if args.accept else "retract"
        _print(asdict(curated.review(args.id, action=action, author="operator", note=args.note)))
    elif sub == "history":
        history = curated.history(args.id)
        _print({"chain": [asdict(r) for r in history["chain"]], "events": history["events"]})
    elif sub == "import":
        source = (args.source or app.legacy_curated_dir).expanduser()
        if args.dry_run:
            plan = curated_md.plan_import(curated, source)
            _print({"dry_run": True, **plan.as_dict()})
            return 0 if plan.ok else 2
        _print(curated_md.apply_import(curated, source))
    elif sub == "export":
        _print(curated_md.export(curated, args.dest.expanduser()))
    return 0


def status(conn, app: MemApp) -> dict:
    """Effective settings, coverage of every configured profile and stored space, runtime health. No side effects."""
    embedder = app.embedder()
    count = lambda sql, *a: conn.execute(sql, a).fetchone()[0]  # noqa: E731
    cursor = conn.execute("SELECT value FROM meta WHERE key='backfill_cursor'").fetchone()
    chunks = store.current_chunk_count(conn)
    spaces = store.space_coverage(conn)
    stored = {space["model_id"]: space for space in spaces}
    failures = store.failure_summary(conn)
    profiles = {}
    for name, profile in app.config["embedding"]["profiles"].items():
        space_id = space_of(profile).space_id
        vectors = stored.get(space_id, {}).get("current_vectors", 0)
        profiles[name] = {"space_id": space_id, "active": name == app.active_profile, "current_vectors": vectors,
                          "pending": chunks - vectors, "failed": failures.get(space_id, {}).get("failed", 0)}
        if space_id in failures:
            profiles[name]["last_error"] = failures[space_id]["last_error"]
    configured = {p["space_id"] for p in profiles.values()}
    for space in spaces:
        space["pending"] = chunks - space["current_vectors"]
        space["configured"] = space["model_id"] in configured
    out = {
        "home": str(app.home),
        "store": str(app.data_dir / "mem.db"),
        "state_db_present": app.state_db.exists(),
        "episodes": {
            row["status"]: row["n"]
            for row in conn.execute("SELECT status, COUNT(*) AS n FROM episodes GROUP BY status")
        },
        "source_missing": count("SELECT COUNT(*) FROM episodes WHERE status='current' AND source_state='missing'"),
        "sessions_skipped": count("SELECT COUNT(*) FROM sources WHERE eligible=0"),
        "backfill_cursor": cursor[0] if cursor else "",
        "active_profile": app.active_profile,
        "embedding_model": embedder.model_id if embedder else None,
        "active_space": embedder.space.describe() if embedder else None,
        "chunks_current": chunks,
        "profiles": profiles,
        "spaces": spaces,
        "engine": app.runtime.diagnose(),
        "model_cache": app.runtime.cache_dir,
        "processing": app.policy.describe(),
        "curated": curated_status(conn, app),
        "settings": {k: v for k, v in app.config.items() if k != "processing"},
    }
    out["settings"]["search"] = {k: v for k, v in app.config["search"].items() if k != "guidance"}
    out["search_guidance_chars"] = len(app.config["search"]["guidance"])
    if embedder:
        out["chunks_embedded"] = next((s["current_vectors"] for s in spaces if s["model_id"] == embedder.model_id), 0)
        out["embed_failures"] = failures.get(embedder.model_id, {}).get("failed", 0)
    return out


def curated_status(conn, app: MemApp) -> dict:
    """Record counts, and Markdown files from an earlier release that `curated import` would still add."""
    curated = app.curated(conn, OWNER)
    out: dict = {"records": curated.counts(), "legacy_markdown": None}
    if app.legacy_curated_dir.is_dir():
        plan = curated_md.plan_import(curated, app.legacy_curated_dir)
        out["legacy_markdown"] = {"path": str(app.legacy_curated_dir), "not_imported": len(plan.new),
                                  "conflicts": len(plan.conflicts), "invalid": len(plan.invalid),
                                  "imported": len(plan.unchanged)}
        if plan.new or not plan.ok:
            out["legacy_markdown"]["action"] = "run `hermes hermes-mem curated import --dry-run`, then without --dry-run"
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hermes-mem")
    parser.add_argument("--home", type=Path, help="Hermes profile home (default: HERMES_HOME or ~/.hermes)")
    setup(parser)
    args = parser.parse_args(argv)
    return run(args)
