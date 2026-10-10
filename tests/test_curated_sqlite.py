import hashlib
import json
import sqlite3
from dataclasses import asdict

import pytest
from pathlib import Path

from hermes_mem import curated_md, store, tools
from hermes_mem.app import OWNER, MemApp
from hermes_mem.cli import main
from hermes_mem.curated import CuratedError

# Byte-for-byte what the Markdown store of the previous release wrote (front matter in dataclass
# order, empty fields omitted), including a revision chain.
LEGACY = {
    "decision/decision-20261001-a1b2c3.md": (
        "---\nid: decision-20261001-a1b2c3\nkind: decision\nscope: owner\nsubject: storage\nstatus: accepted\n"
        "basis: inferred\nconfidence: high\nauthor: operator\ncreated_at: 1790000000.0\nupdated_at: 1790000000.0\n"
        "reviewed_at: 1790000000.0\n---\nUse SQLite, not Postgres.\n"
    ),
    "intent/intent-20261001-000001.md": (
        "---\nid: intent-20261001-000001\nkind: intent\nscope: peer:telegram:100\nsubject: telegram:100\n"
        "status: superseded\nbasis: inferred\nconfidence: low\nauthor: agent\nsuperseded_by: intent-20261002-000002\n"
        "created_at: 1790000001.0\nupdated_at: 1790000100.0\n---\nWants a NAS.\n"
    ),
    "intent/intent-20261002-000002.md": (
        "---\nid: intent-20261002-000002\nkind: intent\nscope: peer:telegram:100\nsubject: telegram:100\n"
        "status: proposed\nbasis: inferred\nconfidence: medium\nauthor: agent\nsupersedes: intent-20261001-000001\n"
        "note: Українською теж\ncreated_at: 1790000100.0\nupdated_at: 1790000100.0\n---\nWants a quiet NAS under 300 EUR.\n"
    ),
}


def write_tree(root, files):
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")


def tree_digest(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def cli(home, capsys, *argv):
    code = main(["--home", str(home), *argv])
    captured = capsys.readouterr()
    return code, (json.loads(captured.out) if captured.out.strip() else None), captured.err


def stored(app):
    conn = app.open_store()
    try:
        return {r.id: r for r in app.curated(conn, OWNER).all()}
    finally:
        conn.close()


def test_real_v1_store_file_upgrades_in_place_without_touching_its_rows(app, home):
    v1 = sqlite3.connect(home / "hermes-mem/mem.db")
    v1.executescript((Path(__file__).parent / "fixtures/store_v1_schema.sql").read_text())
    v1.execute("INSERT INTO meta VALUES ('schema_version', '1'), ('backfill_cursor', 's9')")
    v1.execute("""INSERT INTO episodes(id, episode_key, version, session_id, scope, identity_mode, user_ref, assistant_refs,
                  first_message_id, last_message_id, user_text, assistant_text, started_at, ended_at, compacted,
                  content_hash, status, source_state, created_at, updated_at)
                  VALUES (1, 'owner/user@1', 1, 's1', 'owner', 'role_timestamp', 'user@1', '[]', 1, 2,
                  'Back up the NAS nightly', 'Scheduled.', 1, 2, 0, 'h', 'current', 'present', 1, 1)""")
    v1.commit()
    before = v1.execute("SELECT * FROM episodes").fetchall()
    v1.close()

    for _ in range(2):  # the upgrade is idempotent
        conn = app.open_store()
        assert store.get_meta(conn, "schema_version") == "2" and store.get_meta(conn, "backfill_cursor") == "s9"
        assert [tuple(r) for r in conn.execute("SELECT * FROM episodes")] == before
        conn.close()
    conn = app.open_store()
    record = app.curated(conn, OWNER).propose(kind="decision", scope="owner", subject="backup",
                                              statement="Nightly NAS backup.", author="operator")
    assert app.curated(conn, OWNER).get(record.id).status == "accepted"
    conn.close()


def test_newer_schema_is_refused_before_any_ddl_runs(home, app):
    newer = sqlite3.connect(home / "hermes-mem/mem.db")
    newer.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    newer.execute("INSERT INTO meta VALUES ('schema_version', '3')")
    newer.commit()
    newer.close()
    with pytest.raises(RuntimeError, match="store schema 3 != supported 2"):
        app.open_store()
    check = sqlite3.connect(home / "hermes-mem/mem.db")
    assert [r[0] for r in check.execute("SELECT name FROM sqlite_master")] == ["meta", "sqlite_autoindex_meta_1"]
    check.close()


def test_markdown_import_previews_then_publishes_once_and_never_touches_the_files(app, state, home, capsys):
    legacy = app.legacy_curated_dir
    write_tree(legacy, LEGACY)
    (legacy / ".lock").write_text("")
    before = tree_digest(legacy)

    code, status, _ = cli(home, capsys, "status")
    assert status["curated"]["legacy_markdown"]["not_imported"] == 3 and "action" in status["curated"]["legacy_markdown"]

    code, preview, _ = cli(home, capsys, "curated", "import", "--dry-run")
    assert code == 0 and preview["ok"] and sorted(r["id"] for r in preview["new"]) == sorted(p.split("/")[1][:-3] for p in LEGACY)
    assert {"id": "decision-20261001-a1b2c3", "kind": "decision", "scope": "owner", "subject": "storage",
            "status": "accepted", "author": "operator", "basis": "inferred", "evidence": []} in preview["new"]
    assert stored(app) == {}

    code, done, _ = cli(home, capsys, "curated", "import")
    assert code == 0 and done["imported"] == 3
    records = stored(app)
    assert records["intent-20261002-000002"].statement == "Wants a quiet NAS under 300 EUR."
    assert records["intent-20261002-000002"].note == "Українською теж"
    assert (records["intent-20261001-000001"].status, records["intent-20261001-000001"].superseded_by) == (
        "superseded", "intent-20261002-000002")
    assert records["decision-20261001-a1b2c3"].created_at == 1790000000.0

    code, history, _ = cli(home, capsys, "curated", "history", "intent-20261002-000002")
    assert [r["id"] for r in history["chain"]] == ["intent-20261001-000001", "intent-20261002-000002"]
    assert {e["action"] for e in history["events"]} == {"import"}
    assert all("#sha256=" in e["source"] and e["author"] == "operator" for e in history["events"])

    state.session("tg_alice", "telegram", chat_type="dm", chat_id="100", user_id="100")
    seen = json.loads(tools.mem_curated(app, {"kind": "intent"}, "tg_alice"))["records"]
    assert [r["id"] for r in seen] == ["intent-20261002-000002"]

    code, again, _ = cli(home, capsys, "curated", "import")
    assert code == 0 and again["imported"] == 0 and sorted(again["unchanged"]) == sorted(records)
    assert tree_digest(legacy) == before
    code, status, _ = cli(home, capsys, "status")
    assert status["curated"]["records"] == {"accepted": 1, "proposed": 1, "superseded": 1}
    assert status["curated"]["legacy_markdown"]["not_imported"] == 0 and "action" not in status["curated"]["legacy_markdown"]


def test_malformed_conflicting_or_dangling_markdown_publishes_nothing(app, home, tmp_path, capsys):
    src = tmp_path / "md"
    write_tree(src, {**LEGACY, "decision/decision-20261003-bad000.md": "no front matter here"})
    before = tree_digest(src)
    code, preview, _ = cli(home, capsys, "curated", "import", "--from", str(src), "--dry-run")
    assert code == 2 and not preview["ok"]
    assert preview["invalid"] == ["decision/decision-20261003-bad000.md: missing front matter"]
    code, _, err = cli(home, capsys, "curated", "import", "--from", str(src))
    assert code == 2 and "nothing was imported" in err
    assert stored(app) == {} and tree_digest(src) == before

    (src / "decision/decision-20261003-bad000.md").unlink()
    (src / "intent/intent-20261001-000001.md").unlink()
    code, preview, _ = cli(home, capsys, "curated", "import", "--from", str(src), "--dry-run")
    assert code == 2 and preview["invalid"] == [
        "intent/intent-20261002-000002.md: supersedes intent-20261001-000001 is neither imported nor stored"]

    write_tree(src, LEGACY)
    orphan = LEGACY["intent/intent-20261002-000002.md"].replace("000002", "000003")
    write_tree(src, {"intent/intent-20261002-000003.md": orphan})
    code, preview, _ = cli(home, capsys, "curated", "import", "--from", str(src), "--dry-run")
    assert code == 2 and preview["invalid"] == [
        "intent/intent-20261002-000003.md: supersedes intent-20261001-000001, but intent-20261001-000001 has "
        "superseded_by intent-20261002-000002"]
    (src / "intent/intent-20261002-000003.md").unlink()
    assert cli(home, capsys, "curated", "import", "--from", str(src))[0] == 0
    edited = LEGACY["decision/decision-20261001-a1b2c3.md"].replace("not Postgres", "and Postgres")
    (src / "decision/decision-20261001-a1b2c3.md").write_text(edited)
    code, _, err = cli(home, capsys, "curated", "import", "--from", str(src))
    assert code == 2 and "1 conflict(s)" in err
    assert stored(app)["decision-20261001-a1b2c3"].statement == "Use SQLite, not Postgres."


def test_import_replans_under_the_write_lock_and_a_failure_midway_publishes_nothing(app, tmp_path, monkeypatch):
    src = tmp_path / "md"
    write_tree(src, LEGACY)
    conn = app.open_store()
    curated = app.curated(conn, OWNER)
    preview = curated_md.plan_import(curated, src)
    assert preview.ok and len(preview.new) == 3
    racer = preview.new[0][0]
    with curated.transaction():
        curated.insert(type(racer)(**{**asdict(racer), "statement": "written meanwhile"}), action="add", actor="operator")
    with pytest.raises(CuratedError, match="1 conflict"):
        curated_md.apply_import(curated, src)
    assert [r.id for r in curated.all()] == [racer.id]

    fresh = tmp_path / "fresh"
    (fresh / "hermes-mem").mkdir(parents=True)
    (fresh / "hermes-mem/config.yaml").write_text("embedding:\n  active: none\n")
    other = MemApp(fresh)
    conn2 = other.open_store()
    target = other.curated(conn2, OWNER)
    calls = []
    real = type(target).insert

    def flaky(self, record, **kw):
        calls.append(record.id)
        if len(calls) == 3:
            raise CuratedError("disk full")
        return real(self, record, **kw)

    monkeypatch.setattr(type(target), "insert", flaky)
    with pytest.raises(CuratedError, match="import rolled back, nothing was imported: disk full"):
        curated_md.apply_import(target, src)
    assert target.all() == [] and len(calls) == 3
    conn.close()
    conn2.close()


def test_import_rejects_divergent_snapshots_cycles_bad_types_and_unbacked_declarations(app, home, tmp_path, capsys):
    src = tmp_path / "md"
    first = LEGACY["intent/intent-20261001-000001.md"].replace("status: superseded", "status: proposed").replace(
        "superseded_by: intent-20261002-000002\n", "")
    write_tree(src, {"intent/intent-20261001-000001.md": first})
    assert cli(home, capsys, "curated", "import", "--from", str(src))[0] == 0
    write_tree(src, LEGACY)  # a later snapshot that revised the record outside this store
    code, preview, _ = cli(home, capsys, "curated", "import", "--from", str(src), "--dry-run")
    assert code == 2 and preview["conflicts"] == [
        "intent/intent-20261001-000001.md: stored intent-20261001-000001 is proposed, the file says superseded"]
    assert preview["invalid"] == ["intent/intent-20261002-000002.md: supersedes intent-20261001-000001, "
                                  "but intent-20261001-000001 has superseded_by None"]
    assert stored(app)["intent-20261001-000001"].status == "proposed"

    bad = tmp_path / "bad"
    loop = ("---\nid: intent-20261003-00000a\nkind: intent\nscope: owner\nsubject: x\nstatus: superseded\n"
            "basis: inferred\nconfidence: low\nauthor: agent\nsupersedes: intent-20261003-00000a\n"
            "superseded_by: intent-20261003-00000a\ncreated_at: 1.0\nupdated_at: 1.0\n---\nLoop.\n")
    typed = LEGACY["decision/decision-20261001-a1b2c3.md"].replace("subject: storage", "subject: 5")
    dated = LEGACY["decision/decision-20261001-a1b2c3.md"].replace("a1b2c3", "d4e5f6").replace(
        "created_at: 1790000000.0", "created_at: 2025-01-01")
    declared = LEGACY["decision/decision-20261001-a1b2c3.md"].replace("a1b2c3", "999999").replace(
        "basis: inferred", "basis: declared\nevidence:\n- episode:999999")
    write_tree(bad, {"intent/intent-20261003-00000a.md": loop, "decision/decision-20261001-a1b2c3.md": typed,
                     "decision/decision-20261001-d4e5f6.md": dated, "decision/decision-20261001-999999.md": declared})
    code, preview, _ = cli(home, capsys, "curated", "import", "--from", str(bad), "--dry-run")
    assert code == 2 and sorted(preview["invalid"]) == [
        "decision/decision-20261001-999999.md: evidence 'episode:999999' does not exist in the episode store",
        "decision/decision-20261001-a1b2c3.md: subject must be a string, got 5",
        "decision/decision-20261001-d4e5f6.md: created_at must be a number, got datetime.date(2025, 1, 1)",
        "intent/intent-20261003-00000a.md: intent-20261003-00000a superseded_by itself",
        "intent/intent-20261003-00000a.md: intent-20261003-00000a supersedes itself",
    ]
    write_tree(app.legacy_curated_dir, {"decision/decision-20261001-a1b2c3.md": typed})
    code, status, _ = cli(home, capsys, "status")
    assert code == 0 and status["curated"]["legacy_markdown"]["invalid"] == 1


def test_db_ahead_of_the_markdown_is_not_a_conflict_and_export_round_trips_whitespace(app, home, tmp_path, capsys):
    write_tree(app.legacy_curated_dir, LEGACY)
    assert cli(home, capsys, "curated", "import")[0] == 0
    assert cli(home, capsys, "curated", "review", "intent-20261002-000002", "--accept")[0] == 0
    code, again, _ = cli(home, capsys, "curated", "import")
    assert code == 0 and again["imported"] == 0 and len(again["unchanged"]) == 3

    code, padded, _ = cli(home, capsys, "curated", "add", "decision", "db", "  use postgres \n", "--basis", "inferred")
    assert padded["statement"] == "use postgres"
    out = tmp_path / "export"
    assert cli(home, capsys, "curated", "export", "--to", str(out))[0] == 0
    code, preview, _ = cli(home, capsys, "curated", "import", "--from", str(out), "--dry-run")
    assert code == 0 and preview["new"] == [] and len(preview["unchanged"]) == 4


def test_curated_content_and_history_cannot_be_rewritten(app, home, capsys):
    code, record, _ = cli(home, capsys, "curated", "add", "decision", "storage", "Use SQLite.", "--basis", "inferred")
    conn = app.open_store()
    for sql in ("UPDATE curated_records SET statement='Use Postgres.'",
                "UPDATE curated_records SET scope='peer:telegram:200'",
                "DELETE FROM curated_records",
                "UPDATE curated_events SET author='agent'",
                "DELETE FROM curated_events"):
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute(sql)
    conn.close()
    code, retracted, _ = cli(home, capsys, "curated", "review", record["id"], "--retract", "--note", "wrong")
    assert code == 0 and retracted["status"] == "retracted"
    conn = app.open_store()
    with pytest.raises(sqlite3.DatabaseError, match="final"):
        conn.execute("UPDATE curated_records SET status='accepted'")
    conn.close()
    code, _, err = cli(home, capsys, "curated", "review", record["id"], "--accept")
    assert code == 2 and "is retracted" in err

    code, second, _ = cli(home, capsys, "curated", "add", "intent", "owner", "Learn Rust.", "--basis", "inferred")
    assert cli(home, capsys, "curated", "revise", second["id"], "Learn Rust and Go.")[0] == 0
    code, _, err = cli(home, capsys, "curated", "revise", second["id"], "Learn Go.")
    assert code == 2 and "is superseded; revise its successor" in err
    conn = app.open_store()
    with pytest.raises(sqlite3.DatabaseError, match="superseded_by"):
        conn.execute("UPDATE curated_records SET superseded_by=id WHERE id=?", (second["id"],))
    code, third, _ = cli(home, capsys, "curated", "add", "intent", "owner", "Learn Zig.", "--basis", "inferred")
    with pytest.raises(sqlite3.DatabaseError, match="accepted"):
        conn.execute("UPDATE curated_records SET status='proposed' WHERE id=?", (third["id"],))
    conn.close()
    code, history, _ = cli(home, capsys, "curated", "history", second["id"])
    assert [(e["record_id"] == second["id"], e["action"]) for e in history["events"]] == [
        (True, "add"), (False, "revise"), (True, "supersede")]


def test_export_round_trips_into_a_fresh_profile(app, home, tmp_path, capsys):
    write_tree(app.legacy_curated_dir, LEGACY)
    assert cli(home, capsys, "curated", "import")[0] == 0
    code, accepted, _ = cli(home, capsys, "curated", "review", "intent-20261002-000002", "--accept")
    out = tmp_path / "export"
    code, done, _ = cli(home, capsys, "curated", "export", "--to", str(out))
    assert code == 0 and done["exported"] == 3
    code, _, err = cli(home, capsys, "curated", "export", "--to", str(out))
    assert code == 2 and "not an empty directory" in err

    fresh = tmp_path / "fresh"
    (fresh / "hermes-mem").mkdir(parents=True)
    (fresh / "hermes-mem/config.yaml").write_text("embedding:\n  active: none\nsync:\n  background: false\n")
    code, done, _ = cli(fresh, capsys, "curated", "import", "--from", str(out))
    assert code == 0 and done["imported"] == 3
    assert {k: asdict(v) for k, v in stored(MemApp(fresh)).items()} == {k: asdict(v) for k, v in stored(app).items()}


def test_two_record_supersession_cycle_is_invalid(app, tmp_path):
    def record(rid, other):
        return (f"---\nid: {rid}\nkind: intent\nscope: owner\nsubject: x\nstatus: superseded\nbasis: inferred\n"
                f"confidence: low\nauthor: agent\nsupersedes: {other}\nsuperseded_by: {other}\n"
                "created_at: 1.0\nupdated_at: 1.0\n---\nLoop.\n")
    a, b = "intent-20261003-00000a", "intent-20261003-00000b"
    write_tree(tmp_path, {f"intent/{a}.md": record(a, b), f"intent/{b}.md": record(b, a)})
    conn = app.open_store()
    plan = curated_md.plan_import(app.curated(conn, OWNER), tmp_path)
    assert plan.invalid == [f"intent/{a}.md: {a} is part of a supersession cycle",
                            f"intent/{b}.md: {b} is part of a supersession cycle"]
    conn.close()
