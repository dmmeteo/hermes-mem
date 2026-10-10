import json

import pytest

from hermes_mem import sync as sync_mod
from hermes_mem.search import SearchParams, search
from hermes_mem.sync import embed_pending, sync


def run_sync(app, **kw):
    conn = app.open_store()
    try:
        return sync(conn, app.state_db, policy=app.policy, **kw)
    finally:
        conn.close()


def find(app, query, mode="keyword", scopes=None):
    conn = app.open_store()
    try:
        embed_pending(conn, app.embedder())
        return search(conn, app.state_db, app.embedder(), SearchParams(query=query, scopes=scopes, mode=mode))
    finally:
        conn.close()


def current(app):
    conn = app.open_store()
    rows = conn.execute("SELECT user_text, status, source_state FROM episodes ORDER BY id").fetchall()
    conn.close()
    return [tuple(r) for r in rows]


def test_bounded_backfill_resumes_and_repeat_is_a_no_op(app, state):
    for i in range(5):
        state.session(f"2026_0{i}", "cli")
        state.exchange(f"2026_0{i}", f"question number {i} about backups", f"answer {i}")
    passes = []
    while True:
        report = run_sync(app, max_sessions=2)
        passes.append((report.sessions, report.inserted, report.complete))
        if report.complete:
            break
    assert passes == [(2, 2, False), (2, 2, False), (1, 1, True)]
    again = run_sync(app, max_sessions=10)
    assert (again.sessions, again.inserted, again.revised, again.unchanged, again.complete) == (5, 0, 0, 5, True)
    assert len(current(app)) == 5


def test_routine_cron_subagent_and_delegated_sessions_are_skipped(app, state):
    state.session("a_cron", "cron")
    state.exchange("a_cron", "daily digest", "digest body")
    state.session("b_sub", "subagent")
    state.exchange("b_sub", "subtask", "subtask result")
    state.session("c_deleg", "telegram", model_config=json.dumps({"_delegate_from": "x"}))
    state.exchange("c_deleg", "delegated", "delegated result")
    state.session("d_cli", "cli")
    state.exchange("d_cli", "real question", "real answer")
    report = run_sync(app)
    assert (report.skipped, report.inserted) == (3, 1)
    assert current(app) == [("real question", "current", "present")]


def test_compacted_history_stays_searchable_without_duplicate_clones(app, state):
    state.session("s1", "cli")
    u, a = state.exchange("s1", "We decided to keep nightly restic backups", "Yes, restic nightly to the NAS.")
    # In-place compaction: originals archived, summary + byte-exact clone of the tail inserted.
    state.sql("UPDATE messages SET active=0, compacted=1 WHERE session_id='s1'")
    state.msg("s1", "user", "[CONTEXT COMPACTION — REFERENCE ONLY] summary", _compressed_summary=1, display_kind="hidden")
    ts = state.conn.execute("SELECT timestamp FROM messages WHERE id=?", (a,)).fetchone()[0]
    state.msg("s1", "assistant", "Yes, restic nightly to the NAS.", timestamp=ts)
    run_sync(app)
    hits = find(app, "restic")["results"]
    assert [(h["user"], h["compacted_in_source"], h["message_ids"][0]) for h in hits] == [
        ("We decided to keep nightly restic backups", True, u)
    ]
    assert hits[0]["source"]["state"] == "present"


def test_changed_source_supersedes_and_search_never_returns_the_old_version(app, state):
    state.session("s1", "cli")
    u, _ = state.exchange("s1", "Use port 8080 for the dashboard", "Noted: 8080.")
    run_sync(app)
    state.sql("UPDATE messages SET content='Use port 9090 for the dashboard' WHERE id=?", u)
    report = run_sync(app)
    assert report.revised == 1
    assert [h["user"] for h in find(app, "dashboard")["results"]] == ["Use port 9090 for the dashboard"]
    assert find(app, "8080 dashboard")["results"][0]["user"] == "Use port 9090 for the dashboard"
    assert sorted(current(app)) == [
        ("Use port 8080 for the dashboard", "superseded", "present"),
        ("Use port 9090 for the dashboard", "current", "present"),
    ]


def test_source_changed_after_indexing_is_reported_not_hydrated(app, state):
    state.session("s1", "cli")
    u, _ = state.exchange("s1", "Grafana lives on host alpha", "Got it.")
    run_sync(app)
    state.sql("UPDATE messages SET content='Grafana lives on host beta' WHERE id=?", u)
    (hit,) = find(app, "grafana")["results"]
    assert hit["user"] == "Grafana lives on host alpha"
    assert hit["source"]["state"] == "changed"
    assert hit["source"]["context"] == "unavailable"


def test_changed_assistant_answer_is_reported_too(app, state):
    state.session("s1", "cli")
    _, a = state.exchange("s1", "Where does Grafana live?", "On host alpha.")
    run_sync(app)
    state.sql("UPDATE messages SET content='On host beta.' WHERE id=?", a)
    (hit,) = find(app, "grafana")["results"]
    assert (hit["assistant"], hit["source"]["state"]) == ("On host alpha.", "changed")


def test_rewound_exchange_is_retracted(app, state):
    state.session("s1", "cli")
    state.exchange("s1", "Keep Hindsight forever", "Okay, keeping it.")
    u2, _ = state.exchange("s1", "Actually drop the Kafka idea", "Dropped.")
    run_sync(app)
    state.sql("UPDATE messages SET active=0 WHERE session_id='s1' AND id>=?", u2)
    report = run_sync(app)
    assert report.retracted == 1
    assert [h["user"] for h in find(app, "Hindsight Kafka")["results"]] == ["Keep Hindsight forever"]


def test_deleted_session_keeps_excerpt_but_reports_missing_source(app, state):
    state.session("s1", "cli")
    state.exchange("s1", "My NAS admin password policy: rotate quarterly", "Quarterly rotation noted.")
    state.session("s2", "cli")
    state.exchange("s2", "neighbour", "x")
    run_sync(app)
    state.sql("DELETE FROM messages WHERE session_id='s1'")
    state.sql("DELETE FROM sessions WHERE id='s1'")
    report = run_sync(app)
    assert report.source_missing == 1
    (hit,) = find(app, "rotation quarterly")["results"]
    assert hit["user"].startswith("My NAS admin password policy")
    assert hit["source"] == {"state": "missing", "context": "unavailable", "reason": "source session no longer exists"}


def test_interrupted_reconcile_rolls_back_and_rerun_converges(app, state, monkeypatch):
    state.session("s1", "cli")
    state.exchange("s1", "first fact about zfs", "a")
    state.exchange("s1", "second fact about zfs", "b")
    real = sync_mod._insert_episode
    calls = {"n": 0}

    def crash_on_second(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt("process killed")
        return real(*args, **kwargs)

    monkeypatch.setattr(sync_mod, "_insert_episode", crash_on_second)
    with pytest.raises(KeyboardInterrupt):
        run_sync(app)
    assert current(app) == []
    monkeypatch.setattr(sync_mod, "_insert_episode", real)
    report = run_sync(app)
    assert report.inserted == 2
    assert run_sync(app).inserted == 0


def test_pruned_parent_does_not_duplicate_or_rekey_the_child_copy(app, state):
    state.session("a_parent", "cli", end_reason="compression")
    u, a = state.exchange("a_parent", "Our VLAN for cameras is 30", "VLAN 30 for cameras.")
    ts = [r[0] for r in state.conn.execute("SELECT timestamp FROM messages ORDER BY id")]
    state.session("b_child", "cli", parent_session_id="a_parent")
    state.msg("b_child", "user", "Our VLAN for cameras is 30", timestamp=ts[0])
    state.msg("b_child", "assistant", "VLAN 30 for cameras.", timestamp=ts[1])
    run_sync(app)
    assert current(app) == [("Our VLAN for cameras is 30", "current", "present")]
    conn = app.open_store()
    episode_id = conn.execute("SELECT id FROM episodes").fetchone()[0]
    conn.close()
    state.sql("UPDATE sessions SET parent_session_id=NULL WHERE id='b_child'")
    state.sql("DELETE FROM messages WHERE session_id='a_parent'")
    state.sql("DELETE FROM sessions WHERE id='a_parent'")
    run_sync(app)
    run_sync(app)
    (hit,) = find(app, "cameras VLAN")["results"]
    assert (hit["episode_id"], hit["session_id"], hit["source"]["state"]) == (episode_id, "b_child", "present")


def test_one_poison_chunk_does_not_block_the_rest_and_backoff_grows(app, state):
    from conftest import HashEmbedder
    from hermes_mem.embed import EmbeddingUnavailable

    class Picky(HashEmbedder):
        def embed(self, texts, *, kind):
            if any("POISON" in t for t in texts):
                raise EmbeddingUnavailable("cannot embed this text")
            return super().embed(texts, kind=kind)

    state.session("s", "cli")
    for i in range(6):
        state.exchange("s", f"healthy fact {i}", "ok noted")
    state.exchange("s", "POISON text", "ok noted")
    run_sync(app)
    conn = app.open_store()
    picky = Picky()
    assert embed_pending(conn, picky)["embedded"] == 6
    conn.execute("UPDATE embed_failures SET retry_after = 0")
    assert embed_pending(conn, picky)["status"] == "failed"
    attempts, delay = conn.execute("SELECT attempts, retry_after - strftime('%s','now') FROM embed_failures").fetchone()
    conn.close()
    assert attempts == 2 and delay > 30
