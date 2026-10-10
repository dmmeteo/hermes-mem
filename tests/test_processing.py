import json
import socket

import pytest

from hermes_mem.app import ConfigError, MemApp
from hermes_mem.cli import main
from hermes_mem.select import sha256
from hermes_mem.sync import embed_pending, sync

from conftest import HashEmbedder

EXPLICIT_DEFAULTS = "processing:\n  retain: none\n  compression: none\n  expiry: off\n"
BASE_CONFIG = "embedding:\n  provider: none\nsync:\n  background: false\n"
LONG_ANSWER = "\n\n".join(f"Крок {i}: перевірити restic snapshot і журнал NAS." + " деталі" * 40 for i in range(6))


def stored(app):
    conn = app.open_store()
    try:
        episodes = [dict(r) for r in conn.execute(
            """SELECT episode_key, version, session_id, scope, user_ref, assistant_refs, user_text, assistant_text,
                      content_hash, status FROM episodes ORDER BY id""")]
        chunks = [tuple(r) for r in conn.execute("SELECT episode_id, chunk_index, text FROM chunks ORDER BY id")]
        return episodes, chunks
    finally:
        conn.close()


@pytest.mark.parametrize("processing", ["", "processing:\n", EXPLICIT_DEFAULTS], ids=["omitted", "empty", "explicit-defaults"])
def test_default_processing_stores_selected_originals_verbatim_without_model_calls(home, state, monkeypatch, processing):
    (home / "hermes-mem/config.yaml").write_text(BASE_CONFIG + processing)
    app = MemApp(home)
    embedder = HashEmbedder()
    app._embedder, app._embedder_built = embedder, True
    state.session("tg_dm", "telegram", chat_type="dm", chat_id="42", user_id="42", session_key="agent:main:telegram:dm:42")
    state.exchange("tg_dm", "Ні, тільки Postgres.", "Зрозумів: лише Postgres, без MySQL.")
    state.exchange("tg_dm", "Як перевіряти бекапи?", LONG_ANSWER)

    def no_network(*_args, **_kwargs):
        raise AssertionError("sync must not call any model or network service")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    conn = app.open_store()
    try:
        report = sync(conn, app.state_db, policy=app.policy)
    finally:
        conn.close()

    assert report.processing == {"excerpts": "raw", "retain": "none", "compression": "none", "expiry": "none"}
    assert (report.inserted, report.revised, report.retracted) == (2, 0, 0)
    assert embedder.calls == 0
    assert not (home / "hermes-mem/curated").exists() or not any((home / "hermes-mem/curated").rglob("*.md"))
    episodes, chunks = stored(app)
    assert [(e["episode_key"], e["version"], e["session_id"], e["scope"], e["user_ref"],
             json.loads(e["assistant_refs"]), e["user_text"], e["assistant_text"], e["content_hash"], e["status"])
            for e in episodes] == [
        ("peer:telegram:42/user@1700000001.0", 1, "tg_dm", "peer:telegram:42", "user@1700000001.0",
         ["assistant@1700000002.0"], "Ні, тільки Postgres.", "Зрозумів: лише Postgres, без MySQL.",
         sha256("Ні, тільки Postgres.\x00Зрозумів: лише Postgres, без MySQL."), "current"),
        ("peer:telegram:42/user@1700000003.0", 1, "tg_dm", "peer:telegram:42", "user@1700000003.0",
         ["assistant@1700000004.0"], "Як перевіряти бекапи?", LONG_ANSWER,
         sha256("Як перевіряти бекапи?\x00" + LONG_ANSWER), "current"),
    ]
    assert chunks[0] == (1, 0, "User: Ні, тільки Postgres.\n\nAssistant: Зрозумів: лише Postgres, без MySQL.")
    long_chunks = [text for episode_id, _, text in chunks if episode_id == 2]
    assert len(long_chunks) > 1
    assert all(text.startswith("User: Як перевіряти бекапи?\n\nAssistant: Крок ") for text in long_chunks)
    assert "\n\n".join(text.removeprefix("User: Як перевіряти бекапи?\n\nAssistant: ") for text in long_chunks) == LONG_ANSWER

    monkeypatch.undo()
    conn = app.open_store()
    try:
        assert embed_pending(conn, app.embedder())["status"] == "complete"
        assert sync(conn, app.state_db, policy=app.policy).unchanged == 1
    finally:
        conn.close()


@pytest.mark.parametrize("processing, mentions", [
    ("processing:\n  retain: llm\n", "processing.retain: 'llm' is not implemented"),
    ("processing:\n  compression: summarize\n", "processing.compression: 'summarize' is not implemented"),
    ("processing:\n  expiry: 90d\n", "processing.expiry: '90d' is not implemented"),
    ("processing:\n  retain: on\n", "processing.retain: True is not supported; only 'none' is implemented"),
    ("processing:\n  retian: none\n", "unknown processing key(s) retian; supported: retain, compression, expiry"),
    ("processing:\n  on: none\n", "unknown processing key(s) True; supported: retain, compression, expiry"),
    ("processing: raw\n", "processing must be a mapping of retain, compression, expiry, got 'raw'"),
])
def test_unsupported_processing_fails_with_actionable_error_before_any_indexing(home, state, capsys, processing, mentions):
    config = home / "hermes-mem/config.yaml"
    config.write_text(BASE_CONFIG + processing)
    state.session("s", "cli")
    state.exchange("s", "Remember the NAS backup plan", "Nightly restic at 02:00.")

    with pytest.raises(ConfigError) as error:
        MemApp(home)
    assert str(error.value).startswith(f"{config}: ")
    assert mentions in str(error.value)

    assert main(["--home", str(home), "sync", "--all"]) == 2
    out = capsys.readouterr()
    assert out.out == ""
    assert out.err == f"error: {error.value}\n"
    assert not (home / "hermes-mem/mem.db").exists()


def test_status_reports_the_processing_policy_in_effect(home, state, capsys):
    assert main(["--home", str(home), "status"]) == 0
    assert json.loads(capsys.readouterr().out)["processing"] == {
        "excerpts": "raw", "retain": "none", "compression": "none", "expiry": "none"
    }
