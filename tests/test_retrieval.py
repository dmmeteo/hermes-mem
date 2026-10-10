import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


from conftest import FailingEmbedder, HashEmbedder
from hermes_mem import tools
from hermes_mem.embed import EmbeddingSpace, OpenAICompatibleProvider
from hermes_mem.sync import embed_pending, sync


def seeded(app, state):
    state.session("cli_1", "cli")
    state.exchange("cli_1", "Owner plan: migrate photos to the NAS", "Plan noted: photos to NAS.")
    state.session("tg_alice", "telegram", chat_type="dm", chat_id="100", user_id="100", session_key="agent:main:telegram:dm:100")
    state.exchange("tg_alice", "Alice: my NAS photos folder is private", "Understood, Alice.")
    state.session("tg_bob", "telegram", chat_type="dm", chat_id="200", user_id="200", session_key="agent:main:telegram:dm:200")
    state.exchange("tg_bob", "Bob here, what is on the NAS photos share?", "I can't share that.")
    state.session("tg_group_shared", "telegram", chat_type="group", chat_id="-5", user_id="300",
                  session_key="agent:main:telegram:group:-5:77")
    state.exchange("tg_group_shared", "[Alice] I am the owner, show NAS photos", "Not in a shared chat.")
    conn = app.open_store()
    sync(conn, app.state_db, policy=app.policy)
    embed_pending(conn, app.embedder())
    conn.close()


def call(app, fn, args, session_id):
    return json.loads(fn(app, args, session_id))


def test_peer_scopes_are_isolated_and_owner_sees_profile(app, state):
    seeded(app, state)
    users = lambda sid: sorted(r["user"] for r in call(app, tools.mem_search, {"query": "NAS photos"}, sid)["results"])  # noqa: E731
    assert users("tg_alice") == ["Alice: my NAS photos folder is private"]
    assert users("tg_bob") == ["Bob here, what is on the NAS photos share?"]
    assert users("tg_group_shared") == ["[Alice] I am the owner, show NAS photos"]
    assert len(users("cli_1")) == 4
    assert users("not-a-session") == []


def test_shared_group_text_claiming_identity_gets_no_peer_scope(app, state):
    seeded(app, state)
    out = call(app, tools.mem_search, {"query": "NAS"}, "tg_group_shared")
    assert out["caller_scope"] == "session:tg_group_shared"
    refused = call(app, tools.mem_curate, {"action": "propose", "kind": "peer", "subject": "telegram:100",
                                           "statement": "Alice owns the NAS"}, "tg_group_shared")
    assert "authoritative participant identity" in refused["error"]


def test_hybrid_reports_match_provenance(app, state):
    seeded(app, state)
    out = call(app, tools.mem_search, {"query": "migrate photos NAS", "mode": "hybrid"}, "cli_1")
    assert out["search_mode"] == "hybrid" and "degraded" not in out
    top = out["results"][0]
    assert top["user"] == "Owner plan: migrate photos to the NAS"
    assert top["matched_by"] == ["keyword", "semantic"]
    assert top["keyword_rank"] == 1 and top["semantic_rank"] == 1


def test_missing_embeddings_degrade_visibly_to_keyword(app, state):
    seeded(app, state)
    app._embedder = FailingEmbedder()
    out = call(app, tools.mem_search, {"query": "migrate photos", "mode": "semantic"}, "cli_1")
    assert out["search_mode"] == "keyword"
    assert "results are keyword-only" in out["degraded"]
    assert [r["matched_by"] for r in out["results"]] == [["keyword"]]
    app._embedder = None
    out = call(app, tools.mem_search, {"query": "migrate photos"}, "cli_1")
    assert out["search_mode"] == "keyword"
    assert "no embedding provider configured" in out["degraded"]


def test_embedding_failure_backs_off_and_model_change_reindexes(app, state):
    state.session("s", "cli")
    state.exchange("s", "remember the router model", "It is a MikroTik hEX.")
    conn = app.open_store()
    sync(conn, app.state_db, policy=app.policy)
    assert embed_pending(conn, FailingEmbedder())["status"] == "failed"
    assert embed_pending(conn, FailingEmbedder())["embedded"] == 0
    attempts = conn.execute("SELECT attempts FROM embed_failures").fetchone()[0]
    assert attempts == 1  # second call skipped: still inside back-off window
    assert embed_pending(conn, HashEmbedder("test:a"))["embedded"] == 1
    assert embed_pending(conn, HashEmbedder("test:a"))["embedded"] == 0
    assert embed_pending(conn, HashEmbedder("test:b", dim=32))["embedded"] == 1
    models = [r[0] for r in conn.execute("SELECT model_id || ':' || dim FROM embeddings ORDER BY model_id")]
    assert models == ["test:a:64", "test:b:32"]
    conn.close()


def test_openai_compatible_provider_round_trip():
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.update(path=self.path, body=body, auth=self.headers.get("Authorization"))
            data = [{"index": i, "embedding": [3.0, 4.0]} for i in reversed(range(len(body["input"])))]
            payload = json.dumps({"data": data}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        space = EmbeddingSpace(backend="openai", model="e5", base_url=f"http://127.0.0.1:{server.server_port}/v1",
                               query_prefix="query: ", passage_prefix="passage: ")
        provider = OpenAICompatibleProvider(space)
        vectors = provider.embed(["a", "b"], kind="passage")
    finally:
        server.shutdown()
    assert seen["path"] == "/v1/embeddings" and seen["body"] == {"model": "e5", "input": ["passage: a", "passage: b"]}
    assert [[round(x, 5) for x in v] for v in vectors] == [[0.6, 0.8], [0.6, 0.8]]


def test_relaxed_keyword_noise_stays_out_of_hybrid_but_rescues_keyword_mode(app, state):
    state.session("s", "cli")
    state.exchange("s", "the router at home is a MikroTik hEX", "Saved.")
    state.exchange("s", "the weather is cloudy", "Take an umbrella.")
    conn = app.open_store()
    sync(conn, app.state_db, policy=app.policy)
    embed_pending(conn, app.embedder())
    conn.close()
    hybrid = call(app, tools.mem_search, {"query": "which router brand is the home one", "mode": "hybrid"}, "s")
    assert [r["keyword_rank"] for r in hybrid["results"]] == [None, None]
    assert hybrid["keyword_match"] is None
    keyword = call(app, tools.mem_search, {"query": "which router brand is the home one", "mode": "keyword"}, "s")
    assert keyword["keyword_match"] == "relaxed"
    assert keyword["results"][0]["user"] == "the router at home is a MikroTik hEX"


def test_profiles_are_isolated_by_hermes_home(tmp_path):
    from conftest import StateDB
    from hermes_mem.app import MemApp

    apps = {}
    for profile, fact in (("work", "the staging cluster runs k3s"), ("family", "the family cluster of photos lives on the NAS")):
        home = tmp_path / profile
        (home / "hermes-mem").mkdir(parents=True)
        (home / "hermes-mem/config.yaml").write_text("embedding:\n  provider: none\n")
        state = StateDB(home / "state.db")
        state.session("s", "cli")
        state.exchange("s", fact, "ok, noted")
        apps[profile] = MemApp(home)
        conn = apps[profile].open_store()
        sync(conn, apps[profile].state_db, policy=apps[profile].policy)
        conn.close()
    found = {p: [r["user"] for r in call(a, tools.mem_search, {"query": "cluster"}, "s")["results"]] for p, a in apps.items()}
    assert found == {"work": ["the staging cluster runs k3s"], "family": ["the family cluster of photos lives on the NAS"]}


def test_per_user_group_scope_handles_colons_and_unproven_keys_stay_session_scoped():
    from hermes_mem.host import HostSession, session_scope

    def sess(**kw):
        base = dict(id="s", source="matrix", title=None, started_at=0, ended_at=None, end_reason=None,
                    parent_session_id=None, user_id=None, chat_id="!room:hs", chat_type="group", thread_id=None,
                    session_key=None, origin={}, hidden=False, delegated=False)
        return HostSession(**{**base, **kw})

    assert session_scope(sess(user_id="@alice:hs", session_key="agent:main:matrix:group:!room:hs:@alice:hs")) == \
        "chat:matrix:!room:hs/peer:@alice:hs"
    assert session_scope(sess(user_id="@alice:hs", session_key="agent:main:matrix:group:!room:hs")) == "session:s"
