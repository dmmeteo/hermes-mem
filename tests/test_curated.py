import json

from hermes_mem import tools
from hermes_mem.cli import main
from hermes_mem.sync import sync


def call(app, fn, args, sid):
    return json.loads(fn(app, args, sid))


def setup_peer(app, state):
    state.session("tg_alice", "telegram", chat_type="dm", chat_id="100", user_id="100")
    state.exchange("tg_alice", "I always want answers in Ukrainian", "Добре, українською.")
    state.session("tg_bob", "telegram", chat_type="dm", chat_id="200", user_id="200")
    conn = app.open_store()
    sync(conn, app.state_db, policy=app.policy)
    episode = conn.execute("SELECT id FROM episodes").fetchone()[0]
    conn.close()
    return episode


def test_agent_proposal_needs_operator_acceptance_and_revision_supersedes(app, state, home, capsys):
    episode = setup_peer(app, state)
    proposed = call(app, tools.mem_curate, {
        "action": "propose", "kind": "peer", "subject": "telegram:100", "basis": "declared",
        "confidence": "high", "statement": "Prefers answers in Ukrainian.", "evidence": [f"episode:{episode}"],
    }, "tg_alice")["record"]
    assert (proposed["status"], proposed["basis"], proposed["author"]) == ("proposed", "declared", "agent")

    accept_by_agent = call(app, tools.mem_curate, {"action": "retract", "id": proposed["id"]}, "tg_bob")
    assert accept_by_agent == {"error": "record is outside this conversation's scope"}

    assert main(["--home", str(home), "curated", "review", proposed["id"], "--accept"]) == 0
    capsys.readouterr()
    listed = call(app, tools.mem_curated, {"kind": "peer"}, "tg_alice")["records"]
    assert [(r["id"], r["status"]) for r in listed] == [(proposed["id"], "accepted")]
    assert call(app, tools.mem_curated, {"kind": "peer"}, "tg_bob")["records"] == []

    demote = call(app, tools.mem_curate, {"action": "revise", "id": proposed["id"], "statement": "Prefers English."}, "tg_alice")
    assert "accepted by the operator" in demote["error"]
    assert main(["--home", str(home), "curated", "revise", proposed["id"], "Prefers Ukrainian, English for code."]) == 0
    revised = json.loads(capsys.readouterr().out)
    assert (revised["status"], revised["supersedes"]) == ("accepted", proposed["id"])
    everything = call(app, tools.mem_curated, {"include_inactive": True}, "tg_alice")["records"]
    assert {(r["id"], r["status"]) for r in everything} == {(proposed["id"], "superseded"), (revised["id"], "accepted")}
    assert [r["id"] for r in call(app, tools.mem_curated, {}, "tg_alice")["records"]] == [revised["id"]]


def test_declared_needs_real_user_evidence_and_peer_cards_need_own_identity(app, state):
    setup_peer(app, state)
    no_evidence = call(app, tools.mem_curate, {"action": "propose", "kind": "intent", "subject": "telegram:100",
                                               "basis": "declared", "statement": "Wants a NAS"}, "tg_alice")
    assert "needs evidence" in no_evidence["error"]
    fake = call(app, tools.mem_curate, {"action": "propose", "kind": "intent", "subject": "telegram:100",
                                        "basis": "declared", "statement": "Wants a NAS", "evidence": ["episode:999"]}, "tg_alice")
    assert "does not exist" in fake["error"]
    other = call(app, tools.mem_curate, {"action": "propose", "kind": "peer", "subject": "telegram:200",
                                         "statement": "Bob is admin"}, "tg_alice")
    assert "only describe its own participant 'telegram:100'" in other["error"]


def test_operator_record_is_scoped_to_owner_sessions(app, state, home, capsys):
    assert main(["--home", str(home), "curated", "add", "decision", "storage", "Use SQLite, not Postgres, for hermes-mem.",
                 "--basis", "inferred"]) == 0
    capsys.readouterr()
    out = call(app, tools.mem_curated, {"kind": "decision"}, "cli_owner_unknown")
    assert out["records"] == []  # unknown session: narrowest scope, owner records hidden
    state.session("cli_owner", "cli")
    out = call(app, tools.mem_curated, {"kind": "decision"}, "cli_owner")
    assert [r["statement"] for r in out["records"]] == ["Use SQLite, not Postgres, for hermes-mem."]
    assert out["records"][0]["status"] == "accepted" and out["records"][0]["author"] == "operator"
    assert not (home / "hermes-mem/curated").exists()


def test_record_ids_cannot_escape_the_profile_and_bad_input_is_an_error(app, state, home, tmp_path, capsys):
    setup_peer(app, state)
    other = tmp_path / "other-profile/hermes-mem/curated/peer"
    other.mkdir(parents=True)
    (other / "peer-20261010-abcdef.md").write_text("---\nid: peer-20261010-abcdef\nkind: peer\nscope: peer:telegram:100\n"
                                                    "subject: telegram:100\n---\nSECRET\n")
    escaped = call(app, tools.mem_curated, {"action": "get", "id": str(other / "peer-20261010-abcdef")}, "tg_alice")
    assert "malformed record id" in escaped["error"]
    for evidence in (["episode:abc"], ["garbage"], ["session:x#msg:1#msg:2"]):
        out = call(app, tools.mem_curate, {"action": "propose", "kind": "intent", "subject": "telegram:100",
                                           "statement": "x", "evidence": evidence}, "tg_alice")
        assert "malformed evidence ref" in out["error"]
    huge = call(app, tools.mem_curate, {"action": "propose", "kind": "intent", "subject": "telegram:100",
                                        "statement": "x", "note": "n" * 5000}, "tg_alice")
    assert "note exceeds 1000 chars" in huge["error"]
    bad_mode = call(app, tools.mem_search, {"query": "x", "mode": "psychic"}, "tg_alice")
    assert "mode must be" in bad_mode["error"]


def test_declared_cannot_rest_on_a_message_reference_alone(app, state):
    setup_peer(app, state)
    conn = app.open_store()
    last = conn.execute("SELECT last_message_id FROM episodes").fetchone()[0]
    conn.close()
    out = call(app, tools.mem_curate, {"action": "propose", "kind": "intent", "subject": "telegram:100",
                                       "basis": "declared", "statement": "Wants Ukrainian",
                                       "evidence": [f"session:tg_alice#msg:{last}"]}, "tg_alice")
    assert "needs evidence from a user-authored message" in out["error"]
