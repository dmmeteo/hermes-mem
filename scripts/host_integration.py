"""Exercise hermes-mem inside the real Hermes host runtime with an isolated HERMES_HOME.

Usage: python scripts/host_integration.py HOST_SOURCE WORK_DIR EMBED_CONFIG_YAML
Run with a Python the host supports and HOST_SOURCE on PYTHONPATH. The script writes
state.db only through Hermes' own SessionDB API, then drives real plugin discovery,
the `hermes` CLI, tool dispatch and the post_llm_call hook.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
host_source, work, embed_config = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
if work.exists():
    shutil.rmtree(work)
home = work / "hermes-home"
plugin_dir = home / "plugins" / "hermes-mem"
plugin_dir.mkdir(parents=True)
for item in ("plugin.yaml", "__init__.py", "hermes_mem", "engine"):
    src = REPO / item
    (shutil.copytree if src.is_dir() else shutil.copy)(src, plugin_dir / item)
(home / "config.yaml").write_text("plugins:\n  enabled: [hermes-mem]\n")
(home / "hermes-mem").mkdir()
(home / "hermes-mem" / "config.yaml").write_text(Path(embed_config).read_text())
os.environ["HERMES_HOME"] = str(home)
os.environ.pop("HERMES_SAFE_MODE", None)

results: list[dict] = []


def check(name: str, ok: bool, detail=None) -> None:
    results.append({"check": name, "ok": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f" :: {detail}"), flush=True)


# --- Write realistic history through Hermes' own writer -------------------------------
from hermes_state import SessionDB  # noqa: E402

db = SessionDB(db_path=home / "state.db")
db.create_session("s_backup", "cli")
u1 = db.append_message("s_backup", "user", "Ми вирішили робити щоночі restic-бекапи фото на NAS")
db.append_message("s_backup", "assistant", None, tool_calls=[{"id": "c1", "type": "function",
                  "function": {"name": "terminal", "arguments": "{\"cmd\": \"restic snapshots\"}"}}])
db.append_message("s_backup", "tool", "snapshot 1a2b3c 2026-10-09 /photos", tool_call_id="c1", tool_name="terminal")
db.append_message("s_backup", "assistant", "Так: restic щоночі о 02:00, репозиторій на NAS, 30 щоденних копій.")
compacted_active = db.archive_and_compact("s_backup", [
    {"role": "user", "content": "[CONTEXT COMPACTION — REFERENCE ONLY] earlier: backup plan agreed",
     "_compressed_summary": True},
])
db.append_message("s_backup", "user", "And the offsite copy?")
db.append_message("s_backup", "assistant", "Weekly rclone sync of the restic repo to Backblaze B2.")

db.create_session("s_rewind", "cli")
db.append_message("s_rewind", "user", "Let's move everything to Kubernetes")
db.append_message("s_rewind", "assistant", "Okay, planning a Kubernetes migration.")
undo = db.append_message("s_rewind", "user", "Actually, let's adopt Kafka for the photo pipeline")
db.append_message("s_rewind", "assistant", "Kafka it is.")

db.create_session("s_gone", "cli")
db.append_message("s_gone", "user", "The garage door code rotates every quarter")
db.append_message("s_gone", "assistant", "Noted: quarterly rotation for the garage door code.")

db.create_session("s_cron", "cron")
db.append_message("s_cron", "user", "Daily digest: restic NAS status")
db.append_message("s_cron", "assistant", "All restic NAS jobs green.")
db.close()

state_db = home / "state.db"
raw = __import__("sqlite3").connect(state_db)
flags = raw.execute("SELECT active, compacted FROM messages WHERE id=?", (u1,)).fetchone()
raw.close()
check("host compaction archived the original user row as active=0, compacted=1", flags == (0, 1), flags)


def state_digest() -> str:
    conn = __import__("sqlite3").connect(state_db.resolve().as_uri() + "?mode=ro", uri=True)
    rows = conn.execute("SELECT * FROM messages ORDER BY id").fetchall() + conn.execute("SELECT * FROM sessions ORDER BY id").fetchall()
    conn.close()
    return hashlib.sha256(repr(rows).encode()).hexdigest()


before = state_digest()

# --- Real discovery and registration -----------------------------------------------------
from hermes_cli import plugins as hp  # noqa: E402

hp._reset_plugin_managers_for_tests()
hp.discover_plugins(force=True)
listing = {p.get("name") or p.get("key"): p for p in hp.get_plugin_manager().list_plugins()}
entry = listing.get("hermes-mem")
check("plugin discovered and enabled in isolated HERMES_HOME", bool(entry) and entry.get("enabled") and not entry.get("error"), entry)

import model_tools  # noqa: E402  (registers built-in tools, including session_search)
from tools.registry import registry  # noqa: E402

registered = [name for name in ("mem_search", "mem_curated", "mem_curate") if registry.get_entry(name) is not None]
check("tools registered in the host registry", registered == ["mem_search", "mem_curated", "mem_curate"], registered)
session_search = registry.get_entry("session_search")
check("existing session_search tool still registered by the host, not replaced",
      session_search is not None and getattr(session_search, "toolset", None) == "session_search",
      getattr(session_search, "toolset", None))

# --- Real CLI: hermes plugins list / hermes hermes-mem ... ---------------------------------
env = dict(os.environ, HERMES_HOME=str(home))


def hermes(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "hermes_cli.main", *args], env=env, capture_output=True, text=True, timeout=600)


plugins_list = hermes("plugins", "list")
check("`hermes plugins list` shows hermes-mem", plugins_list.returncode == 0 and "hermes-mem" in plugins_list.stdout,
      plugins_list.stdout[-800:] + plugins_list.stderr[-800:])
synced = hermes("hermes-mem", "sync", "--all", "--max-sessions", "2", "--embed")
check("`hermes hermes-mem sync --all --embed` exits 0", synced.returncode == 0, synced.stderr[-1500:])
sync_out = json.loads(synced.stdout[synced.stdout.index("{"):]) if synced.returncode == 0 else {}
passes = sync_out.get("passes", [])
check("bounded backfill took several resumable passes and completed", len(passes) >= 2 and passes[-1]["complete"], passes)
check("identity fallback reported honestly (no message_uid column)", passes and passes[0]["identity_mode"] == "role_timestamp", passes[:1])
check("embedding ran with the real model", sync_out.get("embedding", {}).get("status") == "complete", sync_out.get("embedding"))
RAW_POLICY = {"excerpts": "raw", "retain": "none", "compression": "none", "expiry": "none"}
check("sync reports the default raw processing policy (config has no processing section)",
      passes and all(p["processing"] == RAW_POLICY for p in passes), passes[:1])
status_run = hermes("hermes-mem", "status")
check("`hermes hermes-mem status` reports the raw processing policy", status_run.returncode == 0
      and json.loads(status_run.stdout[status_run.stdout.index("{"):])["processing"] == RAW_POLICY, status_run.stdout[-800:])
again = hermes("hermes-mem", "sync", "--all")
again_pass = json.loads(again.stdout[again.stdout.index("{"):])["passes"] if again.returncode == 0 else []
check("repeat backfill is a no-op", again.returncode == 0 and sum(p["inserted"] + p["revised"] for p in again_pass) == 0, again_pass)


def dispatch(name, args, session_id):
    return json.loads(model_tools.handle_function_call(name, args, session_id=session_id))


out = dispatch("mem_search", {"query": "як ми бекапимо фотографії?", "mode": "semantic"}, "s_backup")
top = (out.get("results") or [{}])[0]
check("semantic-only search through real dispatch finds the compacted Ukrainian episode",
      out.get("search_mode") == "semantic" and "restic" in top.get("user", "") and top.get("compacted_in_source"), out)
out = dispatch("mem_search", {"query": "photo backup schedule", "mode": "hybrid"}, "s_backup")
check("English hybrid query reaches the Ukrainian episode", out.get("search_mode") == "hybrid"
      and any("restic" in r["user"] for r in out["results"][:3]), [r["user"] for r in out.get("results", [])])
out = dispatch("mem_search", {"query": "1a2b3c", "mode": "keyword"}, "s_backup")
check("tool output (only place '1a2b3c' occurs) is not indexed", out["results"] == [], out["results"])
out = dispatch("mem_search", {"query": "digest", "mode": "keyword"}, "s_backup")
check("cron session (only place 'digest' occurs) is excluded", out["results"] == [], out["results"])
host_fts = __import__("sqlite3").connect(state_db.resolve().as_uri() + "?mode=ro", uri=True)
host_hits = host_fts.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH '1a2b3c OR digest'").fetchone()[0]
host_fts.close()
check("control: the host's own FTS does contain those terms", host_hits >= 2, host_hits)

check("state.db rows unchanged by plugin discovery, sync, embedding and search", state_digest() == before)

# --- Rewind and delete through the host, then reconcile -----------------------------------
db = SessionDB(db_path=state_db)
db.rewind_to_message("s_rewind", undo)
db.delete_session("s_gone")
db.close()
recon = hermes("hermes-mem", "sync", "--all")
recon_pass = json.loads(recon.stdout[recon.stdout.index("{"):])["passes"] if recon.returncode == 0 else []
check("reconcile retracts the rewound exchange and flags the deleted source",
      sum(p["retracted"] for p in recon_pass) == 1 and sum(p["source_missing"] for p in recon_pass) == 1, recon_pass)
out = dispatch("mem_search", {"query": "Kafka", "mode": "keyword"}, "s_backup")
check("rewound exchange no longer returned", not any("Kafka" in r["user"] for r in out["results"]), out["results"])
out = dispatch("mem_search", {"query": "garage door code", "mode": "hybrid"}, "s_backup")
gone = next((r for r in out["results"] if "garage" in r["user"]), None)
check("deleted source keeps excerpt with source_missing and no context",
      gone is not None and gone["source"]["state"] == "missing" and gone["source"]["context"] == "unavailable", gone)

# --- Hook path: post_llm_call schedules background sync -----------------------------------
db = SessionDB(db_path=state_db)
db.create_session("s_live", "cli")
db.append_message("s_live", "user", "Запам'ятай: домашній роутер — MikroTik hEX")
db.append_message("s_live", "assistant", "Запам'ятав: MikroTik hEX.")
db.close()
state_digest_after_host_writes = state_digest()
started = time.monotonic()
hp.invoke_hook("post_llm_call", session_id="s_live", user_message="x", assistant_response="y", model="m", platform="cli")
hook_latency = time.monotonic() - started
check("hook returns without waiting for indexing", hook_latency < 2.0, hook_latency)
deadline, found = time.monotonic() + 120, False
while time.monotonic() < deadline and not found:
    found = any("MikroTik" in r["user"] for r in dispatch("mem_search", {"query": "MikroTik", "mode": "keyword"}, "s_backup")["results"])
    time.sleep(0.5)
check("background worker indexed the new turn", found)

# --- Curated round trip via real dispatch + CLI -------------------------------------------
proposed = dispatch("mem_curate", {"action": "propose", "kind": "decision", "subject": "backups",
                                   "statement": "Nightly restic to NAS, weekly offsite to B2.", "basis": "inferred",
                                   "confidence": "medium", "evidence": [f"episode:{top['episode_id']}"]}, "s_backup")
rid = proposed.get("record", {}).get("id")
check("agent proposal stored as proposed", proposed.get("record", {}).get("status") == "proposed", proposed)
reviewed = hermes("hermes-mem", "curated", "review", str(rid), "--accept")
listed = dispatch("mem_curated", {"kind": "decision"}, "s_backup")
check("operator acceptance via CLI is visible to the agent", reviewed.returncode == 0
      and [(r["id"], r["status"]) for r in listed["records"]] == [(rid, "accepted")], listed)
mem_db = __import__("sqlite3").connect((home / "hermes-mem" / "mem.db").as_uri() + "?mode=ro", uri=True)
persisted = (mem_db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone(),
             mem_db.execute("SELECT status, author FROM curated_records WHERE id=?", (rid,)).fetchone(),
             mem_db.execute("SELECT action, author FROM curated_events WHERE record_id=? ORDER BY id", (rid,)).fetchall())
mem_db.close()
check("curated record and its history persisted in mem.db (schema 2), with no Markdown store",
      persisted == (("2",), ("accepted", "agent"), [("propose", "agent"), ("accept", "operator")])
      and not (home / "hermes-mem" / "curated").exists(), persisted)

# --- Unsupported processing policy: the host reports an actionable load error -------------
mem_config = home / "hermes-mem" / "config.yaml"
mem_config.write_text(mem_config.read_text() + "processing:\n  retain: llm\n")
hp._reset_plugin_managers_for_tests()
hp.discover_plugins(force=True)
entry = {p.get("name") or p.get("key"): p for p in hp.get_plugin_manager().list_plugins()}.get("hermes-mem") or {}
check("unsupported processing.retain makes the host report an actionable plugin load error",
      not entry.get("enabled") and "processing.retain: 'llm' is not implemented" in str(entry.get("error"))
      and "Set processing.retain: none" in str(entry.get("error")), entry)
refused = hermes("hermes-mem", "status")
check("refused plugin registers no tools, hooks or CLI, so no indexing runs",
      (entry.get("tools"), entry.get("hooks"), entry.get("commands")) == (0, 0, 0) and refused.returncode != 0
      and "'hermes-mem' is not a `hermes` command" in refused.stdout + refused.stderr,
      [entry, refused.stdout[-400:] + refused.stderr[-400:]])
check("state.db rows unchanged by the post-turn hook, background sync, curation and the refused load",
      state_digest() == state_digest_after_host_writes)

out_path = work / "host-integration.json"
out_path.write_text(json.dumps({"results": results, "plugins_list": plugins_list.stdout[-2000:]}, ensure_ascii=False, indent=2))
failed = [r for r in results if not r["ok"]]
print(f"{len(results) - len(failed)}/{len(results)} host checks passed")
sys.exit(1 if failed else 0)
