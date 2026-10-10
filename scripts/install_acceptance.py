"""Installed-copy acceptance and host compatibility check, through the documented user paths.

Usage:
  python scripts/install_acceptance.py --host-source PATH --host-python SELECTOR --work DIR [--cache-home DIR]

`--host-source` is a Hermes source tree and `--host-python` any `uv venv --python` selector
(3.13, /usr/bin/python3.12, ...). Point both at a candidate host to check an update without
touching a live install. Everything lives under --work: a host venv built from the host's
own pyproject, an isolated HERMES_HOME, and a git snapshot of this checkout (tracked and
untracked, not ignored) that is installed like any other plugin:

  hermes plugins install file://<snapshot>/hermes-mem --enable --no-deps
  hermes hermes-mem setup
  hermes hermes-mem index --dry-run / index

Every command runs as a subprocess from an unrelated cwd, with the default embedding profile
(BGE-M3 INT8) and the default shared model cache under --cache-home. Curated Markdown left by the
previous release is imported into mem.db. The host venv is then deleted and rebuilt, and the
plugin reinstalled with --force; the engine runtime, store and curated records must survive
both. This script needs only the stdlib; it never imports Hermes.
"""

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HEAVY = ("numpy", "onnxruntime", "fastembed", "tokenizers", "huggingface_hub")

results: list[dict] = []


def check(name: str, ok: bool, detail=None) -> bool:
    results.append({"check": name, "ok": bool(ok), "detail": None if ok else detail})
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f" :: {str(detail)[-1500:]}"), flush=True)
    return bool(ok)


def run(argv: list, *, env=None, cwd=None, timeout=1800) -> subprocess.CompletedProcess:
    return subprocess.run([str(a) for a in argv], env=env, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def last_json(text: str):
    """The JSON document a command printed last (Hermes may print banners before it)."""
    for start in [i for i, ch in enumerate(text) if ch in "{[" and (i == 0 or text[i - 1] == "\n")][::-1]:
        try:
            return json.loads(text[start:])
        except ValueError:
            continue
    return None


def sha_file(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def state_digest(state_db: Path) -> str:
    conn = sqlite3.connect(state_db.resolve().as_uri() + "?mode=ro", uri=True)
    rows = conn.execute("SELECT * FROM messages ORDER BY id").fetchall() + conn.execute("SELECT * FROM sessions ORDER BY id").fetchall()
    conn.close()
    return hashlib.sha256(repr(rows).encode()).hexdigest()


def snapshot(dest: Path) -> Path:
    """A local git package of this checkout as it is now, so the install clones the real tree."""
    files = run(["git", "-C", REPO, "ls-files", "-co", "--exclude-standard", "-z"]).stdout.split("\0")
    pkg = dest / "hermes-mem"
    for rel in filter(None, files):
        src = REPO / rel
        if src.is_file():
            (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, pkg / rel)
    env = dict(os.environ, GIT_AUTHOR_NAME="acceptance", GIT_AUTHOR_EMAIL="acceptance@localhost",
               GIT_COMMITTER_NAME="acceptance", GIT_COMMITTER_EMAIL="acceptance@localhost")
    for argv in (["git", "init", "-q", pkg], ["git", "-C", pkg, "add", "-A"],
                 ["git", "-C", pkg, "commit", "-q", "-m", "hermes-mem acceptance snapshot"]):
        done = run(argv, env=env)
        if done.returncode != 0:
            raise SystemExit(f"snapshot failed: {done.stderr}")
    return pkg


class Host:
    def __init__(self, source: Path, selector: str, venv: Path, home: Path, cwd: Path, cache_home: Path):
        self.source, self.selector, self.venv, self.home, self.cwd = source, selector, venv, home, cwd
        self.python = venv / "bin" / "python"
        self.env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "HERMES_SAFE_MODE")}
        self.env.update(HERMES_HOME=str(home), PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE="1",
                        XDG_CACHE_HOME=str(cache_home))

    def build(self) -> subprocess.CompletedProcess:
        """A host environment from the host's own pyproject; nothing from hermes-mem goes into it."""
        done = run(["uv", "venv", "-q", "--python", self.selector, self.venv])
        if done.returncode != 0:
            return done
        version = run([self.python, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"]).stdout.strip()
        reqs = self.venv.parent / f"host-reqs-{version}.txt"
        if not reqs.exists():
            done = run(["uv", "pip", "compile", "-q", "--python-version", version, self.source / "pyproject.toml", "-o", reqs])
            if done.returncode != 0:
                return done
        return run(["uv", "pip", "install", "-q", "--python", self.python, "-r", reqs])

    def packages(self) -> str:
        return run(["uv", "pip", "freeze", "--python", self.python]).stdout

    def _env(self, home: Path | None, extra: dict | None) -> dict:
        return {**self.env, **({"HERMES_HOME": str(home)} if home else {}), **(extra or {})}

    def hermes(self, *args, home: Path | None = None, extra: dict | None = None) -> subprocess.CompletedProcess:
        return run([self.python, "-m", "hermes_cli.main", *args], env=self._env(home, extra), cwd=self.cwd)

    def py(self, code: str, home: Path | None = None) -> subprocess.CompletedProcess:
        return run([self.python, "-c", code], env=self._env(home, None), cwd=self.cwd)


SEED = r"""
from hermes_state import SessionDB
import os
from pathlib import Path
db = SessionDB(db_path=Path(os.environ["HERMES_HOME"]) / "state.db")
db.create_session("s_price", "cli")
db.append_message("s_price", "user", "Нова тарифна політика: базовий план 9 євро на місяць, студентам безкоштовно")
db.append_message("s_price", "assistant", None, tool_calls=[{"id": "c1", "type": "function",
                  "function": {"name": "terminal", "arguments": "{\"cmd\": \"cat prices.csv\"}"}}])
db.append_message("s_price", "tool", "toolonly-7f3e9 basic,9", tool_call_id="c1", tool_name="terminal")
db.append_message("s_price", "assistant", "Записав: базовий план €9/міс, для студентів безкоштовно.")
db.create_session("s_refund", "cli")
db.append_message("s_refund", "user", "Refunds: full refund within 14 days, none after a downgrade")
db.append_message("s_refund", "assistant", "Noted: 14-day full refund, no refund after a downgrade.")
db.create_session("s_vpn", "cli")
db.append_message("s_vpn", "user", "Use WireGuard on the travel router")
db.append_message("s_vpn", "assistant", "Configured WireGuard on the travel router.")
db.create_session("s_cron", "cron")
db.append_message("s_cron", "user", "Daily digest: prices")
db.append_message("s_cron", "assistant", "All green.")
db.close()
"""

DISPATCH = r"""
import json, sys
heavy = lambda: sorted(m for m in %r if m in sys.modules)
from hermes_cli import plugins as hp
hp.discover_plugins(force=True)
import model_tools
after_load = heavy()  # the host's own tools may import these; only what the plugin adds is gated
out = json.loads(model_tools.handle_function_call(
    "mem_search", {"query": "taryfy dlia studentiv", "mode": "semantic", "variants": ["тарифи для студентів", "student pricing"]},
    session_id="s_price"))
leak = json.loads(model_tools.handle_function_call("mem_search", {"query": "toolonly-7f3e9", "mode": "keyword"}, session_id="s_price"))
curated = json.loads(model_tools.handle_function_call("mem_curated", {"kind": "decision"}, session_id="s_price"))
loaded = sorted({getattr(m, "__file__", None) or "" for n, m in list(sys.modules.items()) if n.endswith("hermes_mem.runtime")})
print(json.dumps({"out": out, "leak": leak["results"], "curated": curated, "loaded_from": loaded, "heavy_after_load": after_load,
                  "heavy_added_by_search": sorted(set(heavy()) - set(after_load))}, ensure_ascii=False))
""" % (HEAVY,)


BGE_REVISION = "4de13258303883538bd53b696b452bf8099f0858"
# What the previous release's Markdown store wrote, including a revision chain.
LEGACY = {
    "decision/decision-20261001-a1b2c3.md": (
        "---\nid: decision-20261001-a1b2c3\nkind: decision\nscope: owner\nsubject: pricing\nstatus: accepted\n"
        "basis: inferred\nconfidence: high\nauthor: operator\ncreated_at: 1790000000.0\nupdated_at: 1790000000.0\n"
        "reviewed_at: 1790000000.0\n---\nStudents get the basic plan free.\n"
    ),
    "intent/intent-20261001-000001.md": (
        "---\nid: intent-20261001-000001\nkind: intent\nscope: owner\nsubject: owner\nstatus: superseded\n"
        "basis: inferred\nconfidence: low\nauthor: agent\nsuperseded_by: intent-20261002-000002\n"
        "created_at: 1790000001.0\nupdated_at: 1790000100.0\n---\nWants a NAS.\n"
    ),
    "intent/intent-20261002-000002.md": (
        "---\nid: intent-20261002-000002\nkind: intent\nscope: owner\nsubject: owner\nstatus: proposed\n"
        "basis: inferred\nconfidence: medium\nauthor: agent\nsupersedes: intent-20261001-000001\n"
        "created_at: 1790000100.0\nupdated_at: 1790000100.0\n---\nWants a quiet NAS under 300 EUR.\n"
    ),
}
LEGACY_IDS = [rel.split("/")[1][:-3] for rel in LEGACY]


def tree_digest(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): sha_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


def sqlite_state(home: Path) -> dict:
    conn = sqlite3.connect((home / "hermes-mem" / "mem.db").as_uri() + "?mode=ro", uri=True)
    try:
        return {"schema_version": conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0],
                "records": conn.execute("SELECT COUNT(*) FROM curated_records").fetchone()[0],
                "events": conn.execute("SELECT COUNT(*) FROM curated_events").fetchone()[0],
                "statuses": dict(conn.execute("SELECT id, status FROM curated_records").fetchall())}
    finally:
        conn.close()


def curated_rows(home: Path) -> int:
    return sqlite_state(home)["records"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host-source", type=Path, required=True)
    parser.add_argument("--host-python", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--cache-home", type=Path, help="XDG cache root whose hermes-mem/models is reused (default: under --work)")
    args = parser.parse_args()
    work = args.work.resolve()
    if work.exists():
        shutil.rmtree(work)
    elsewhere = work / "elsewhere"
    elsewhere.mkdir(parents=True)
    home = work / "hermes-home"
    (home / "hermes-mem").mkdir(parents=True)
    cache_home = (args.cache_home or work / "cache").resolve()
    models = cache_home / "hermes-mem" / "models"
    # No cache_dir and no profile: the defaults under test. A model's first use downloads it inside
    # one engine request, hence the long timeout.
    config = "embedding:\n  threads: 2\n  batch_size: 8\nengine:\n  timeout: 1800\n"
    (home / "hermes-mem" / "config.yaml").write_text(config)
    legacy = home / "hermes-mem" / "curated"
    for rel, text in LEGACY.items():
        (legacy / rel).parent.mkdir(parents=True, exist_ok=True)
        (legacy / rel).write_text(text, encoding="utf-8")
    legacy_digest = tree_digest(legacy)
    second = work / "second-profile"
    (second / "hermes-mem").mkdir(parents=True)
    (second / "hermes-mem" / "config.yaml").write_text(config)
    pkg = snapshot(work / "package")
    host = Host(args.host_source.resolve(), args.host_python, work / "host-venv", home, elsewhere, cache_home)

    built = host.build()
    if not check("host venv built from the host's own pyproject", built.returncode == 0, built.stderr):
        return finish(work)
    before_packages = host.packages()

    # --- Install and setup through the documented commands --------------------------------
    installed = host.hermes("plugins", "install", f"file://{pkg}", "--enable", "--no-deps")
    target = home / "plugins" / "hermes-mem"
    check("`hermes plugins install file://… --enable --no-deps` installs a copy into HERMES_HOME",
          installed.returncode == 0 and (target / "plugin.yaml").exists(), installed.stdout[-1500:] + installed.stderr[-1500:])
    listed = host.hermes("plugins", "list")
    check("`hermes plugins list` shows hermes-mem", listed.returncode == 0 and "hermes-mem" in listed.stdout, listed.stdout[-800:])
    status0 = last_json(host.hermes("hermes-mem", "status").stdout) or {}
    check("status before setup diagnoses a missing engine and provisions nothing",
          status0.get("engine", {}).get("state") == "missing" and not (home / "hermes-mem" / "runtime").exists(), status0.get("engine"))
    check("the installed copy, not the checkout, is what runs",
          str(target) in str(status0.get("engine", {}).get("lock")) and str(REPO) not in json.dumps(status0), status0.get("engine"))
    check("default profile is the pinned BGE-M3 INT8 space and the model cache is the shared per-user default",
          status0.get("active_profile") == "bge-m3-int8" and status0.get("active_space", {}).get("revision") == BGE_REVISION
          and status0.get("model_cache") == str(models), [status0.get("active_space"), status0.get("model_cache")])
    check("status flags previous-release curated Markdown as not imported",
          (status0.get("curated") or {}).get("legacy_markdown", {}).get("not_imported") == len(LEGACY)
          and status0["curated"]["records"] == {}, status0.get("curated"))

    setup = host.hermes("hermes-mem", "setup")
    setup_out = last_json(setup.stdout) or {}
    check("`hermes hermes-mem setup` provisions the locked engine runtime",
          setup.returncode == 0 and setup_out.get("status") == "provisioned", setup.stdout[-1500:] + setup.stderr[-1500:])
    runtime = Path(setup_out.get("runtime") or home / "missing")
    ready_before = sha_file(runtime / ".ready")
    check("the engine venv is built from a base interpreter, not the host venv",
          setup_out.get("built_from") and str(host.venv) not in setup_out.get("built_from", ""), setup_out.get("built_from"))
    again = last_json(host.hermes("hermes-mem", "setup").stdout) or {}
    check("repeat setup is a no-op", again.get("status") == "unchanged" and sha_file(runtime / ".ready") == ready_before, again)
    failed = host.hermes("hermes-mem", "setup", "--force", "--python", work / "no-such-python")
    failed_out = last_json(failed.stdout) or {}
    check("failed setup exits 1, keeps the working runtime and leaves no staging",
          failed.returncode == 1 and failed_out.get("status") == "failed" and failed_out.get("kept_existing")
          and sha_file(runtime / ".ready") == ready_before and not list(runtime.parent.glob(".staging-*")), failed_out)

    # --- Previous-release curated Markdown -> mem.db -----------------------------------------
    blocked_dir = work / "broken-md"
    (blocked_dir / "decision").mkdir(parents=True)
    (blocked_dir / "decision" / "decision-20261001-a1b2c3.md").write_text(LEGACY["decision/decision-20261001-a1b2c3.md"])
    (blocked_dir / "decision" / "decision-20261001-bad000.md").write_text("no front matter")
    blocked = host.hermes("hermes-mem", "curated", "import", "--from", blocked_dir)
    check("import with a malformed file exits 2 and publishes nothing",
          blocked.returncode == 2 and "nothing was imported" in blocked.stderr and curated_rows(home) == 0,
          blocked.stdout[-800:] + blocked.stderr[-800:])
    dry = host.hermes("hermes-mem", "curated", "import", "--dry-run")
    dd = last_json(dry.stdout) or {}
    check("curated import --dry-run previews every legacy record and writes none",
          dry.returncode == 0 and dd.get("ok") and len(dd.get("new", [])) == len(LEGACY) and curated_rows(home) == 0, dd)
    imported = host.hermes("hermes-mem", "curated", "import")
    io = last_json(imported.stdout) or {}
    check("curated import publishes the legacy records into mem.db",
          imported.returncode == 0 and io.get("imported") == len(LEGACY) and curated_rows(home) == len(LEGACY),
          imported.stdout[-800:] + imported.stderr[-800:])
    again_import = last_json(host.hermes("hermes-mem", "curated", "import").stdout) or {}
    check("repeat import is a no-op and the Markdown files are untouched",
          again_import.get("imported") == 0 and len(again_import.get("unchanged", [])) == len(LEGACY)
          and tree_digest(legacy) == legacy_digest and curated_rows(home) == len(LEGACY), again_import)
    hist = last_json(host.hermes("hermes-mem", "curated", "history", "intent-20261002-000002").stdout) or {}
    check("imported revision chain and provenance are queryable",
          [r["id"] for r in hist.get("chain", [])] == ["intent-20261001-000001", "intent-20261002-000002"]
          and all(e["action"] == "import" and "#sha256=" in e["source"] for e in hist.get("events", [])), hist)

    # --- Selected history -> episodes -> embeddings -----------------------------------------
    seeded = host.py(SEED)
    if not check("history written through the host's SessionDB", seeded.returncode == 0, seeded.stderr):
        return finish(work)
    state_db = home / "state.db"
    digest = state_digest(state_db)
    store_files = lambda: {f.name: sha_file(f) for f in (home / "hermes-mem").glob("mem.db*")}  # noqa: E731
    store_before = store_files()  # `status` above already created the empty store
    preview = host.hermes("hermes-mem", "index", "--dry-run", "--excerpts", "2")
    pv = last_json(preview.stdout) or {}
    check("index --dry-run reports selected/excluded sessions and message reasons",
          preview.returncode == 0 and pv.get("sessions_selected") == 3 and pv.get("sessions_excluded")
          and pv.get("messages_excluded", {}).get("assistant tool call") == 1 and len(pv.get("samples", [])) == 2, pv)
    check("dry-run writes nothing: state.db and the store unchanged", state_digest(state_db) == digest
          and store_files() == store_before, [store_before, store_files()])
    bounded = host.hermes("hermes-mem", "index", "--batch-sessions", "1", "--max-batches", "1")
    bd = last_json(bounded.stdout) or {}
    check("bounded index stops after one batch and says how to resume",
          bounded.returncode == 0 and bd.get("batches") == 1 and bd.get("sessions_remaining", 0) > 0 and bd.get("resume")
          and '"stage"' in bounded.stderr, bounded.stdout[-1200:] + bounded.stderr[-600:])
    full = host.hermes("hermes-mem", "index")
    fd = last_json(full.stdout) or {}
    check("index resumes to completion and embeds with the real engine",
          full.returncode == 0 and fd.get("sessions_remaining") == 0 and fd.get("embedding", {}).get("status") == "complete",
          full.stdout[-1500:] + full.stderr[-800:])
    repeat = last_json(host.hermes("hermes-mem", "index").stdout) or {}
    sel = repeat.get("selection", {})
    check("repeat index is a no-op", sel.get("inserted") == 0 and sel.get("revised") == 0
          and repeat.get("embedding", {}).get("embedded") == 0, repeat)
    check("state.db unchanged by install, setup and indexing", state_digest(state_db) == digest)

    dispatched = host.py(DISPATCH)
    dd = last_json(dispatched.stdout) or {}
    out = dd.get("out", {})
    check("real tool dispatch: semantic search with variants finds the Ukrainian pricing episode",
          dispatched.returncode == 0 and out.get("search_mode") == "semantic" and "degraded" not in out
          and any("тариф" in r.get("user", "") for r in out.get("results", [])[:2]), dispatched.stdout[-1500:] + dispatched.stderr[-1500:])
    check("variants are reported as provenance", len(out.get("query_provenance", {}).get("semantic_queries") or []) == 3,
          out.get("query_provenance"))
    check("tool output is not indexed", dd.get("leak") == [], dd.get("leak"))
    check("real tool dispatch: mem_curated reads the imported SQLite record for the owner session",
          [r["id"] for r in dd.get("curated", {}).get("records", [])] == ["decision-20261001-a1b2c3"], dd.get("curated"))
    check("embedding through the plugin imported no heavy package into the Hermes process",
          dd.get("heavy_added_by_search") == [], dd)
    check("the Hermes process loaded the installed copy", dd.get("loaded_from") and all(str(target) in p for p in dd["loaded_from"]),
          dd.get("loaded_from"))

    added = host.hermes("hermes-mem", "curated", "add", "decision", "pricing", "Basic plan is 9 EUR per month; students free.",
                        "--basis", "inferred", "--confidence", "medium")
    record = (last_json(added.stdout) or {}).get("id")
    check("operator curated record stored", added.returncode == 0 and record, added.stdout[-800:] + added.stderr[-800:])
    status1 = last_json(host.hermes("hermes-mem", "status").stdout) or {}
    profiles = status1.get("profiles", {})
    check("status shows coverage for every configured profile, including an unembedded one",
          profiles.get("bge-m3-int8", {}).get("pending") == 0 and profiles.get("bge-m3-int8", {}).get("current_vectors", 0) > 0
          and profiles.get("bge-m3-int8", {}).get("failed") == 0 and profiles.get("minilm", {}).get("current_vectors") == 0
          and profiles.get("minilm", {}).get("pending") == status1.get("chunks_current"), profiles)
    check("the pinned BGE snapshot was downloaded into the shared cache",
          (models / "models--Xenova--bge-m3" / "snapshots" / BGE_REVISION).is_dir(), sorted(p.name for p in models.glob("*")))
    second_install = host.hermes("plugins", "install", f"file://{pkg}", "--enable", "--no-deps", home=second)
    other = last_json(host.hermes("hermes-mem", "status", home=second).stdout) or {}
    check("a second profile shares the model cache but not the engine runtime, store or curated records",
          second_install.returncode == 0 and other.get("model_cache") == str(models)
          and other.get("engine", {}).get("state") == "missing"
          and other.get("chunks_current") == 0 and other.get("curated", {}).get("records") == {},
          [other.get("model_cache"), other.get("engine"), other.get("curated")])
    weights = lambda: {k: v for k, v in tree_digest(models).items() if not k.startswith('.locks/')}  # noqa: E731
    cache_before = weights()
    second_setup = host.hermes("hermes-mem", "setup", home=second)
    second_seed = host.py(SEED, home=second)
    offline = host.hermes("hermes-mem", "index", home=second, extra={"HF_HUB_OFFLINE": "1"})
    od = last_json(offline.stdout) or {}
    check("the second profile embeds offline from the shared cache, adding nothing to it",
          second_setup.returncode == 0 and second_seed.returncode == 0 and offline.returncode == 0
          and od.get("embedding", {}).get("status") == "complete" and od.get("embedding", {}).get("embedded", 0) > 0
          and weights() == cache_before,
          [second_setup.stdout[-400:], second_seed.stderr[-400:], offline.stdout[-800:] + offline.stderr[-800:]])
    persisted = sqlite_state(home)
    check("SQLite state persisted: schema 2, curated records and their events", persisted["schema_version"] == "2"
          and persisted["records"] == len(LEGACY) + 1 and persisted["events"] >= len(LEGACY) + 1, persisted)
    check("plugin install, setup and indexing changed no host package", host.packages() == before_packages)

    # --- Replace the host environment, then reinstall the plugin -----------------------------
    episodes = status1.get("episodes")
    shutil.rmtree(host.venv)
    rebuilt = host.build()
    check("host venv deleted and rebuilt", rebuilt.returncode == 0, rebuilt.stderr)
    for label, step in (("after host venv replacement", None),
                        ("after `plugins install --force` reinstall", ("plugins", "install", f"file://{pkg}", "--force", "--enable", "--no-deps"))):
        if step:
            redo = host.hermes(*step)
            check(f"reinstall exits 0 ({label})", redo.returncode == 0, redo.stdout[-800:] + redo.stderr[-800:])
        st = last_json(host.hermes("hermes-mem", "status").stdout) or {}
        check(f"engine runtime, store and coverage retained {label}",
              st.get("engine", {}).get("state") == "provisioned" and sha_file(runtime / ".ready") == ready_before
              and st.get("episodes") == episodes and st.get("profiles", {}).get("bge-m3-int8", {}).get("pending") == 0, st)
        found = host.hermes("hermes-mem", "search", "student pricing", "--mode", "semantic", "--variant", "тарифи для студентів")
        fo = last_json(found.stdout) or {}
        check(f"semantic search works without re-setup {label}", found.returncode == 0 and fo.get("search_mode") == "semantic"
              and "degraded" not in fo and fo.get("results"), found.stdout[-800:] + found.stderr[-800:])
        cur = last_json(host.hermes("hermes-mem", "curated", "list", "--all").stdout) or {}
        check(f"curated records retained {label}", sorted(r["id"] for r in cur.get("records", [])) == sorted([*LEGACY_IDS, record])
              and sqlite_state(home) == persisted, [cur, sqlite_state(home)])
        same = last_json(host.hermes("hermes-mem", "setup").stdout) or {}
        check(f"setup stays a no-op {label}", same.get("status") == "unchanged", same)
    check("state.db unchanged by the whole run", state_digest(state_db) == digest)
    return finish(work)


def finish(work: Path) -> int:
    failed = [r for r in results if not r["ok"]]
    (work / "install-acceptance.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"{len(results) - len(failed)}/{len(results)} install checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
