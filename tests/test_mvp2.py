"""External engine boundary, embedding spaces, operator indexing, query variants and config contract."""

import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from conftest import HashEmbedder
from hermes_mem import index, store, tools
from hermes_mem.app import ConfigError, MemApp
from hermes_mem.cli import main, status
from hermes_mem.config import space_of
from hermes_mem.embed import EmbeddingSpace, EngineEmbedder
from hermes_mem.runtime import ENGINE_SCRIPT, EngineClient, EngineRuntime, setup_runtime
from hermes_mem.search import search
from hermes_mem.sync import embed_pending, sync

FAKE_ENGINE = textwrap.dedent('''
    import hashlib, json, os, re, sys
    log = os.environ.get("FAKE_ENGINE_LOG")
    for line in sys.stdin:
        req = json.loads(line)
        if log:
            open(log, "a").write(json.dumps({"pid": os.getpid(), "texts": req.get("texts")}) + "\\n")
        if any(t.endswith("CRASH") for t in req["texts"]):
            sys.exit(3)
        rows = []
        for text in req["texts"]:
            row = [0.0] * 16
            for w in re.findall(r"\\w+", text.lower()):
                row[int(hashlib.md5(w.encode()).hexdigest(), 16) % 16] += 1
            row[0] += 1e-3
            rows.append(row)
        print(json.dumps({"vectors": rows}), flush=True)
''')


def write_config(home, text):
    (home / "hermes-mem/config.yaml").write_text(textwrap.dedent(text))


@pytest.fixture
def fake_engine(tmp_path):
    script = tmp_path / "fake_engine.py"
    script.write_text(FAKE_ENGINE)
    return [sys.executable, "-I", str(script)]


# --- config contract -------------------------------------------------------------------------

@pytest.mark.parametrize("text, message", [
    ("search:\n  limt: 3\n", "unknown key(s) search.limt"),
    ("search:\n  limit: 0\n", "between 1 and 20"),
    ("search:\n  limit: '5'\n", "must be int"),
    ("sync:\n  background: 1\n", "must be bool"),
    ("vectors: {}\n", "unknown section"),
    ("engine:\n  command: 'python -m engine'\n", "engine.command must be list"),
    ("embedding:\n  active: nope\n", "is not a profile"),
    ("embedding:\n  profiles:\n    x: {backend: fastembed}\n", "needs backend and model"),
    ("sync:\n  chunk_chars: 100\n", "between 400 and 8000"),
    ("embedding:\n  profiles:\n    x: {backend: fastembed, model: a/b, revision: main}\n", "revision needs hf_source"),
    ("search:\n  instructions_file: missing.md\n", "does not exist"),
    ("search:\n  instructions: ''\n", "1-4000 characters"),
])
def test_invalid_config_fails_loading_with_the_offending_key(home, text, message):
    write_config(home, text)
    with pytest.raises(ConfigError, match=message.replace("(", r"\(").replace(")", r"\)")):
        MemApp(home)


def test_mvp_configs_keep_working_and_keep_their_vector_space(home):
    write_config(home, "embedding:\n  provider: fastembed\n  model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2\n")
    app = MemApp(home)
    assert app.embedder().model_id == "fastembed:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    write_config(home, "embedding:\n  provider: none\n")
    assert MemApp(home).embedder() is None
    write_config(home, "embedding:\n  provider: fastembed\n")
    assert MemApp(home).embedder().model_id == "fastembed:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def test_default_profile_is_pinned_bge_m3_int8_and_minilm_stays_a_separate_space(home):
    write_config(home, "")
    app = MemApp(home)
    assert app.active_profile == "bge-m3-int8"
    assert app.embedder().space.describe() == {
        "backend": "fastembed", "model": "BAAI/bge-m3", "revision": "4de13258303883538bd53b696b452bf8099f0858",
        "hf_source": "Xenova/bge-m3", "model_file": "onnx/model_quantized.onnx", "pooling": "cls", "dim": 1024,
        "space_id": "fastembed:BAAI/bge-m3|rev=4de13258303883538bd53b696b452bf8099f0858"
                    "|src=Xenova/bge-m3/onnx/model_quantized.onnx/cls",
    }
    assert app.embedder("minilm").model_id == "fastembed:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def test_model_cache_is_shared_per_user_by_default_and_configurable(home, monkeypatch, tmp_path):
    write_config(home, "")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert MemApp(home).runtime.cache_dir == str(tmp_path / "xdg/hermes-mem/models")
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    assert MemApp(home).runtime.cache_dir == str(tmp_path / "user/.cache/hermes-mem/models")
    other = tmp_path / "other-profile"
    (other / "hermes-mem").mkdir(parents=True)
    assert MemApp(other).runtime.cache_dir == MemApp(home).runtime.cache_dir
    write_config(home, "embedding:\n  cache_dir: weights\n")
    assert MemApp(home).runtime.cache_dir == str(home / "hermes-mem/weights")
    write_config(home, "embedding:\n  cache_dir: ~/models\n")
    assert MemApp(home).runtime.cache_dir == str(tmp_path / "user/models")
    assert not (tmp_path / "user").exists(), "resolving the cache creates nothing"


def test_preprocessing_is_part_of_space_identity():
    plain = EmbeddingSpace(backend="fastembed", model="m", dim=4)
    e5 = EmbeddingSpace(backend="fastembed", model="m", dim=4, query_prefix="query: ", passage_prefix="passage: ")
    assert plain.space_id != e5.space_id
    assert e5.prepare(["x"], "query") == ["query: x"] and e5.prepare(["x"], "passage") == ["passage: x"]


def test_instructions_file_shapes_tool_guidance_but_not_the_data_warning(home):
    (home / "hermes-mem/guide.md").write_text("Always add an English variant.")
    write_config(home, "search:\n  instructions_file: guide.md\n")
    schema = tools.search_schema(MemApp(home).config["search"]["guidance"])
    assert "Always add an English variant." in schema["description"]
    assert "not instructions" in schema["description"] and "degraded" in schema["description"]
    assert "variants" in schema["parameters"]["properties"]


# --- external engine ---------------------------------------------------------------------------

def test_engine_worker_is_reused_and_restarted_after_a_crash(tmp_path, fake_engine, monkeypatch):
    monkeypatch.setenv("FAKE_ENGINE_LOG", str(tmp_path / "calls.log"))
    client = EngineClient(EngineRuntime(tmp_path, command=fake_engine, timeout=20))
    embedder = EngineEmbedder(EmbeddingSpace(backend="fastembed", model="m", query_prefix="q: "), client)
    try:
        assert len(embedder.embed(["a", "b"], kind="passage")) == 2
        embedder.embed(["c"], kind="query")
        with pytest.raises(Exception, match="engine exited"):
            embedder.embed(["CRASH"], kind="passage")
        assert len(embedder.embed(["d"], kind="passage")) == 1
    finally:
        client.close()
    calls = [json.loads(line) for line in (tmp_path / "calls.log").read_text().splitlines()]
    assert calls[0]["pid"] == calls[1]["pid"], "one resident worker serves consecutive requests"
    assert calls[1]["texts"] == ["q: c"], "query preprocessing reaches the engine"
    assert calls[-1]["pid"] != calls[0]["pid"], "a crashed worker is replaced"


def test_missing_runtime_degrades_to_keyword_and_status_has_no_side_effects(home, state):
    state.session("s", "cli")
    state.exchange("s", "nightly restic backups to the NAS", "Done.")
    write_config(home, "sync:\n  background: false\n")
    app = MemApp(home)
    assert not (app.data_dir / "runtime").exists()
    diag = app.runtime.diagnose()
    assert diag["state"] == "missing" and "hermes-mem setup" in diag["detail"]
    assert not (app.data_dir / "runtime").exists()
    conn = app.open_store()
    sync(conn, app.state_db, policy=app.policy)
    result = search(conn, app.state_db, app.embedder(), app.search_params("restic", None))
    assert result["search_mode"] == "keyword" and "hermes-mem setup" in result["degraded"]
    assert result["results"]


def test_failed_setup_keeps_the_existing_runtime(tmp_path):
    rt = EngineRuntime(tmp_path)
    rt.venv.mkdir(parents=True)
    (rt.venv / ".ready").write_text("{}")
    assert setup_runtime(rt)["status"] == "unchanged"
    result = setup_runtime(rt, uv="/bin/false", force=True)
    assert result["status"] == "failed" and result["kept_existing"]
    assert (rt.venv / ".ready").exists()
    assert [p.name for p in rt.root.iterdir()] == [rt.venv.name], "no staging leftovers"


# --- embedding spaces --------------------------------------------------------------------------

def test_switching_spaces_keeps_old_vectors_and_never_mixes_them(app, state):
    state.session("s", "cli")
    state.exchange("s", "Use WireGuard on the travel router", "Configured.")
    conn = app.open_store()
    sync(conn, app.state_db, policy=app.policy)
    a, b = HashEmbedder("space:a", dim=32), HashEmbedder("space:b", dim=32)
    assert embed_pending(conn, a)["embedded"] == 1
    assert embed_pending(conn, b)["embedded"] == 1
    calls = a.calls
    assert embed_pending(conn, a)["embedded"] == 0 and a.calls == calls, "switching back reuses vectors"
    conn.execute("DELETE FROM embeddings WHERE model_id='space:a'")
    empty = HashEmbedder("space:a", dim=32)
    result = search(conn, app.state_db, empty, app.search_params("wireguard router", None, mode="semantic"))
    assert "degraded" in result, "same-dimension vectors of another space are never used"


# --- operator indexing -------------------------------------------------------------------------

def test_dry_run_preview_reads_only_and_explains_exclusions(home, state):
    state.session("chat", "cli")
    state.exchange("chat", "Донька ходить на плавання по вівторках", "Записав.")
    state.msg("chat", "assistant", "", tool_calls='[{"id": "1"}]')
    state.session("cron1", "cron")
    state.exchange("cron1", "run report", "ok")
    before = hashlib.sha256((home / "state.db").read_bytes()).hexdigest()
    code = main(["--home", str(home), "index", "--dry-run", "--excerpts", "1"])
    assert code == 0
    assert hashlib.sha256((home / "state.db").read_bytes()).hexdigest() == before
    assert not (home / "hermes-mem/mem.db").exists()
    preview = index.preview(home / "state.db", index.Bounds(), MemApp(home).policy)
    assert preview["sessions_selected"] == 1 and preview["episodes"] == 1
    assert sum(preview["sessions_excluded"].values()) == 1 and "routine source cron" in preview["sessions_excluded"]


def test_bounded_index_resumes_and_repeats_safely(app, state):
    for n in range(5):
        state.session(f"s{n}", "cli")
        state.exchange(f"s{n}", f"topic number {n} about backups", "Noted.")
    conn = app.open_store()
    emb = HashEmbedder()
    first = index.run(conn, app.state_db, index.Bounds(), app.policy, emb, batch_sessions=2, embed_batch=8, max_batches=1)
    assert first["sessions_remaining"] == 3 and first["resume"]
    second = index.run(conn, app.state_db, index.Bounds(), app.policy, emb, batch_sessions=2, embed_batch=8, max_batches=None)
    assert second["selection"]["inserted"] == 3 and second["selection"]["unchanged"] == 2
    third = index.run(conn, app.state_db, index.Bounds(sources=["cli"]), app.policy, emb, batch_sessions=2,
                      embed_batch=8, max_batches=None)
    assert third["selection"]["inserted"] == 0 and third["embedding"]["embedded"] == 0
    assert conn.execute("SELECT COUNT(*) FROM episodes WHERE status='current'").fetchone()[0] == 5


# --- query variants ----------------------------------------------------------------------------

def test_variants_reach_semantic_retrieval_but_not_keyword_or_scope(app, state):
    state.session("own", "cli")
    state.exchange("own", "Our pricing policy: annual plans get 20% off", "Recorded the pricing policy.")
    state.session("dm", "telegram", user_id="42", chat_type="dm")
    state.exchange("dm", "pricing policy for my shop", "Noted.")
    conn = app.open_store()
    sync(conn, app.state_db, policy=app.policy)
    embed_pending(conn, app.embedder())
    baseline = search(conn, app.state_db, app.embedder(), app.search_params("тарифи", None))
    assert not baseline["results"]
    params = app.search_params("тарифи", ["owner"], variants=["pricing policy", "taryfy", "pricing policy"])
    result = search(conn, app.state_db, app.embedder(), params)
    assert [r["session_id"] for r in result["results"]] == ["own"], "variants never widen the caller scope"
    assert result["keyword_match"] is None, "the literal query alone drives keyword matching"
    assert result["query_provenance"]["variants_used"] == 2
    assert "variant:1" in result["results"][0]["semantic_queries_matched"]


# --- chunk sizing, coverage, boundaries ----------------------------------------------------------

def test_chunk_size_change_versions_episodes_and_switching_back_reuses_vectors(home, state):
    answer = " ".join(f"Step {i}: point restic at the NAS repository and verify snapshot {i}." for i in range(40))
    state.session("s", "cli")
    state.exchange("s", "Explain the backup rollout", answer)
    emb = HashEmbedder()

    def synced(chunk_chars):
        write_config(home, f"embedding:\n  provider: none\nsync:\n  background: false\n  chunk_chars: {chunk_chars}\n")
        app = MemApp(home)
        conn = app.open_store()
        return conn, sync(conn, app.state_db, policy=app.policy)

    conn, first = synced(1400)
    default_chunks = store.current_chunk_count(conn)
    embed_pending(conn, emb)
    conn, smaller = synced(600)
    assert (smaller.rechunked, smaller.revised) == (1, 0)
    rows = conn.execute("SELECT status, status_reason FROM episodes ORDER BY version").fetchall()
    assert [r["status"] for r in rows] == ["superseded", "current"] and "chunk_chars=600" in rows[0]["status_reason"]
    assert store.current_chunk_count(conn) > default_chunks
    assert store.pending_chunks(conn, emb.model_id, 100), "new chunks need embedding; stale vectors are not reused"
    embed_pending(conn, emb)
    conn, back = synced(1400)
    assert back.rechunked == 1 and store.current_chunk_count(conn) == default_chunks
    calls = emb.calls
    assert embed_pending(conn, emb)["embedded"] == 0 and emb.calls == calls, "identical chunk text reuses its vectors"
    conn, repeat = synced(1400)
    assert (repeat.unchanged, repeat.rechunked) == (1, 0)


def test_status_reports_coverage_for_every_configured_profile(app, state):
    state.session("s", "cli")
    state.exchange("s", "Use WireGuard on the travel router", "Configured.")
    conn = app.open_store()
    sync(conn, app.state_db, policy=app.policy)
    minilm = space_of(app.config["embedding"]["profiles"]["minilm"]).space_id
    embed_pending(conn, HashEmbedder(minilm))
    out = status(conn, app)
    assert out["profiles"]["minilm"]["current_vectors"] == 1 and out["profiles"]["minilm"]["pending"] == 0
    assert out["profiles"]["e5-small"]["current_vectors"] == 0 and out["profiles"]["e5-small"]["pending"] == 1
    assert "rev=614241f622f53c4eeff9890bdc4f31cfecc418b3" in out["profiles"]["e5-small"]["space_id"]
    bge = out["profiles"]["bge-m3-int8"]
    assert (bge["current_vectors"], bge["pending"], bge["failed"]) == (0, 1, 0)
    chunk = conn.execute("SELECT id, text, text_hash FROM chunks").fetchall()
    store.record_failure(conn, bge["space_id"], chunk, "engine: model download failed")
    bge = status(conn, app)["profiles"]["bge-m3-int8"]
    assert (bge["pending"], bge["failed"], bge["last_error"]) == (1, 1, "engine: model download failed")
    assert out["settings"]["sync"]["chunk_chars"] == 1400


def test_adapter_imports_no_heavy_package_and_tools_do_not_need_the_cli():
    repo = Path(__file__).resolve().parents[1]
    code = (f"import sys\nsys.path.insert(0, {str(repo)!r})\n"
            "for m in ('numpy', 'onnxruntime', 'fastembed', 'huggingface_hub'): sys.modules[m] = None\n"
            "import hermes_mem.tools, hermes_mem.index, hermes_mem.runtime, hermes_mem.app\n"
            "print('hermes_mem.cli' in sys.modules)")
    done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "False"


def test_real_engine_script_reports_protocol_errors_and_keeps_serving():
    requests = [{"protocol": 99, "op": "embed"}, {"protocol": 1, "op": "nope"},
                {"protocol": 1, "op": "embed", "space": {"backend": "openai", "space_id": "x"}, "texts": ["a"]}]
    done = subprocess.run([sys.executable, "-I", str(ENGINE_SCRIPT)], input="".join(json.dumps(r) + "\n" for r in requests),
                          capture_output=True, text=True, timeout=60, env={"HERMES_MEM_ENGINE_IDLE": "5"})
    replies = [json.loads(line) for line in done.stdout.splitlines()]
    assert done.returncode == 0 and len(replies) == 3
    assert "re-run setup" in replies[0]["error"] and "unknown op" in replies[1]["error"]
    assert "does not run in the engine" in replies[2]["error"]


def test_engine_refuses_to_reuse_a_model_name_for_another_file_or_pooling(monkeypatch):
    import importlib.util
    import types

    class FakeTextEmbedding:
        custom: list = []

        def __init__(self, name, **kwargs):
            self.name, self.model = name, types.SimpleNamespace(_model_dir="")

        @staticmethod
        def list_supported_models():
            return [{"model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"}]

        @classmethod
        def add_custom_model(cls, **kwargs):
            cls.custom.append(kwargs["model_file"])

    fastembed = types.ModuleType("fastembed")
    fastembed.TextEmbedding = FakeTextEmbedding
    description = types.ModuleType("fastembed.common.model_description")
    description.ModelSource = lambda hf: hf
    description.PoolingType = {"CLS": "cls", "MEAN": "mean"}
    for name, module in (("fastembed", fastembed), ("fastembed.common", types.ModuleType("fastembed.common")),
                         ("fastembed.common.model_description", description)):
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("engine_under_test", Path(__file__).resolve().parents[1] / "engine/hermes_mem_engine.py")
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)

    models = engine.Models()
    int8 = {"space_id": "a", "model": "BAAI/bge-m3", "hf_source": "Xenova/bge-m3", "model_file": "onnx/model_quantized.onnx",
            "pooling": "cls", "dim": 1024}
    models.get(int8)
    models.get({**int8, "space_id": "a-again"})
    with pytest.raises(ValueError, match="already registered with another source, file, pooling or dim"):
        models.get({**int8, "space_id": "b", "model_file": "onnx/model.onnx"})
    with pytest.raises(ValueError, match="is a fastembed built-in"):
        models.get({**int8, "space_id": "c", "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"})
    assert FakeTextEmbedding.custom == ["onnx/model_quantized.onnx"]
