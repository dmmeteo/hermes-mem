"""Compare embedding profiles on one frozen corpus with real, locally executed models.

Usage: python scripts/compare_models.py WORK_DIR EMBED_CONFIG_YAML OUT_JSON [PROFILE ...]
Default profiles: bge-m3-int8 minilm (the default and its control), run one after the other in separate engine processes.

The corpus is eval/paraphrase_uk_en.json: the original smoke set plus its `pricing` section
(Ukrainian, English and transliterated pricing-policy items and unrelated distractors). Every
profile indexes the same episodes and chunks. Each question runs as the literal query
(baseline) and, where the dataset froze them, with its semantic variants (enhanced), in
semantic and hybrid mode. Follow-ups run with and without their conversation.

Gates (fixed before the first run) are about the run being real, not about which model wins:
every profile embedded the whole corpus with real vectors, no search degraded to keyword,
every profile has its own embedding space, and a pinned revision is the snapshot that ran.
Quality (hit@k, MRR, ranks) is reported only. The data is synthetic and small, so no number
here supports a production quality claim.
"""

import hashlib
import json
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from hermes_mem import store as mem_store  # noqa: E402
from hermes_mem.app import MemApp  # noqa: E402
from hermes_mem.search import search  # noqa: E402
from hermes_mem.sync import embed_pending, sync  # noqa: E402

DATASET = REPO / "eval/paraphrase_uk_en.json"
LIMIT = 10


def build_home(work: Path, embed_config: Path, sessions: dict) -> Path:
    home = work / "home"
    (home / "hermes-mem").mkdir(parents=True)
    config = yaml.safe_load(embed_config.read_text()) or {}
    # Profiles come from the built-in table; the legacy single-model keys would pin one model.
    embedding = {k: v for k, v in (config.get("embedding") or {}).items()
                 if k not in ("provider", "model", "base_url", "api_key_env")}
    config["embedding"] = embedding
    (home / "hermes-mem/config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    schema = json.loads((REPO / "tests/fixtures/state_schema.json").read_text())
    conn = sqlite3.connect(home / "state.db")
    for name, sql in schema.items():
        if not name.startswith("messages_fts_") and name != "sqlite_sequence":
            conn.execute(sql)
    clock = 1_790_000_000.0
    for sid, exchanges in sessions.items():
        conn.execute("INSERT INTO sessions(id, source, started_at) VALUES (?, 'cli', ?)", (sid, clock))
        for user, assistant in exchanges:
            for role, text in (("user", user), ("assistant", assistant)):
                clock += 1
                conn.execute("INSERT INTO messages(session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                             (sid, role, text, clock))
    conn.commit()
    conn.close()
    return home


def with_profile(home: Path, profile: str) -> MemApp:
    path = home / "hermes-mem/config.yaml"
    config = yaml.safe_load(path.read_text())
    config["embedding"]["active"] = profile
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return MemApp(home)


def rank_of(expect: str, result: dict) -> int | None:
    for i, r in enumerate(result["results"], start=1):
        if r["session_id"] == expect:
            return i
    return None


def metrics(ranks: list) -> dict:
    n = len(ranks)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "hit@1": round(sum(1 for r in ranks if r == 1) / n, 3),
        "hit@5": round(sum(1 for r in ranks if r and r <= 5) / n, 3),
        "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
    }


def git_revision() -> dict:
    def git(*args):
        done = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
        return done.stdout.strip() if done.returncode == 0 else None

    status = git("status", "--porcelain")
    return {"head": git("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def run_profile(home: Path, profile: str, queries: list, follow_ups: list) -> dict:
    app = with_profile(home, profile)
    embedder = app.embedder()
    conn = app.open_store()
    started = time.monotonic()
    embedding = embed_pending(conn, embedder, batch_size=app.config["embedding"]["batch_size"], max_batches=10**6)
    index_seconds = round(time.monotonic() - started, 3)
    index_load = embedder.load_seconds
    chunks = mem_store.current_chunk_count(conn)
    space = next((s for s in mem_store.space_coverage(conn) if s["model_id"] == embedder.model_id), {})
    conn.close()
    app.close()  # the next search starts a fresh engine: a cold query

    app = with_profile(home, profile)
    embedder = app.embedder()
    conn = app.open_store()
    rows, degraded, latencies = [], [], []
    cold = None
    for q in queries:
        strategies = [("baseline", [])] + ([("variants", q["variants"])] if q.get("variants") else [])
        for mode in ("semantic", "hybrid"):
            for strategy, variants in strategies:
                t0 = time.monotonic()
                res = search(conn, app.state_db, embedder, app.search_params(q["query"], None, mode=mode, limit=LIMIT,
                                                                             variants=variants))
                elapsed = time.monotonic() - t0
                if cold is None:
                    cold = {"seconds": round(elapsed, 3), "engine_load_seconds": embedder.load_seconds}
                else:
                    latencies.append(elapsed)
                if res["search_mode"] != mode or "degraded" in res:
                    degraded.append({"id": q["id"], "mode": mode, "strategy": strategy, "detail": res.get("degraded")})
                rows.append({"id": q["id"], "set": q["set"], "lang": q["lang"], "mode": mode, "strategy": strategy,
                             "rank": rank_of(q["expect"], res),
                             "top": res["results"][0]["session_id"] if res["results"] else None,
                             "semantic_queries": res.get("query_provenance", {}).get("semantic_queries")})
    follow = []
    for f in follow_ups:
        for context in (False, True):
            for strategy, variants in [("baseline", [])] + ([("variants", f["variants"])] if f.get("variants") else []):
                res = search(conn, app.state_db, embedder, app.search_params(
                    f["query"], None, mode="hybrid", limit=LIMIT, variants=variants,
                    conversation=f["conversation"] if context else None))
                if "degraded" in res:
                    degraded.append({"id": f["id"], "mode": "hybrid", "strategy": strategy, "detail": res["degraded"]})
                follow.append({"id": f["id"], "with_context": context, "strategy": strategy,
                               "rank": rank_of(f["expect"], res)})
    conn.close()
    app.close()

    summary = {}
    for mode in ("semantic", "hybrid"):
        for strategy in ("baseline", "variants"):
            for subset in ("all", "original", "pricing"):
                ranks = [r["rank"] for r in rows if r["mode"] == mode and r["strategy"] == strategy
                         and (subset == "all" or r["set"] == subset)]
                summary[f"{mode}/{strategy}/{subset}"] = metrics(ranks)
    # Fair baseline for the enhanced arm: the same questions (those with frozen variants) without them.
    with_variants = {r["id"] for r in rows if r["strategy"] == "variants"}
    for mode in ("semantic", "hybrid"):
        summary[f"{mode}/baseline/with-variant-questions"] = metrics(
            [r["rank"] for r in rows if r["mode"] == mode and r["strategy"] == "baseline" and r["id"] in with_variants])
    dim = space.get("dim")
    return {
        "profile": profile,
        "space": embedder.space.describe(),
        "source_revision": embedder.source_revision,
        "engine": app.runtime.diagnose(),
        "embedding_run": embedding,
        "vectors": {"dim": dim, "bytes_per_vector": dim * 4 if dim else None,
                    "stored": space.get("stored_vectors", 0), "current": space.get("current_vectors", 0),
                    "chunks_current": chunks},
        "latency": {
            "index_seconds_incl_cold_engine_start": index_seconds,
            "index_engine_load_seconds": index_load,
            "cold_first_query": cold,
            "warm_query_median_seconds": round(statistics.median(latencies), 4) if latencies else None,
            "warm_query_p90_seconds": round(sorted(latencies)[int(0.9 * (len(latencies) - 1))], 4) if latencies else None,
            "warm_queries": len(latencies),
            "note": "wall time in this process, including the pure-Python vector scan of the whole store",
        },
        "summary": summary,
        "follow_ups": follow,
        "degraded": degraded,
        "per_query": rows,
    }


def main() -> int:
    work, embed_config, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    profiles = sys.argv[4:] or ["bge-m3-int8", "minilm"]
    data = json.loads(DATASET.read_text())
    pricing = data["pricing"]
    sessions = {**data["sessions"], **pricing["sessions"]}
    queries = [{**q, "set": "original"} for q in data["queries"]] + [{**q, "set": "pricing"} for q in pricing["queries"]]
    follow_ups = data["follow_ups"] + pricing["follow_ups"]
    if work.exists():
        shutil.rmtree(work)
    home = build_home(work, embed_config, sessions)

    app = MemApp(home)
    conn = app.open_store()
    selection = sync(conn, app.state_db, policy=app.policy).as_dict()
    episodes = conn.execute("SELECT COUNT(*) FROM episodes WHERE status='current'").fetchone()[0]
    conn.close()
    app.close()

    runs = [run_profile(home, profile, queries, follow_ups) for profile in profiles]
    keyword = {}
    app = MemApp(home)
    conn = app.open_store()
    for q in queries:
        res = search(conn, app.state_db, None, app.search_params(q["query"], None, mode="keyword", limit=LIMIT))
        keyword[q["id"]] = rank_of(q["expect"], res)
    conn.close()
    app.close()

    gates = {
        "every_profile_ran": [r["profile"] for r in runs] == profiles and len(profiles) >= 2,
        "full_coverage_real_vectors": all(r["embedding_run"]["status"] == "complete"
                                          and r["vectors"]["current"] == r["vectors"]["chunks_current"] > 0 for r in runs),
        "no_degraded_search": all(not r["degraded"] for r in runs),
        "distinct_spaces": len({r["space"]["space_id"] for r in runs}) == len(runs),
        "pinned_revision_ran": all(r["source_revision"] == r["space"]["revision"]
                                   for r in runs if len(r["space"].get("revision", "")) == 40),
    }
    report = {
        "dataset": {"path": str(DATASET.relative_to(REPO)), "sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
                    "provenance": [data["provenance"], pricing["provenance"]],
                    "sessions": len(sessions), "episodes": episodes, "queries": len(queries), "follow_ups": len(follow_ups)},
        "source": git_revision(),
        "selection": selection,
        "chunking": {"chunk_chars": app.policy.chunk_chars, "same_chunks_for_every_profile": True},
        "query_strategy": {
            "baseline": "the literal query only",
            "variants": "literal query plus the dataset's frozen variants (semantic arm only; keyword stays literal)",
            "fusion": "RRF across the keyword arm and one semantic arm per query text, deduplicated by episode",
            "limit": LIMIT,
        },
        "keyword": {"summary": metrics(list(keyword.values())), "per_query": keyword},
        "profiles": runs,
        "gates": gates,
        "caveat": "Synthetic, hand-labelled, small. Use it to compare candidates, not to claim production quality.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({
        "profiles": {r["profile"]: {"space_id": r["space"]["space_id"], "source_revision": r["source_revision"],
                                    "dim": r["vectors"]["dim"], "latency": r["latency"],
                                    "summary": {k: v for k, v in r["summary"].items() if k.endswith("/all")
                                                or k.endswith("/pricing")}} for r in runs},
        "keyword": report["keyword"]["summary"],
        "gates": gates,
    }, ensure_ascii=False, indent=1))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
