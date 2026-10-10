"""Ukrainian/English paraphrase smoke eval with a real multilingual embedding model.

Usage: python scripts/eval_paraphrase.py WORK_DIR EMBED_CONFIG_YAML [OUT_JSON]
Compares keyword, semantic and hybrid retrieval on independent expected session ids.
Gates (fixed before the first run): every semantic/hybrid query must run with real
vectors (no degradation); semantic hit@5 >= 0.70; hybrid hit@5 >= 0.70 and >= keyword hit@5.
Follow-up queries are reported, not gated (n=2).
"""

import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from hermes_mem.app import MemApp  # noqa: E402
from hermes_mem.search import SearchParams, search  # noqa: E402
from hermes_mem.sync import embed_pending, sync  # noqa: E402

work, embed_config = Path(sys.argv[1]), Path(sys.argv[2])
out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else work / "eval.json"
data = json.loads((REPO / "eval/paraphrase_uk_en.json").read_text())
schema = json.loads((REPO / "tests/fixtures/state_schema.json").read_text())

if work.exists():
    shutil.rmtree(work)
home = work / "home"
(home / "hermes-mem").mkdir(parents=True)
shutil.copy(embed_config, home / "hermes-mem/config.yaml")
conn = sqlite3.connect(home / "state.db")
for name, sql in schema.items():
    if not name.startswith("messages_fts_") and name != "sqlite_sequence":
        conn.execute(sql)
clock = 1_790_000_000.0
for sid, exchanges in data["sessions"].items():
    conn.execute("INSERT INTO sessions(id, source, started_at) VALUES (?, 'cli', ?)", (sid, clock))
    for user, assistant in exchanges:
        for role, text in (("user", user), ("assistant", assistant)):
            clock += 1
            conn.execute("INSERT INTO messages(session_id, role, content, timestamp) VALUES (?, ?, ?, ?)", (sid, role, text, clock))
conn.commit()
conn.close()

app = MemApp(home)
store = app.open_store()
started = time.monotonic()
sync(store, app.state_db, policy=app.policy)
embedding = embed_pending(store, app.embedder())
index_seconds = time.monotonic() - started
assert embedding["status"] == "complete", embedding


def rank_of(expect: str, result: dict) -> int | None:
    for i, r in enumerate(result["results"], start=1):
        if r["session_id"] == expect:
            return i
    return None


rows, degraded = [], []
for mode in ("keyword", "semantic", "hybrid"):
    for q in data["queries"]:
        res = search(store, app.state_db, app.embedder(), SearchParams(query=q["query"], scopes=None, mode=mode, limit=10))
        if mode != "keyword" and (res["search_mode"] != mode or "degraded" in res):
            degraded.append((mode, q["id"], res.get("degraded")))
        rows.append({"mode": mode, "id": q["id"], "lang": q["lang"], "rank": rank_of(q["expect"], res),
                     "top": res["results"][0]["session_id"] if res["results"] else None})
follow = []
for f in data["follow_ups"]:
    for with_context in (False, True):
        res = search(store, app.state_db, app.embedder(), SearchParams(
            query=f["query"], scopes=None, mode="hybrid", limit=10, conversation=f["conversation"] if with_context else None))
        follow.append({"id": f["id"], "with_context": with_context, "rank": rank_of(f["expect"], res), "mode": res["search_mode"]})


def metrics(mode: str) -> dict:
    ranks = [r["rank"] for r in rows if r["mode"] == mode]
    n = len(ranks)
    return {
        "n": n,
        "hit@1": round(sum(1 for r in ranks if r == 1) / n, 3),
        "hit@5": round(sum(1 for r in ranks if r and r <= 5) / n, 3),
        "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
    }


summary = {mode: metrics(mode) for mode in ("keyword", "semantic", "hybrid")}
gates = {
    "real_vectors_every_query": not degraded,
    "semantic_hit5_ge_0.70": summary["semantic"]["hit@5"] >= 0.70,
    "hybrid_hit5_ge_0.70": summary["hybrid"]["hit@5"] >= 0.70,
    "hybrid_hit5_ge_keyword": summary["hybrid"]["hit@5"] >= summary["keyword"]["hit@5"],
}
report = {
    "model": app.embedder().model_id,
    "dataset": "eval/paraphrase_uk_en.json",
    "provenance": data["provenance"],
    "episodes": len(data["sessions"]),
    "index_seconds": round(index_seconds, 2),
    "summary": summary,
    "follow_ups": follow,
    "gates": gates,
    "degraded": degraded,
    "per_query": rows,
}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps({k: report[k] for k in ("model", "summary", "follow_ups", "gates")}, ensure_ascii=False, indent=1))
sys.exit(0 if all(gates.values()) else 1)
