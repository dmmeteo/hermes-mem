# AGENTS.md

Read this before changing hermes-mem. Read [README.md](README.md) for usage and limits, [ROADMAP.md](ROADMAP.md) for scope, and [docs/adr/index.md](docs/adr/index.md) before you touch storage, identity or the embedding default.

## Map

| Path | Owns |
|---|---|
| `__init__.py`, `plugin.yaml` | Hermes entry: `register(ctx)` registers the tools, the `hermes-mem` CLI and the post-turn hooks. Keep `kind: standalone`. |
| `hermes_mem/config.py` | Strict config schema, defaults, model profiles (default `bge-m3-int8`), shared model cache, search instructions |
| `hermes_mem/runtime.py`, `engine/` | Engine runtime diagnosis, `setup`, resident worker client; the worker and its lock ([ADR 0004](docs/adr/0004-external-engine-and-embedding-spaces.md)) |
| `hermes_mem/index.py` | Operator indexing: read-only preview, bounded resumable runs over `sync` |
| `hermes_mem/host.py` | Read-only `state.db` access, host row filters, eligibility and authoritative scope |
| `hermes_mem/select.py` | Pure rules: messages → episodes, and `ProcessingPolicy` (episodes → stored chunks, `chunk_chars`) |
| `hermes_mem/store.py` | Plugin SQLite schema and upgrades (episodes, vectors per embedding space, curated records and events), coverage and embed failure back-off |
| `hermes_mem/sync.py` | Idempotent reconcile and backfill, embedding, background worker |
| `hermes_mem/search.py` | Hybrid retrieval, RRF, hydration and source status |
| `hermes_mem/curated.py` | Curated record shape, authority rules and the SQLite curated store ([ADR 0005](docs/adr/0005-curated-records-in-sqlite.md)) |
| `hermes_mem/curated_md.py` | Explicit Markdown import (preview, all or nothing) and export of curated records |
| `hermes_mem/app.py`, `tools.py`, `cli.py` | Wiring, caller scope and search requests; agent tools; operator CLI. `tools` never imports `cli` |
| `tests/` | Behaviour tests on the exported live host schema (`tests/fixtures/state_schema.json`) |
| `scripts/verify.sh` | All acceptance gates |
| `scripts/host_integration.py` | The real host run |
| `scripts/eval_paraphrase.py` | The real-embedding eval |
| `scripts/compare_models.py` | Real models (default: BGE-M3 INT8 and MiniLM) on one frozen corpus |
| `scripts/install_acceptance.py` | Installed-copy acceptance and the candidate-host compatibility command |
| `docs/soul-example.md` | Install-time SOUL instruction example; the plugin never edits `SOUL.md` |

## Invariants

These are accepted constraints; changing one needs owner agreement.

- **state.db access:**
  - Never write to `state.db`, add columns to it, or patch or monkeypatch Hermes core.
  - Open it only via `host.open_state` (`mode=ro`, through the host's tracked connector when available). Never byte-probe the file.
  - Do not replace or shadow `session_search`, and do not touch `USER.md`, `MEMORY.md` or `SOUL.md`.
- **Selection:**
  - Index only user requests and visible assistant answers, using the host predicate `(active = 1 OR compacted = 1)`.
  - Exclude tool traffic, reasoning, compaction summaries, routine sessions and observed chatter.
  - Never drop a message only because it is short.
- **Durability:**
  - Excerpts are durable and never deleted automatically.
  - Changes create versions (`superseded`), rewinds `retract`, and vanished sources are `source_missing`.
  - Search returns only `current` episodes and reports source state honestly.
- **Processing:**
  - `processing.retain`, `compression` and `expiry` default to `none` and are the only implemented values ([ADR 0003](docs/adr/0003-processing-policy.md)).
  - Any other value must fail config loading. Never accept a policy that does not run.
  - Every sync caller passes `MemApp.policy`.
- **Identity:**
  - Scope comes from the authoritative session row only.
  - Never infer a speaker from a display name or from message text. Unknown or unprovable identity means the session's own scope.
  - Profiles stay isolated, and global sharing stays disabled.
- **Retrieval:**
  - Vector failure must surface as `degraded` keyword mode.
  - Hybrid keyword stays strict ([ADR 0001](docs/adr/0001-durable-excerpt-store.md)).
  - No LLM reranker, no auto-recall.
- **Curated records:**
  - `mem.db` is the only authoritative store. Markdown is read only by an explicit `curated import`, and import never modifies or deletes files.
  - Content is immutable and rows are never deleted (enforced by triggers); every change appends a `curated_events` row in the same transaction.
  - Agent writes are `proposed`; only the operator accepts, and the agent cannot revise or retract an accepted record.
  - `declared` needs user-authored evidence.
  - Revisions supersede and never overwrite.
  - Retrieved text is data, not instructions, and is bounded on output.
- **Engine boundary:**
  - Nothing heavy (numpy, ONNX, fastembed) is imported in the Hermes process, and nothing is pip-installed into Hermes.
  - Only `setup` provisions; discovery, `register` and `status` never install, download or start anything.
  - A query only meets vectors of its own embedding space ([ADR 0004](docs/adr/0004-external-engine-and-embedding-spaces.md)).
  - The default profile is pinned; never substitute another model or precision for it silently ([ADR 0006](docs/adr/0006-bge-m3-default-and-shared-model-cache.md)).
  - Model weights may be shared across profiles; episodes, curated records and the engine runtime stay per profile.
  - Expose a config knob only if it is wired into behaviour.
- **Work off the response path:** embedding and sync never block it, and every pass is idempotent and resumable.

## Working rules

- Run `bash scripts/verify.sh` before committing. It must exit 0. Do not weaken a gate or replace real embeddings with test doubles in the eval.
- The host integration needs a Hermes source tree (`HERMES_HOST_SOURCE`) and Python 3.11–3.13. The plugin code itself must stay compatible with 3.11+.
- Test the evaluation or live install only in a temporary `HERMES_HOME`, never in a live profile.
- Check a ROADMAP item only with executed acceptance evidence.
