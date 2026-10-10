# hermes-mem

A standalone [Hermes Agent](https://github.com/NousResearch/hermes-agent) plugin that remembers meaningful past conversation **episodes**: a user request together with the assistant's visible answer. It finds them again with hybrid keyword + multilingual semantic search, and keeps curated records (peer cards, intents, observations, decisions) with evidence, status and an immutable version history.

**Status: experimental (0.2.0).** It stores selected verbatim episodes plus separately curated records. It does not use an LLM to pick or rewrite facts. Its quality evidence comes from small synthetic Ukrainian/English sets, not real history (see [Limits](#limits)).

It runs beside Hermes and does not replace anything:
- `state.db` is only read, never written.
- The built-in `session_search` tool keeps working.
- `USER.md`, `MEMORY.md` and `SOUL.md` are never touched.
- Recall is explicit (tools and CLI). Nothing is injected into prompts automatically.

## Quickstart

Needs Hermes Agent with Python 3.11+ and network access for the first setup. Run these in the profile you want to give memory (`hermes -p <profile> ...` targets another one):

```bash
hermes plugins install dmmeteo/hermes-mem --enable --no-deps
hermes hermes-mem setup                 # once: builds the plugin's own embedding runtime
hermes hermes-mem index --dry-run       # read-only preview: what would be indexed and why
hermes hermes-mem index                 # select, store and embed existing conversations
hermes hermes-mem status                # coverage, pending and failures
```

`--no-deps` is right here: the plugin itself needs only `pyyaml`, which Hermes already ships, and `setup` installs the embedding engine into a separate locked venv. The first embedding downloads the pinned BGE-M3 INT8 model (about 590 MB) into `~/.cache/hermes-mem/models`, shared by your profiles.

Then start a new Hermes session (restart the gateway if one is running) so the agent gets the tools. New turns are indexed in the background after each reply. For a reproducible install, pin a full commit with `--ref <40-char sha>`. For large histories, index in bounded, resumable batches; see [docs/operations.md](docs/operations.md#index-history).

## How retrieval works

The agent looks things up only when it decides to call a tool:

**Agent tools** (toolset `hermes_mem`):

| Tool | What it does |
|---|---|
| `mem_curated` | Reads structured curated records (peer cards, intents, observations, decisions) in the caller's scope: list or get. |
| `mem_search` | Finds past discussions. `query`, `mode` (hybrid, keyword or semantic), `limit` ≤ 20, `in_session`. `conversation` takes recent turns for follow-ups such as "why did we choose that?". `variants` takes up to `search.max_variants` reformulations (for example Ukrainian/English equivalents and transliterations, as the default guidance asks). `conversation` and `variants` feed only the semantic arm, one ranked list per text, fused with RRF and deduplicated by episode; the keyword arm keeps the literal query, and none of them can widen the caller's scope. `query_provenance` reports the model space, strategy and the query texts used. |
| `mem_curate` | Proposes, revises or retracts records. Agent proposals stay `proposed`. `declared` requires evidence from a user-authored episode. A peer card may only describe the caller's own authoritative identity. |

Retrieval is explicit; nothing is recalled automatically. The stock `session_search` tool keeps reading the original messages of a session, so the agent can combine the three: `mem_curated` for what is known, `mem_search` for where it was discussed, and `session_search` for the original transcript.

To make the agent use them deliberately, add a short instruction such as [docs/soul-example.md](docs/soul-example.md) to the profile's `SOUL.md` yourself. The plugin never edits it.

## What it does

- **Selects** user requests and final visible assistant answers from `state.db` and groups them into episodes. Long answers are split into chunks, each carrying the question.
  - Skipped: tool calls, tool output, reasoning, synthetic compaction summaries, hidden or internal rows, passively observed group chatter, pasted logs, bare acknowledgements, and `cron`/`subagent`/`kanban`/`tool`/delegated/hidden sessions.
  - Short corrections and decisions are kept.
- **Stores** excerpts, source references, content hashes, vectors (with model id and dimension) and curated records in `$HERMES_HOME/hermes-mem/mem.db`. Excerpts are durable:
  - If the source session is deleted or pruned, search reports `source.state = missing` with context `unavailable`.
  - Edited sources create a new version and the old one becomes `superseded`.
  - Rewound exchanges become `retracted`.
  - Search returns only current versions.
  - If the source changed after indexing, search reports `changed` until the next sync.
- **Compacted history** stays searchable. It uses the same `(active = 1 OR compacted = 1)` predicate as the host, and compaction clones collapse into one episode.
- **Searches** with three components:
  - FTS5 BM25 (`unicode61`, then `trigram`).
  - Exact cosine similarity over local multilingual vectors.
  - Reciprocal rank fusion of the two.
  
  Each result carries `matched_by`, ranks and similarity, plus session/message references and bounded neighbouring context. Without usable vectors, search answers in keyword mode and says so in `degraded`; it never claims a vector match.
- **Scopes** results by authoritative identity from the session row. Display names and message text are never treated as identity.

  | Caller | Sees |
  |---|---|
  | Local owner sessions (`cli`, `tui`, ...) | The whole profile |
  | A DM peer | Only that peer's episodes |
  | A per-user group session (the session key provably ends with the participant id) | Only that chat and participant |
  | A shared group session, or one whose per-user key can't be proven | Only that session |
  | An unknown caller | Nothing beyond its own session scope |

  Each Hermes profile has its own `HERMES_HOME`, so data is never shared across profiles.
- **Syncs** after each turn (`post_llm_call` / `on_session_end`) on a background thread, so it never blocks the response.
  - Bounded `sync` runs resume from a cursor, and each full pass replays everything.
  - Each background run also advances the backfill by one bounded step (25 sessions). Turns whose hook never fired, or whose sync lock stayed busy, are therefore repaired on a later turn. `hermes hermes-mem sync --all` does the same immediately.
  - Embedding failures back off per chunk.
  - Changing the embedding profile re-embeds lazily into new rows keyed by embedding space; old spaces are kept.

## Documentation

- [docs/configuration.md](docs/configuration.md): the optional `config.yaml`, model profiles, the shared model cache, embedding spaces and search instructions.
- [docs/operations.md](docs/operations.md): the engine runtime, indexing history in batches, the operator CLI, upgrade and rollback.
- [docs/development.md](docs/development.md): the acceptance gates and how to check a candidate Hermes version.
- [docs/adr/index.md](docs/adr/index.md): design decisions. [ROADMAP.md](ROADMAP.md): what is done and what is not.

## Limits

- **Eval scope:** the eval and the model comparison use small hand-written synthetic sets (15 + 8 episodes, 22 questions, 4 follow-ups). They show the pipeline works and let candidates be compared; they do not show quality on real history, and they make no comparison with Hindsight or `session_search`.
- **Query variants** come from the calling agent, as the guidance asks. The plugin makes no model call to write them, has no translation dictionary, and runs a variant only through the semantic arm.
- **Engine:** fastembed models only; OpenAI-compatible endpoints run in-process over HTTP. The first `setup` and each model's first use need network access. BGE-M3 needs more memory in the engine process and about 2.7× more vector-scan time per chunk than MiniLM (1024 against 384 dimensions). Its ONNX INT8 export is third-party, and its fidelity to the upstream weights was not measured here.
- **Compatibility:** tested against the Hermes source at the revision in the roadmap. Other versions need `scripts/install_acceptance.py` run against them first.
- **Message identity:** no `message_uid` exists in the supported host schema, so identity falls back to `(role, timestamp)` ([ADR 0002](docs/adr/0002-message-identity-fallback.md)).
- **Shared group chats:** these have no per-message speaker identity, so speakers stay unknown and results stay inside that one session. A reset shared chat therefore cannot recall its earlier sessions.
- **Search cost:** vector search is an exact pure-Python scan of current, in-scope vectors of the active space, repeated for each semantic variant, so its cost grows with the store. There is no ANN index. `compare_models.py` reports warm latency, but only on its 23-chunk store; no larger store has been measured.
- **Not implemented:** no automatic inference of curated statements (background LLM reflection is a roadmap item, off and unbuilt), no retain/extraction, compression or expiry policy (only the `processing` boundary exists), no LLM reranker, no auto-recall, no cross-profile sharing.
- **Hindsight:** this plugin runs beside it. Turning Hindsight off and importing its old data are future rollout steps (see the [roadmap](ROADMAP.md)); neither has been done.

## License

MIT, see [LICENSE](LICENSE). Nothing third-party is vendored in this repository. `setup` installs the engine's packages from PyPI, pinned in `engine/requirements.lock`, and the first embedding downloads model weights from Hugging Face. They keep their own licenses: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (MIT; the default loads the third-party ONNX export [Xenova/bge-m3](https://huggingface.co/Xenova/bge-m3)), [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) (Apache-2.0) and fastembed (Apache-2.0).

## For agents

Coding agents start at [AGENTS.md](AGENTS.md).
