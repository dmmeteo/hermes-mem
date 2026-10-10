# 0004 External embedding engine and embedding spaces

Status: accepted, 2026-10-10.

## Context

The MVP imported numpy and fastembed into the Hermes interpreter and needed a manual `pip install` there. A host update or a recreated Hermes venv broke it, and heavy compiled packages shared Hermes' process and dependency set. The host at d806379 has no external-runtime facility for plugins: its loader only warns about `python_dependencies`. Separately, a stored vector is only comparable to a query embedded by the same model *and* the same preprocessing. Two models with equal dimensions would silently mix.

## Decision

- **Adapter.** The plugin Hermes imports is stdlib plus `pyyaml` (host-provided). Vectors are `array('f')`, and cosine scoring is pure Python.
- **Engine.** `engine/hermes_mem_engine.py` runs from `<HERMES_HOME>/hermes-mem/runtime/engine-<lock hash>`, a venv the plugin owns.
  - The venv is built by the explicit `hermes hermes-mem setup` from `engine/requirements.lock` (hash-pinned) with the host's *base* interpreter, so it does not depend on the Hermes venv.
  - It is built in staging, probed, then swapped in. A failure keeps the previous runtime.
  - Discovery and `status` only diagnose.
- **Process.** There is one resident worker per Hermes process. It is started on first use and speaks JSON lines over stdin/stdout, with a timeout per request and one restart after a crash. It exits on idle or EOF. There is no daemon or socket, `engine.command` is an argv list, and secrets never go in argv.
- **Spaces.** `EmbeddingSpace` (backend, model, revision, query/passage prefixes, HF source/file/pooling, dim, base_url) has a `space_id` that keys every stored vector.
  - Search compares a query only with vectors of its own `space_id`.
  - The MVP key `fastembed:<model>` is kept when nothing else differs, so existing stores need no migration (schema version 1 is unchanged).
- **Pinning.** fastembed 0.9.0 ignores a `revision` and fetches the repository head. A pinned profile is therefore fetched by the engine with `huggingface_hub.snapshot_download(revision=…)` and loaded through `specific_model_path`. The engine reports the snapshot it loaded and refuses a mismatch with a pinned commit.
- **Chunking** (`sync.chunk_chars`) is part of `ProcessingPolicy`, so every sync caller uses it. A change creates new episode versions rather than editing chunks in place.

## Consequences

- Hermes venv replacement and plugin reinstall keep the runtime, the store and curated records. A lock change creates a new `engine-<hash>` directory, which needs `setup` again; old runtimes are listed by `status`, never deleted automatically.
- Each Hermes process that embeds holds one model in a child process. Memory is the model's, not Hermes'.
- Pure-Python scoring costs O(chunks × dim) per semantic arm. Capacity beyond thousands of chunks needs measurement, then sqlite-vec or numpy in the engine.
- OpenAI-compatible endpoints stay in-process (HTTP only, no heavy imports).
