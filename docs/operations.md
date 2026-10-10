# Operations

Install, index, upgrade and roll back hermes-mem in one Hermes profile. The [README](../README.md#quickstart) has the short path.

## Engine runtime

The plugin that Hermes imports is pure Python plus `pyyaml`, which the host already ships. numpy, ONNX Runtime and fastembed run only in a separate **engine** process, from a locked venv the plugin owns. Nothing is pip-installed into Hermes, and recreating the Hermes venv leaves the engine, the store and curated records alone ([ADR 0004](adr/0004-external-engine-and-embedding-spaces.md)).

- `setup` builds the venv from `engine/requirements.lock` (hash-pinned, Python 3.11+; every pin has a Linux aarch64 or pure-Python wheel). It uses the host's *base* interpreter, with `uv` when present and `venv` + `pip` otherwise. Use `--python PATH` to pick another interpreter. The first run needs network access.
- `setup` is idempotent and prints `unchanged` when the runtime matches the lock. `setup --force` rebuilds in a staging directory and swaps the result in only after a probe passes, so a failed build keeps the old runtime.
- Discovery, `register` and `status` never install, download or start anything. Until `setup` has run, `status` shows `engine.state: missing` and search answers in keyword mode with a `degraded` note.
- After a host update, reinstalling the plugin (`hermes plugins install dmmeteo/hermes-mem --force`) keeps the runtime, because its path depends only on the lock.
- The engine starts on the first embedding call, once per Hermes process. It speaks JSON lines over a pipe and exits after `engine.idle_seconds` idle or when Hermes exits. It is not a daemon or a network service.
- The default model's first use downloads the pinned BGE-M3 INT8 snapshot (about 590 MB) into the shared model cache, `~/.cache/hermes-mem/models` (or `$XDG_CACHE_HOME/hermes-mem/models`). Other profiles of the same user reuse it. Raise `engine.timeout` if that first request is slow. Runtime, store and curated records stay per profile.

## Index history

The operator flow for existing conversations. Every step is safe to repeat.

```bash
hermes hermes-mem setup                              # once per profile; `unchanged` when already provisioned
hermes hermes-mem index --dry-run --excerpts 3       # read-only: selected/excluded counts, reasons, samples
hermes hermes-mem index --batch-sessions 50 --max-batches 4   # bounded; re-run the same command to resume
hermes hermes-mem index                              # to completion: select, store, embed
hermes hermes-mem status                             # coverage, pending and failures per profile
```

- `--dry-run` opens `state.db` read-only and writes nothing: no store, config, runtime or model download. Excerpts appear only with `--excerpts N` (at most 20), and each is cut to 160 characters.
- `index` selects, stores episodes and embeds in one pass, reusing the background sync's rules. `--session`, `--source`, `--since` and `--until` only narrow which sessions are visited; they cannot admit tool traffic, reasoning or ineligible sessions, or change scopes. `--profile P` embeds the same episodes into another space, and `--no-embed` only stores episodes.
- Progress goes to stderr as JSON lines. `--max-batches N` stops early. Ctrl-C exits 130. In both cases the output's `resume` field says to re-run the same command; unchanged sessions and existing vectors are skipped, so repeats are no-ops.
- `status.profiles.<name>` shows `current_vectors`, `pending` and `failed` (with `last_error`) for every configured profile. Failed chunks back off and are retried by later runs. While coverage is partial, search adds a `warning` that unembedded episodes can only match by keyword.
- `status.curated.legacy_markdown` counts curated Markdown files from an earlier release that are not imported yet ([Upgrade and rollback](#upgrade-and-rollback)).

## Operator CLI

Run as `hermes hermes-mem ...` in the profile, or as `python -m hermes_mem --home PATH ...`:

```text
setup [--python PATH] [--force]
index [--dry-run [--excerpts N]] [--session ID] [--source S] [--since DATE] [--until DATE]
      [--batch-sessions N] [--max-batches N] [--profile P] [--no-embed]
sync [--all] [--max-sessions N] [--session ID] [--embed]
embed [--profile P] [--max-batches N]
search QUERY [--mode M] [--scope S] [--conversation TEXT] [--variant TEXT ...] [--profile P]
status
history EPISODE_ID
curated list|show|add|revise|review ID --accept|--retract|history ID
curated import [--from DIR] [--dry-run]
curated export --to DIR
```

Curated records live in `mem.db` ([ADR 0005](adr/0005-curated-records-in-sqlite.md)). A record's content never changes: revising inserts a successor and marks the old record `superseded`, and accepting or retracting changes only the status. Accepting is an operator action. Every change is recorded in an append-only history, which `curated history ID` shows with the whole revision chain. Markdown is only an explicit format: `curated import` reads the previous on-disk layout, and `curated export` writes it.

## Upgrade and rollback

From a release with Markdown curated records and schema 1:

1. Back up the store: `sqlite3 $HERMES_HOME/hermes-mem/mem.db ".backup mem.db.bak"`.
2. Reinstall the plugin (`hermes plugins install dmmeteo/hermes-mem --force --enable --no-deps`) and run `hermes hermes-mem setup` (the lock is unchanged, so it prints `unchanged`). `status` opens the store, which upgrades it to schema 2 in place, and shows `curated.legacy_markdown`.
3. `hermes hermes-mem curated import --dry-run` lists new records (with the status, author, basis and scope each claims, so you can review its authority), unchanged, conflicting and invalid files. Evidence must exist and `declared` needs a user-authored episode. Any conflict or invalid file blocks the import, and nothing is published. Fix the file, then run `curated import`. The import never changes a stored record. The Markdown directory is left untouched, and a repeat import is a no-op.
4. The new default model re-embeds lazily. Run `hermes hermes-mem index` to embed everything now, then check `status.profiles.bge-m3-int8.pending == 0`. The MiniLM vectors are kept, so `embedding.active: minilm` switches back without re-embedding.

Rollback: run `curated export --to DIR` with this release first, so curated changes made since the upgrade survive in the previous format. Then restore `mem.db.bak`, replace `curated/` with the export, and reinstall the previous release. That release refuses a schema 2 store rather than misreading it.
