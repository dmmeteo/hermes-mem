# 0003 Processing policy: raw by default

Status: accepted, 2026-10-10.

## Context

Later versions may want to process selected episodes in the plugin store. That could mean model extraction or consolidation ("retain"), lossy compression, or expiry. None of it exists yet. A setting that is accepted but never acted on would claim work that did not happen. Disabling ingestion is not a valid reading of "no retain" either.

## Decision

- `hermes-mem/config.yaml` takes a `processing` section with `retain`, `compression` and `expiry`. Each defaults to `none`, and `none` is the only implemented value. YAML `off`, `no`, `false` and an empty value mean `none`, and an empty `processing:` section means all defaults.
- `ProcessingPolicy` (`hermes_mem/select.py`) is a frozen value that cannot hold anything else. `MemApp` builds it from config. An unknown key, a non-mapping or an unimplemented value raises `ConfigError`, which names the config file, the key and the fix. Inside Hermes, `register()` fails, the host shows that message as the plugin's load error, and no tools, hooks, CLI or sync are registered. `python -m hermes_mem` prints the message and exits 2.
- Every sync path (CLI, background worker, eval) must pass the policy, because `sync(..., policy=...)` has no default. `reconcile_session` stores only `policy.apply(build_episodes(...))`. The raw policy turns each selected episode into verbatim chunks of the original user request and visible answer.
- Sync reports and `status` show the policy that actually ran (`excerpts: raw`, `retain/compression/expiry: none`).

## How a future policy connects

- **compression.** Add the value to `ProcessingPolicy` and make `apply` derive different chunks. Episodes keep the original `user_text`/`assistant_text`, the source refs, scope and `content_hash`, so originals are never overwritten. Also add the policy to the source fingerprint, so that sessions already synced are reprocessed, and have `describe()` report the excerpt form it actually produced instead of the fixed `raw`.
- **retain (extraction/consolidation).** Run after `reconcile_session` commits, off the response path. Write only `proposed` curated records whose evidence is the episode (the existing curated invariants apply), and never auto-accept. Each model call needs its own configured provider; nothing here adds one.
- **expiry.** Excerpt durability is an AGENTS.md invariant. Any expiry needs owner agreement first, and should mark records rather than delete them.

## Consequences

Default behaviour is unchanged, and every existing test now runs through the policy boundary. Hindsight stays enabled. Disabling it is a step in the production-rollout checklist, and importing its data is a separate roadmap item. Neither is part of this decision.
