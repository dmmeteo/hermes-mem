# 0001 Durable excerpt store beside state.db

Status: accepted, 2026-10-10.

## Context

hermes-mem must recall meaningful past exchanges in Ukrainian and English, keep them after Hermes prunes or deletes sessions, and never write to `state.db`. Hermes already has FTS5 tables over `messages` (`messages_fts`, `messages_fts_trigram`; the CJK table needs a loadable tokenizer).

## Options compared

**A. Pointer index.** The plugin DB holds only `(session_id, message_id)` references and vectors. Keyword search queries the host's `messages_fts` read-only, and text is read from `state.db` at query time.
- Pro: no duplicated text, and keyword semantics are exactly the host's.
- Con: excerpts disappear when Hermes auto-prunes (on by default, 90 days) or deletes a session, which breaks durability.
- Con: host FTS also indexes tool rows, so every query needs host-side filtering.
- Con: an in-place content update silently changes what an old vector stood for.

**B. Durable excerpt store (chosen).** The plugin DB copies the selected episode text, keeps its own FTS5 tables with the host's tokenizer families (`unicode61` and `trigram`), and stores float32 vectors keyed by model and text hash. `state.db` is read only for ingestion, reconciliation and context hydration.
- Pro: excerpts survive source deletion and are reported as `source_missing`.
- Pro: changes become explicit versions (`superseded`), and rewinds become `retracted`.
- Pro: the index can be rebuilt from `state.db` while it still exists.
- Con: selected text is duplicated, and keyword ranking is BM25 over episodes rather than over raw messages.

## Decision

B. Durability and honest staleness are hard requirements; duplicating only the selected user/assistant text is cheap. Vectors are a plain BLOB table searched exactly with numpy (no `sqlite-vec` and no `vec0` virtual table), which follows the PR #44093 reasoning about extension loading.

In hybrid mode the keyword arm is strict: an AND match, then a trigram AND match. OR-relaxed matching runs only when keyword search is the sole arm. In the first eval run, relaxed matches on common words flooded RRF and cut hybrid hit@1 from 0.93 to 0.14.
