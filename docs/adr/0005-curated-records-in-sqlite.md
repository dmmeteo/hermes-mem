# 0005 Curated records in mem.db

Status: accepted, 2026-10-10.

## Context

Curated records (peer cards, intents, observations, decisions) were Markdown files with YAML front matter under `$HERMES_HOME/hermes-mem/curated/<kind>/<id>.md`, guarded by a file lock. Episodes, vectors and coverage already lived in `mem.db`. That split had real costs:
- A revision wrote two files. A crash between them could leave a successor without a superseded predecessor.
- History was whatever the files said. A hand edit could rewrite an accepted statement in place, and nothing recorded who changed what.
- Every read parsed every file, and evidence checks crossed from files into `mem.db`.
- Backup, rollback and status had two stores to reason about.

## Options compared

**A. Keep Markdown authoritative.** Human-editable, but no transactions, no immutable history, and in-place edits bypass the operator/agent authority rules.

**B. Both stores, kept in sync.** Two sources of truth that can disagree, and a sync job to explain to the operator.

**C. SQLite canonical, Markdown as explicit import/export (chosen).** One store, one transaction per change, and the old format stays usable as a portable file format.

## Decision

- **One record shape.** `curated_records` holds every kind, distinguished by `kind`. Fields are those of the Markdown records: `scope`, `subject`, `statement`, `status`, `basis`, `confidence`, `evidence` (JSON list), `author`, `supersedes`, `superseded_by`, `note` and timestamps. CHECK constraints admit only the known kinds, statuses, bases, confidences and authors.
- **Immutable content.** Triggers reject any update to the content columns (`id`, `kind`, `scope`, `subject`, `statement`, `basis`, `confidence`, `evidence`, `author`, `supersedes`, `created_at`). They also reject:
  - any delete;
  - a status change out of `superseded` or `retracted`, or from `accepted` back to `proposed`;
  - a change to `superseded_by` once it is set.

  A unique index allows at most one successor per record. Statements are stored trimmed, the same way the Markdown format writes them.
- **Provenance.** `curated_events` is append-only (triggers reject update and delete). Each `propose`, `add`, `revise`, `supersede`, `accept`, `retract` and `import` appends a row in the same `BEGIN IMMEDIATE` transaction as the change it records. `hermes hermes-mem curated history ID` shows the whole revision chain and its events.
- **Authority is unchanged.** Agent writes are `proposed`. Only the operator accepts. The agent cannot revise or retract an accepted record. `declared` needs user-authored evidence. Scope comes from the authoritative session row.
- **Schema 2.** The store's schema version goes from 1 to 2. The upgrade is additive (`CREATE ... IF NOT EXISTS`, then the version mark), so it is idempotent and needs no data rewrite. A store newer than the code is refused before any DDL runs.
- **Markdown import.** `curated import [--from DIR] [--dry-run]` reads the previous layout (default: the old `curated/` directory). Planning classifies every file:
  - **new**: the dry-run shows its kind, scope, subject, status, author, basis and evidence, so the operator reviews the authority it claims before publishing it;
  - **unchanged**: the same content is already stored, as written or moved on by the store's own operations (accepted, retracted or revised later);
  - **conflict**: the same id with different content, or a status or successor the stored record never had;
  - **invalid**: a parse, type or validation error; evidence that fails the store's own rule (refs must exist, and `declared` needs a user-authored episode); a supersession link that is dangling, not mirrored by its target, or circular.

  Links are checked against the stored row whenever one exists, so a file can add records but never change stored ones. Any conflict or invalid file blocks the import, and nothing is published. The import plans again inside one `BEGIN IMMEDIATE` transaction and inserts every new record there, each with an `import` event that names the file and its SHA-256. Any failure rolls the whole import back. Files are never modified or deleted, and a repeat import is a no-op.
- **Repairing an old crash.** The Markdown store could leave a successor whose predecessor was never marked superseded. Import reports that pair as invalid. Before importing, fix the predecessor's file (`status: superseded`, `superseded_by: <successor>`). The files are still the operator's to edit at that point.
- **Markdown export.** `curated export --to DIR` writes every record, in all statuses, in the previous format into a new or empty directory. It stages the files in a fresh temporary directory beside the target and moves them into place at the end. The previous release reads the result.
- **No implicit Markdown.** Nothing reads the Markdown directory except `import` and `status`. `status.curated.legacy_markdown` counts files that are not imported yet, plus conflicts and invalid files.

## Rollout and rollback

- **Upgrade.** Back up `mem.db` first (`sqlite3 mem.db ".backup mem.db.bak"`). Install the new release, then run `hermes hermes-mem status`: opening the store upgrades it to schema 2. Run `curated import --dry-run`, fix any conflict or invalid file it reports, then run `curated import`. The Markdown directory stays where it was.
- **Rollback.** The previous release refuses a schema 2 store. First run `curated export --to DIR` with the new release, so that changes made since the upgrade survive in the previous format. Then restore the `mem.db` backup, put the exported files in `curated/`, and reinstall the previous release. Episodes indexed after the backup are rebuilt by the next sync while their sessions still exist in `state.db`. Rebuilt episodes get new ids, so `episode:N` evidence in records created after the backup may no longer resolve; the previous release does not re-check evidence on read.

## Consequences

- Hand editing is gone. Changes go through `curated revise` / `review`, or through an export, an edit and an import into a fresh store.
- One backup of `mem.db` covers episodes, vectors and curated records.
- Curated reads are indexed queries instead of a directory scan.
