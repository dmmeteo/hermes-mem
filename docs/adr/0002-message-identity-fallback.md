# 0002 Message identity without message_uid

Status: accepted, 2026-10-10.

## Context

The installed Hermes schema (host revision d806379, schema version 30) has no `message_uid` column. `messages.id` is AUTOINCREMENT and never reused, but the same logical message can still appear under several ids:
- In-place compaction clones the concurrent tail with fresh ids and byte-exact columns.
- Rotation copies handoff rows into a child session.
- Profile moves and imports re-insert rows.

`set_user_message_content` can also rewrite a user row's content in place.

## Decision

- If a `message_uid` column exists, use `uid:<value>` as the message reference.
- Otherwise use `<role>@<timestamp>`. Within one session, rows sharing that key collapse to one logical message: the lowest id is kept for references, and the content of the active row wins.
- Episode keys are `<scope>/<message ref>`, not session ids. A compaction clone or rotated copy that keeps its original timestamp therefore maps to the existing episode, even after the parent session is pruned. When the original session disappears, the next pass re-points the episode to the surviving copy. Column-cloned tails always keep their timestamp; handoff messages keep it only if the carried message dict has one.
- Content is compared by hash. A change creates a new episode version and supersedes the old one; a changed key never silently mutates an excerpt.
- `sync` reports the identity mode actually used (`role_timestamp` on this install).

## Limits

Two distinct messages with the same role and the exact same float timestamp in one session would merge. Rotated handoff copies or imports that get new timestamps produce duplicate episodes. Copy dedupe and re-pointing are covered by a unit test on the live schema, not by a host-level rotation test. When Hermes ships `message_uid`, it is used automatically, but existing episodes keep their old keys, so a re-key migration is a roadmap item.
