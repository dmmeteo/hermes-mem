CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS sources (
    session_id TEXT PRIMARY KEY,
    source TEXT,
    scope TEXT NOT NULL,
    title TEXT,
    started_at REAL,
    state TEXT NOT NULL CHECK (state IN ('present', 'missing')),
    eligible INTEGER NOT NULL,
    skip_reason TEXT,
    fingerprint TEXT,
    checked_at REAL NOT NULL
);

-- embeddings.model_id holds the EmbeddingSpace.space_id (backend, model, revision, preprocessing).
-- One row per version of an episode. episode_key is stable across versions.
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY,
    episode_key TEXT NOT NULL,
    version INTEGER NOT NULL,
    session_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    identity_mode TEXT NOT NULL,
    user_ref TEXT,
    assistant_refs TEXT NOT NULL,
    first_message_id INTEGER NOT NULL,
    last_message_id INTEGER NOT NULL,
    user_text TEXT NOT NULL,
    assistant_text TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at REAL NOT NULL,
    compacted INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('current', 'superseded', 'retracted')),
    source_state TEXT NOT NULL CHECK (source_state IN ('present', 'missing')),
    superseded_by INTEGER REFERENCES episodes(id),
    status_reason TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE (episode_key, version)
);
CREATE UNIQUE INDEX IF NOT EXISTS episodes_one_current
    ON episodes(episode_key) WHERE status = 'current';
CREATE INDEX IF NOT EXISTS episodes_session ON episodes(session_id, status);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    episode_id INTEGER NOT NULL REFERENCES episodes(id),
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    UNIQUE (episode_id, chunk_index)
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content='chunks', content_rowid='id');
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts_trigram USING fts5(
    text, content='chunks', content_rowid='id', tokenize='trigram'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
    INSERT INTO chunks_fts_trigram(rowid, text) VALUES (new.id, new.text);
END;

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id INTEGER NOT NULL REFERENCES chunks(id),
    model_id TEXT NOT NULL,
    dim INTEGER NOT NULL,
    text_hash TEXT NOT NULL,
    vector BLOB NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (chunk_id, model_id)
);

CREATE TABLE IF NOT EXISTS embed_failures (
    chunk_id INTEGER NOT NULL,
    model_id TEXT NOT NULL,
    attempts INTEGER NOT NULL,
    last_error TEXT NOT NULL,
    retry_after REAL NOT NULL,
    PRIMARY KEY (chunk_id, model_id)
);
