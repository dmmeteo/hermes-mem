import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path

from array import array
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hermes_mem.app import MemApp  # noqa: E402
from hermes_mem.embed import EmbeddingSpace, EmbeddingUnavailable, normalise  # noqa: E402

SCHEMA = json.loads((ROOT / "tests/fixtures/state_schema.json").read_text())


class StateDB:
    """A state.db with the exact exported live schema, written the way Hermes stores rows."""

    def __init__(self, path: Path):
        self.path = path
        conn = sqlite3.connect(path)
        for name, sql in SCHEMA.items():
            if name.startswith("messages_fts_") or name == "sqlite_sequence":
                continue
            conn.execute(sql)
        conn.execute("INSERT INTO state_meta VALUES ('store_instance_id', 'test-instance')")
        conn.commit()
        self.conn = conn
        self.clock = 1_700_000_000.0

    def session(self, sid, source="cli", **cols):
        cols = {"started_at": self.clock, **cols}
        names = ["id", "source", *cols]
        self.conn.execute(
            f"INSERT INTO sessions({','.join(names)}) VALUES ({','.join('?' * len(names))})",
            [sid, source, *cols.values()],
        )
        self.conn.commit()
        return sid

    def msg(self, sid, role, content, **cols):
        self.clock += 1
        cols = {"timestamp": self.clock, **cols}
        names = ["session_id", "role", "content", *cols]
        cur = self.conn.execute(
            f"INSERT INTO messages({','.join(names)}) VALUES ({','.join('?' * len(names))})",
            [sid, role, content, *cols.values()],
        )
        self.conn.commit()
        return cur.lastrowid

    def exchange(self, sid, user, assistant):
        return self.msg(sid, "user", user), self.msg(sid, "assistant", assistant)

    def sql(self, statement, *args):
        self.conn.execute(statement, args)
        self.conn.commit()


class HashEmbedder:
    """Test double: deterministic bag-of-words vectors. Not evidence of semantic quality."""

    def __init__(self, model_id="test:hash", dim=64):
        self.model_id, self.dim, self.calls = model_id, dim, 0
        self.space = EmbeddingSpace(backend="test", model=model_id, dim=dim)
        self.seen: list[tuple[str, list[str]]] = []

    def embed(self, texts, *, kind):
        self.calls += 1
        self.seen.append((kind, list(texts)))
        rows = []
        for text in texts:
            row = array("f", [0.0] * self.dim)
            for word in re.findall(r"\w+", text.lower()):
                row[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1
            row[0] += 1e-3
            rows.append(row)
        return normalise(rows, len(texts))


class FailingEmbedder:
    model_id = "test:down"

    def embed(self, texts, *, kind):
        raise EmbeddingUnavailable("endpoint unreachable")


@pytest.fixture
def home(tmp_path):
    h = tmp_path / "home"
    (h / "hermes-mem").mkdir(parents=True)
    (h / "hermes-mem/config.yaml").write_text("embedding:\n  provider: none\nsync:\n  background: false\n")
    return h


@pytest.fixture
def state(home):
    return StateDB(home / "state.db")


@pytest.fixture
def app(home):
    a = MemApp(home)
    a._embedder, a._embedder_built = HashEmbedder(), True
    return a
