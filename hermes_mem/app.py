"""Wiring for one Hermes profile: paths, config, caller scope and operations."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import config as config_contract
from . import store
from .config import ConfigError, space_of
from .curated import CuratedError, CuratedStore
from .embed import Embedder, EngineEmbedder, OpenAICompatibleProvider
from .host import open_state, session_scope
from .runtime import EngineClient, EngineRuntime
from .search import SearchParams
from .select import ProcessingPolicy

__all__ = ["ConfigError", "MemApp", "OWNER", "Caller"]


def hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home())
    except ImportError:
        return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")


@dataclass
class Caller:
    """Who is asking. scopes=None means the profile owner (all scopes of this profile)."""

    scope: str
    scopes: list[str] | None

    @property
    def is_owner(self) -> bool:
        return self.scopes is None

    def peer_subject(self) -> str | None:
        """Authoritative peer id this caller may describe in a peer card, if any."""
        if self.scope.startswith("peer:"):
            return self.scope[len("peer:"):]
        if self.scope.startswith("chat:") and "/peer:" in self.scope:
            platform = self.scope.split(":", 2)[1]
            return f"{platform}:{self.scope.rsplit('/peer:', 1)[1]}"
        return None


OWNER = Caller(scope="owner", scopes=None)


def processing_policy(section, config_path: Path, chunk_chars: int | None = None) -> ProcessingPolicy:
    supported = ", ".join(ProcessingPolicy.names())
    section = {} if section is None else section
    if not isinstance(section, dict):
        raise ConfigError(f"{config_path}: processing must be a mapping of {supported}, got {section!r}")
    unknown = sorted(repr(key) if not isinstance(key, str) else key for key in set(section) - set(ProcessingPolicy.names()))
    if unknown:
        raise ConfigError(f"{config_path}: unknown processing key(s) {', '.join(unknown)}; supported: {supported}")
    values = {}
    for name, value in section.items():
        # YAML reads `off`, `no` and an empty value as false/null; all mean the same as `none`.
        if value is None or value is False:
            value = "none"
        elif not isinstance(value, str):
            raise ConfigError(
                f"{config_path}: processing.{name}: {value!r} is not supported; only 'none' is implemented. "
                f"Set processing.{name}: none or remove it."
            )
        values[name] = value.strip().lower()
    if chunk_chars is not None:
        values["chunk_chars"] = chunk_chars
    try:
        return ProcessingPolicy(**values)
    except ValueError as exc:
        raise ConfigError(f"{config_path}: {exc}") from None


class MemApp:
    def __init__(self, home: Path | None = None):
        self.home = Path(home) if home else hermes_home()
        self.data_dir = self.home / "hermes-mem"
        self.state_db = self.home / "state.db"
        config_path = self.data_dir / "config.yaml"
        try:
            loaded = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"{config_path}: not valid YAML: {exc}") from None
        self.config = config_contract.load(loaded, config_path)
        self.policy = processing_policy(self.config["processing"], config_path, self.config["sync"]["chunk_chars"])
        emb, eng = self.config["embedding"], self.config["engine"]
        self.runtime = EngineRuntime(
            self.data_dir, command=eng["command"], idle_seconds=eng["idle_seconds"], timeout=eng["timeout"],
            threads=emb["threads"], cache_dir=emb["cache_dir"],
        )
        self._engine: EngineClient | None = None
        self._embedders: dict[str, Embedder | None] = {}
        self._embedder: Embedder | None = None
        self._embedder_built = False

    def open_store(self) -> sqlite3.Connection:
        return store.connect(self.data_dir / "mem.db")

    @property
    def active_profile(self) -> str:
        return self.config["embedding"]["active"]

    def embedder(self, profile: str | None = None) -> Embedder | None:
        """The embedder of `profile` (default: the active one). Building it starts nothing."""
        if profile is None or profile == self.active_profile:
            if not self._embedder_built:
                self._embedder = self._build(self.active_profile)
                self._embedder_built = True
            return self._embedder
        if profile not in self._embedders:
            self._embedders[profile] = self._build(profile)
        return self._embedders[profile]

    def _build(self, name: str) -> Embedder | None:
        if name == "none":
            return None
        profiles = self.config["embedding"]["profiles"]
        if name not in profiles:
            raise ConfigError(f"unknown embedding profile {name!r}; known: {', '.join(['none', *profiles])}")
        profile = profiles[name]
        if profile["backend"] == "openai":
            return OpenAICompatibleProvider(space_of(profile), profile.get("api_key_env"), self.config["embedding"]["timeout"])
        if self._engine is None:
            self._engine = EngineClient(self.runtime)
        return EngineEmbedder(space_of(profile), self._engine)

    def search_params(self, query: str, scopes, **given) -> SearchParams:
        """A bounded search request from config defaults; `scopes` is the caller's authoritative scope."""
        cfg = self.config["search"]
        return SearchParams(
            query=query, scopes=scopes, mode=given.get("mode") or "hybrid", limit=given.get("limit") or cfg["limit"],
            candidates=cfg["candidates"], context_messages=cfg["context_messages"], context_chars=cfg["context_chars"],
            excerpt_chars=cfg["excerpt_chars"], min_similarity=cfg["min_similarity"],
            conversation=given.get("conversation"), session_id=given.get("session_id"),
            variants=list(given.get("variants") or []), max_variants=cfg["max_variants"],
            variant_chars=cfg["variant_chars"], conversation_chars=cfg["conversation_chars"],
        )

    def close(self) -> None:
        if self._engine is not None:
            self._engine.close()

    def curated(self, conn: sqlite3.Connection, caller: Caller) -> CuratedStore:
        def evidence_role(ref: str) -> str | None:
            """'user' only for a current episode that contains the user's own words."""
            if ref.startswith("episode:"):
                row = conn.execute(
                    "SELECT user_text, scope, status FROM episodes WHERE id=?", (int(ref.split(":")[1]),)
                ).fetchone()
            else:
                session_id, message = ref[len("session:"):].split("#msg:")
                row = conn.execute(
                    """SELECT '' AS user_text, scope, status FROM episodes
                       WHERE session_id=? AND ? BETWEEN first_message_id AND last_message_id
                       ORDER BY version DESC LIMIT 1""",
                    (session_id, int(message)),
                ).fetchone()
            if row is None or (caller.scopes is not None and row["scope"] not in caller.scopes):
                return None
            return "user" if row["user_text"] and row["status"] == "current" else "context"

        return CuratedStore(conn, evidence_check=evidence_role)

    @property
    def legacy_curated_dir(self) -> Path:
        """Where releases before the SQLite store kept curated Markdown; read only by `curated import`."""
        return self.data_dir / "curated"

    def caller_for_session(self, session_id: str | None) -> Caller:
        """Resolve scope from the authoritative state.db session row; unknown means narrowest scope."""
        if not session_id or not self.state_db.exists():
            return Caller(scope=f"session:{session_id or 'unknown'}", scopes=[f"session:{session_id or 'unknown'}"])
        with open_state(self.state_db) as host:
            sessions = host.sessions(ids=[session_id])
        if not sessions:
            return Caller(scope=f"session:{session_id}", scopes=[f"session:{session_id}"])
        scope = session_scope(sessions[0])
        return OWNER if scope == "owner" else Caller(scope=scope, scopes=[scope])


def check_peer_card(caller: Caller, kind: str, subject: str) -> None:
    if kind != "peer" or caller.is_owner:
        return
    allowed = caller.peer_subject()
    if allowed is None:
        raise CuratedError("peer cards need an authoritative participant identity; this conversation has none")
    if subject != allowed:
        raise CuratedError(f"this conversation may only describe its own participant {allowed!r}")
