"""Pure selection rules: host message rows -> meaningful conversation episodes."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field, fields, replace

ACKNOWLEDGEMENTS = {
    "ok", "okay", "k", "kk", "thanks", "thank you", "thx", "ty", "cool", "nice", "great",
    "ок", "окей", "добре", "дякую", "спасибі", "спасибо", "угу", "ага", "супер", "клас",
    "+", "+1", "👍", "🙏", "👌", "❤️",
}
LOG_LINE = re.compile(
    r"^\s*(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}|\[\d{2}:\d{2}|Traceback |File \"|at \S+\(|"
    r"(DEBUG|INFO|WARN|WARNING|ERROR|TRACE|FATAL)\b|\s+\^+\s*$|[A-Za-z_.]+(Error|Exception):)"
)
CHUNK_CHARS = 1400
CHUNK_CHARS_RANGE = (400, 8000)
QUESTION_PREFIX_CHARS = 300


@dataclass(frozen=True)
class HostMessage:
    id: int
    ref: str
    role: str
    content: str
    timestamp: float
    has_tool_calls: bool
    compacted: bool
    summary: bool


@dataclass
class Episode:
    key: str
    user_ref: str | None
    assistant_refs: list[str]
    first_message_id: int
    last_message_id: int
    user_text: str
    assistant_text: str
    started_at: float
    ended_at: float
    compacted: bool
    chunks: list[str] = field(default_factory=list)

    @property
    def content_hash(self) -> str:
        return sha256(f"{self.user_text}\x00{self.assistant_text}")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_acknowledgement(text: str) -> bool:
    normal = unicodedata.normalize("NFKC", text).strip().lower().rstrip("!.)( ")
    return normal in ACKNOWLEDGEMENTS


def looks_like_log(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 5:
        return False
    return sum(bool(LOG_LINE.match(line)) for line in lines) / len(lines) >= 0.6


def visible_text(message: HostMessage) -> str | None:
    """The user-visible text of a message worth remembering, or None."""
    if message.summary or message.role not in ("user", "assistant"):
        return None
    text = (message.content or "").strip()
    if not text or looks_like_log(text):
        return None
    if message.role == "assistant" and message.has_tool_calls:
        return None
    return text


def chunk_episode(user_text: str, assistant_text: str, chunk_chars: int = CHUNK_CHARS) -> list[str]:
    whole = f"User: {user_text}\n\nAssistant: {assistant_text}".strip()
    if len(whole) <= chunk_chars:
        return [whole]
    question = user_text if len(user_text) <= QUESTION_PREFIX_CHARS else user_text[:QUESTION_PREFIX_CHARS] + "…"
    prefix = f"User: {question}\n\nAssistant: " if user_text else "Assistant: "
    budget = chunk_chars - len(prefix)
    pieces: list[str] = []
    if len(user_text) > QUESTION_PREFIX_CHARS:
        pieces += [f"User: {part}" for part in _split(user_text, chunk_chars - 6)]
    if assistant_text:
        pieces += [prefix + part for part in _split(assistant_text, budget)]
    return pieces


def _split(text: str, budget: int) -> list[str]:
    parts, current = [], ""
    for paragraph in re.split(r"(\n\s*\n)", text):
        while len(paragraph) > budget:
            cut = paragraph.rfind(" ", 0, budget)
            cut = cut if cut > budget // 2 else budget
            if current:
                parts.append(current)
                current = ""
            parts.append(paragraph[:cut])
            paragraph = paragraph[cut:].lstrip()
        if len(current) + len(paragraph) > budget and current.strip():
            parts.append(current)
            current = ""
        current += paragraph
    if current.strip():
        parts.append(current)
    return [p.strip() for p in parts if p.strip()]


@dataclass(frozen=True)
class ProcessingPolicy:
    """What happens to selected episodes on their way into the store.

    Only the raw defaults exist: excerpts are stored verbatim, no model extracts or
    consolidates them, nothing is compressed and nothing expires. Any other value is
    refused at construction, so a policy that did not run can never be reported as active.
    `chunk_chars` only sizes the verbatim chunks; changing it re-chunks stored episodes as new versions.
    """

    retain: str = "none"
    compression: str = "none"
    expiry: str = "none"
    chunk_chars: int = field(default=CHUNK_CHARS, metadata={"chunking": True})

    def __post_init__(self) -> None:
        low, high = CHUNK_CHARS_RANGE
        if isinstance(self.chunk_chars, bool) or not isinstance(self.chunk_chars, int) or not low <= self.chunk_chars <= high:
            raise ValueError(f"sync.chunk_chars must be an integer between {low} and {high}, got {self.chunk_chars!r}")
        for name in self.names():
            value = getattr(self, name)
            if value != "none":
                raise ValueError(
                    f"processing.{name}: {value!r} is not implemented in this hermes-mem version; "
                    f"only 'none' is supported. Set processing.{name}: none or remove it."
                )

    @classmethod
    def names(cls) -> list[str]:
        return [f.name for f in fields(cls) if not f.metadata.get("chunking")]

    def describe(self) -> dict:
        return {"excerpts": "raw", **{name: getattr(self, name) for name in self.names()}}

    def apply(self, episodes: list[Episode]) -> list[Episode]:
        """Turn selected episodes into the excerpts to store: verbatim chunks of the originals."""
        return [replace(ep, chunks=chunk_episode(ep.user_text, ep.assistant_text, self.chunk_chars)) for ep in episodes]


def build_episodes(session_id: str, messages: list[HostMessage]) -> list[Episode]:
    """Group visible messages into user-request + final-answer episodes.

    A user message opens an episode; visible assistant answers until the next
    user message belong to it. Assistant text without a preceding user message
    forms its own episode. Bare acknowledgements with no substantive answer are
    dropped; short corrections and decisions are kept regardless of length.
    """
    groups: list[tuple[HostMessage | None, list[tuple[HostMessage, str]]]] = []
    for message in sorted(messages, key=lambda m: m.id):
        text = visible_text(message)
        if message.role == "user":
            groups.append((message if text else None, []))
            if text is None:
                continue
        elif text is not None:
            if not groups:
                groups.append((None, []))
            groups[-1][1].append((message, text))
    episodes = []
    for user, answers in groups:
        user_text = (user.content or "").strip() if user else ""
        assistant_text = "\n\n".join(text for _, text in answers)
        if not user_text and not assistant_text:
            continue
        if user_text and is_acknowledgement(user_text) and len(assistant_text) < 80:
            continue
        members = ([user] if user else []) + [m for m, _ in answers]
        anchor = user or answers[0][0]
        episodes.append(
            Episode(
                key=f"{session_id}/{anchor.ref}",
                user_ref=user.ref if user else None,
                assistant_refs=[m.ref for m, _ in answers],
                first_message_id=min(m.id for m in members),
                last_message_id=max(m.id for m in members),
                user_text=user_text,
                assistant_text=assistant_text,
                started_at=min(m.timestamp for m in members),
                ended_at=max(m.timestamp for m in members),
                compacted=any(m.compacted for m in members),
            )
        )
    return episodes
