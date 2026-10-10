"""Embedding spaces and providers. Pure stdlib: compiled model code runs in the external engine.

An embedding space is the exact identity that makes two vectors comparable: backend, model,
revision, and the query/passage preprocessing. Its `space_id` keys every stored vector, so a
query is only ever compared with vectors of its own space, and switching profiles keeps the
old space's vectors for a later switch back.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from array import array
from dataclasses import asdict, dataclass
from typing import Protocol, Sequence

Vector = array  # array('f'), L2-normalised


class EmbeddingUnavailable(RuntimeError):
    """The configured space cannot produce vectors right now."""


@dataclass(frozen=True)
class EmbeddingSpace:
    backend: str  # fastembed | openai
    model: str
    revision: str | None = None
    query_prefix: str = ""
    passage_prefix: str = ""
    # fastembed models outside its built-in list are registered from Hugging Face sources.
    hf_source: str | None = None
    model_file: str | None = None
    pooling: str | None = None
    dim: int | None = None
    base_url: str | None = None

    @property
    def space_id(self) -> str:
        """Stable key. The legacy MVP key `fastembed:<model>` is kept when nothing else differs."""
        base = f"openai:{(self.base_url or '').rstrip('/')}#{self.model}" if self.backend == "openai" else f"{self.backend}:{self.model}"
        extra = []
        if self.revision:
            extra.append(f"rev={self.revision}")
        if self.query_prefix or self.passage_prefix:
            extra.append("pre=" + json.dumps([self.query_prefix, self.passage_prefix], ensure_ascii=False))
        if self.hf_source:
            extra.append(f"src={self.hf_source}/{self.model_file or 'onnx/model.onnx'}/{self.pooling or 'mean'}")
        return base + ("|" + "|".join(extra) if extra else "")

    def prepare(self, texts: Sequence[str], kind: str) -> list[str]:
        prefix = self.query_prefix if kind == "query" else self.passage_prefix
        return [prefix + text for text in texts]

    def describe(self) -> dict:
        data = {k: v for k, v in asdict(self).items() if v not in (None, "")}
        data["space_id"] = self.space_id
        return data


class Embedder(Protocol):
    space: EmbeddingSpace

    @property
    def model_id(self) -> str: ...

    def embed(self, texts: Sequence[str], *, kind: str) -> list[Vector]: ...


def normalise(rows: Sequence[Sequence[float]], expected: int, dim: int | None = None) -> list[Vector]:
    if len(rows) != expected:
        raise EmbeddingUnavailable(f"expected {expected} vectors, got {len(rows)}")
    out = []
    for row in rows:
        values = [float(x) for x in row]
        norm = math.sqrt(math.fsum(x * x for x in values))
        if not values or not all(math.isfinite(x) for x in values) or norm == 0:
            raise EmbeddingUnavailable("provider returned empty, zero or non-finite vectors")
        if dim is not None and len(values) != dim:
            raise EmbeddingUnavailable(f"provider returned {len(values)}-dim vectors; the space declares {dim}")
        out.append(array("f", (x / norm for x in values)))
    if len({len(v) for v in out}) > 1:
        raise EmbeddingUnavailable("provider returned vectors of different sizes")
    return out


def dot(a: Vector, b: Vector) -> float:
    return math.fsum(x * y for x, y in zip(a, b))


class _SpaceEmbedder:
    def __init__(self, space: EmbeddingSpace):
        self.space = space

    @property
    def model_id(self) -> str:
        return self.space.space_id


class EngineEmbedder(_SpaceEmbedder):
    """fastembed/ONNX in the separately provisioned engine process (see runtime.py)."""

    def __init__(self, space: EmbeddingSpace, client):
        super().__init__(space)
        self.client = client
        self.source_revision: str | None = None  # the model snapshot the engine actually loaded
        self.load_seconds: float | None = None  # set by the call that loaded the model

    def embed(self, texts: Sequence[str], *, kind: str) -> list[Vector]:
        reply = self.client.call(
            {"op": "embed", "space": self.space.describe(), "kind": kind, "texts": self.space.prepare(texts, kind)}
        )
        self.source_revision = reply.get("source_revision") or self.source_revision
        if reply.get("load_seconds") is not None:
            self.load_seconds = reply["load_seconds"]
        return normalise(reply["vectors"], len(texts), self.space.dim)


class OpenAICompatibleProvider(_SpaceEmbedder):
    """Any `/v1/embeddings` endpoint (llama.cpp, Ollama, vLLM, TEI, ...). Key read from env, never logged."""

    def __init__(self, space: EmbeddingSpace, api_key_env: str | None = None, timeout: float = 30.0):
        super().__init__(space)
        self.api_key_env, self.timeout = api_key_env, timeout

    def embed(self, texts: Sequence[str], *, kind: str) -> list[Vector]:
        headers = {"Content-Type": "application/json"}
        if self.api_key_env and os.environ.get(self.api_key_env):
            headers["Authorization"] = f"Bearer {os.environ[self.api_key_env]}"
        body = json.dumps({"model": self.space.model, "input": self.space.prepare(texts, kind)}).encode()
        request = urllib.request.Request(
            (self.space.base_url or "").rstrip("/") + "/embeddings", data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise EmbeddingUnavailable(f"embedding endpoint failed: {exc}") from exc
        try:
            data = sorted(payload["data"], key=lambda item: item["index"])
            rows = [item["embedding"] for item in data]
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingUnavailable(f"unexpected embedding response: {exc}") from exc
        return normalise(rows, len(texts), self.space.dim)
