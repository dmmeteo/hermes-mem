"""Strict hermes-mem/config.yaml contract: defaults, validation and model profiles.

Every key here changes real behaviour; anything else fails loading with the offending path.
Scope and access rules are deliberately not configurable.
"""

from __future__ import annotations

import os
from pathlib import Path

from .embed import EmbeddingSpace
from .select import CHUNK_CHARS, CHUNK_CHARS_RANGE

DEFAULT_GUIDANCE = (
    "Write `query` in the user's own words and keep it as the literal anchor. When the topic may have "
    "been discussed in another language or spelling, add 1-3 `variants`: the Ukrainian and English "
    "equivalents and a Latin/Cyrillic transliteration of names or terms (e.g. 'тарифи' / 'pricing' / "
    "'taryfy'). Keep identifiers, numbers, names and negation ('not', 'не') exactly; never add facts "
    "the user did not say. For follow-ups pass the recent turns as `conversation`."
)

DEFAULT_PROFILE = "bge-m3-int8"

BUILTIN_PROFILES: dict[str, dict] = {
    # BAAI/bge-m3 dense embeddings from the third-party ONNX INT8 export Xenova/bge-m3, pinned to the
    # snapshot the model comparison ran (ADR 0006). CLS pooling, 1024 dimensions, no prefixes.
    "bge-m3-int8": {
        "backend": "fastembed",
        "model": "BAAI/bge-m3",
        "revision": "4de13258303883538bd53b696b452bf8099f0858",
        "hf_source": "Xenova/bge-m3",
        "model_file": "onnx/model_quantized.onnx",
        "pooling": "cls",
        "dim": 1024,
    },
    # Smaller control/fallback; selected only explicitly, never substituted for the default.
    "minilm": {"backend": "fastembed", "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "dim": 384},
    # Not in fastembed 0.9.0's built-in list. The file and revision were checked against the public
    # Hugging Face file listing (onnx/model.onnx, fp32, 470 MB at 614241f); the engine downloads exactly that.
    "e5-small": {
        "backend": "fastembed",
        "model": "intfloat/multilingual-e5-small",
        "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        "hf_source": "intfloat/multilingual-e5-small",
        "model_file": "onnx/model.onnx",
        "pooling": "mean",
        "dim": 384,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
}

# (type, default, (min, max) or choices)
SCHEMA: dict[str, dict] = {
    "embedding": {
        "active": (str, DEFAULT_PROFILE, None),
        "profiles": (dict, {}, None),
        # MVP keys, still accepted: they describe one profile named "default" and make it active.
        "provider": (str, None, ("fastembed", "openai", "none")),
        "model": (str, None, None),
        "base_url": (str, None, None),
        "api_key_env": (str, None, None),
        "timeout": ((int, float), 30.0, (1, 600)),
        "threads": (int, 1, (1, 32)),
        "cache_dir": (str, None, None),
        "batch_size": (int, 32, (1, 256)),
    },
    "engine": {
        "command": (list, None, None),
        "idle_seconds": ((int, float), 300.0, (5, 86400)),
        "timeout": ((int, float), 120.0, (5, 3600)),
    },
    "search": {
        "limit": (int, 8, (1, 20)),
        "candidates": (int, 40, (1, 100)),
        "context_messages": (int, 2, (0, 5)),
        "context_chars": (int, 1200, (0, 8000)),
        "excerpt_chars": (int, 900, (100, 4000)),
        "min_similarity": ((int, float), 0.25, (-1.0, 1.0)),
        "max_variants": (int, 3, (0, 6)),
        "variant_chars": (int, 300, (20, 2000)),
        "conversation_chars": (int, 600, (0, 4000)),
        "instructions": (str, None, None),
        "instructions_file": (str, None, None),
    },
    "sync": {
        "max_sessions": (int, 200, (1, 100000)),
        "background": (bool, True, None),
        "chunk_chars": (int, CHUNK_CHARS, CHUNK_CHARS_RANGE),
    },
}

PROFILE_KEYS = {
    "backend": (str, None, ("fastembed", "openai")),
    "model": (str, None, None),
    "revision": (str, None, None),
    "query_prefix": (str, "", None),
    "passage_prefix": (str, "", None),
    "hf_source": (str, None, None),
    "model_file": (str, None, None),
    "pooling": (str, None, ("mean", "cls")),
    "dim": (int, None, (1, 8192)),
    "base_url": (str, None, None),
    "api_key_env": (str, None, None),
}


class ConfigError(ValueError):
    """hermes-mem/config.yaml asks for something this version cannot do."""


def _check(path: str, value, spec, where: Path):
    kind, _default, bounds = spec
    if value is None:
        return None
    kinds = kind if isinstance(kind, tuple) else (kind,)
    if isinstance(value, bool) and bool not in kinds:
        raise ConfigError(f"{where}: {path} must be {'/'.join(k.__name__ for k in kinds)}, got {value!r}")
    if not isinstance(value, kinds):
        raise ConfigError(f"{where}: {path} must be {'/'.join(k.__name__ for k in kinds)}, got {value!r}")
    if bounds and isinstance(bounds[0], str):
        if value.lower() not in bounds:
            raise ConfigError(f"{where}: {path} must be one of {', '.join(bounds)}, got {value!r}")
        return value.lower()
    if bounds and not (bounds[0] <= value <= bounds[1]):
        raise ConfigError(f"{where}: {path} must be between {bounds[0]} and {bounds[1]}, got {value!r}")
    return value


def load(raw: dict | None, where: Path) -> dict:
    """Validated effective config. `processing` is validated separately by app.processing_policy."""
    raw = {} if raw is None else raw
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: top level must be a mapping")
    unknown = sorted(set(raw) - set(SCHEMA) - {"processing"})
    if unknown:
        raise ConfigError(f"{where}: unknown section(s) {', '.join(map(str, unknown))}; supported: {', '.join([*SCHEMA, 'processing'])}")
    out: dict = {"processing": raw.get("processing")}
    for section, keys in SCHEMA.items():
        given = raw.get(section) or {}
        if not isinstance(given, dict):
            raise ConfigError(f"{where}: {section} must be a mapping")
        extra = sorted(map(str, set(given) - set(keys)))
        if extra:
            raise ConfigError(f"{where}: unknown key(s) {section}.{', '.join(extra)}; supported: {', '.join(keys)}")
        out[section] = {
            key: (_check(f"{section}.{key}", given[key], spec, where) if key in given else spec[1])
            for key, spec in keys.items()
        }
    command = out["engine"]["command"]
    if command is not None and (not command or not all(isinstance(a, str) and a for a in command)):
        raise ConfigError(f"{where}: engine.command must be a non-empty argv list of strings (no shell string)")
    out["embedding"]["profiles"] = _profiles(out["embedding"], where)
    if out["embedding"]["active"] != "none" and out["embedding"]["active"] not in out["embedding"]["profiles"]:
        raise ConfigError(
            f"{where}: embedding.active {out['embedding']['active']!r} is not a profile; "
            f"known: {', '.join(['none', *out['embedding']['profiles']])}"
        )
    out["embedding"]["cache_dir"] = str(_model_cache(out["embedding"]["cache_dir"], where))
    out["search"]["guidance"] = _guidance(out["search"], where)
    return out


def _profiles(section: dict, where: Path) -> dict[str, dict]:
    profiles = {name: dict(p) for name, p in BUILTIN_PROFILES.items()}
    for name, given in (section["profiles"] or {}).items():
        if not isinstance(name, str) or not isinstance(given, dict):
            raise ConfigError(f"{where}: embedding.profiles.{name} must be a mapping")
        extra = sorted(map(str, set(given) - set(PROFILE_KEYS)))
        if extra:
            raise ConfigError(f"{where}: unknown key(s) embedding.profiles.{name}.{', '.join(extra)}")
        profiles[name] = {k: _check(f"embedding.profiles.{name}.{k}", v, PROFILE_KEYS[k], where) for k, v in given.items()}
    provider = section.pop("provider")
    if provider == "none":
        section["active"] = "none"
    elif provider:
        legacy = {"backend": provider, "model": section["model"] or BUILTIN_PROFILES["minilm"]["model"]}
        if provider == "openai":
            if not section["base_url"] or not section["model"]:
                raise ConfigError(f"{where}: embedding.provider=openai needs base_url and model")
            legacy.update(base_url=section["base_url"], api_key_env=section["api_key_env"])
        profiles["default"] = legacy
        section["active"] = "default"
    for name, p in profiles.items():
        if not p.get("backend") or not p.get("model"):
            raise ConfigError(f"{where}: embedding.profiles.{name} needs backend and model")
        if p["backend"] == "openai" and not p.get("base_url"):
            raise ConfigError(f"{where}: embedding.profiles.{name} (openai) needs base_url")
        if p.get("hf_source") and not p.get("dim"):
            raise ConfigError(f"{where}: embedding.profiles.{name} with hf_source needs dim")
        if p["backend"] == "fastembed" and p.get("revision") and not p.get("hf_source"):
            # fastembed itself always fetches the repository head; only an explicit source can be pinned.
            raise ConfigError(f"{where}: embedding.profiles.{name}.revision needs hf_source (and model_file, dim) to pin")
    return profiles


def default_model_cache() -> Path:
    """Model weights are shared by every profile of this user: $XDG_CACHE_HOME or ~/.cache, then hermes-mem/models."""
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg and Path(xdg).is_absolute() else Path.home() / ".cache"
    return base / "hermes-mem" / "models"


def _model_cache(given: str | None, where: Path) -> Path:
    if not given:
        return default_model_cache()
    path = Path(given).expanduser()
    return path if path.is_absolute() else where.parent / path


def space_of(profile: dict) -> EmbeddingSpace:
    return EmbeddingSpace(**{k: v for k, v in profile.items() if k != "api_key_env" and v is not None})


def _guidance(section: dict, where: Path) -> str:
    inline, file = section["instructions"], section["instructions_file"]
    if inline and file:
        raise ConfigError(f"{where}: set search.instructions or search.instructions_file, not both")
    if file:
        path = Path(file).expanduser()
        if not path.is_absolute():
            path = where.parent / path
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise ConfigError(f"{where}: search.instructions_file {path} does not exist") from None
        except (OSError, UnicodeDecodeError) as exc:
            raise ConfigError(f"{where}: search.instructions_file {path} is unreadable: {exc}") from None
        inline = text
    if inline is None:
        return DEFAULT_GUIDANCE
    inline = inline.strip()
    if not inline or len(inline) > 4000:
        raise ConfigError(f"{where}: search instructions must be 1-4000 characters, got {len(inline)}")
    return inline
