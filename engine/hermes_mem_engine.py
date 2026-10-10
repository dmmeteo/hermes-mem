"""hermes-mem embedding engine worker. Runs only in the locked engine runtime, never inside Hermes.

Protocol: one JSON request per stdin line, one JSON reply per stdout line.
  {"protocol": 1, "op": "embed", "space": {...EmbeddingSpace...}, "kind": "query"|"passage", "texts": [...]}
  -> {"vectors": [[...], ...], "dim": N, "source_revision": "<hf snapshot>", "load_seconds": S|null} | {"error": "..."}
Texts arrive already prefixed for their space. `load_seconds` is set only on the call that loaded the model. Exits after HERMES_MEM_ENGINE_IDLE seconds idle
or when stdin closes. `--probe` prints the runtime's package versions and exits.
"""

import json
import os
import select
import sys
import time
from pathlib import Path

PROTOCOL = 1


# Files fastembed reads from a model directory besides the ONNX graph.
MODEL_DIR_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
BATCH_SIZE = 16


def probe() -> dict:
    import fastembed
    import huggingface_hub
    import numpy
    import onnxruntime

    return {
        "python": sys.version.split()[0],
        "fastembed": fastembed.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "numpy": numpy.__version__,
        "onnxruntime": onnxruntime.__version__,
        "protocol": PROTOCOL,
    }


class Models:
    def __init__(self):
        self.loaded = {}
        self.registered = {}  # model name -> (source, file, pooling, dim) this worker registered with fastembed
        self.cache_dir = os.environ.get("HERMES_MEM_ENGINE_CACHE") or None
        self.threads = int(os.environ.get("HERMES_MEM_ENGINE_THREADS") or 1)

    def get(self, space: dict):
        """(model, snapshot revision, load seconds or None if it was already loaded)."""
        key = space["space_id"]
        if key in self.loaded:
            model, revision = self.loaded[key]
            return model, revision, None
        started = time.monotonic()
        from fastembed import TextEmbedding

        if space.get("hf_source"):
            from fastembed.common.model_description import ModelSource, PoolingType

            # fastembed keys models by name only. Reusing a name for another file, pooling or source would
            # silently load the first registration under the second space id, so that is an error instead.
            name = space["model"].lower()
            registration = (space["hf_source"], space.get("model_file") or "onnx/model.onnx",
                            (space.get("pooling") or "mean").upper(), int(space["dim"]))
            if name in self.registered:
                if self.registered[name] != registration:
                    raise ValueError(f"model name {space['model']!r} is already registered with another source, file, "
                                     "pooling or dim in this engine; give each such profile a distinct `model` name")
            elif name in {m["model"].lower() for m in TextEmbedding.list_supported_models()}:
                raise ValueError(f"model name {space['model']!r} is a fastembed built-in; a profile with hf_source "
                                 "needs a distinct `model` name")
            else:
                TextEmbedding.add_custom_model(
                    model=space["model"],
                    pooling=PoolingType[registration[2]],
                    normalization=True,
                    sources=ModelSource(hf=space["hf_source"]),
                    dim=registration[3],
                    model_file=registration[1],
                )
                self.registered[name] = registration
        kwargs = {"cache_dir": self.cache_dir, "threads": self.threads}
        if space.get("revision"):
            # fastembed 0.9.0 ignores a `revision` argument and fetches the repository head, so the
            # pinned snapshot is fetched here and handed over as the model directory.
            from huggingface_hub import snapshot_download

            kwargs["specific_model_path"] = snapshot_download(
                repo_id=space["hf_source"], revision=space["revision"], cache_dir=self.cache_dir,
                allow_patterns=[space.get("model_file") or "onnx/model.onnx", *MODEL_DIR_FILES],
            )
        model = TextEmbedding(space["model"], **kwargs)
        model_dir = Path(getattr(model.model, "_model_dir", "") or kwargs.get("specific_model_path") or "")
        revision = model_dir.name if model_dir.parent.name == "snapshots" else None
        self.loaded[key] = (model, revision)
        return model, revision, round(time.monotonic() - started, 3)

    def embed(self, request: dict) -> dict:
        space = request["space"]
        if space.get("backend") != "fastembed":
            return {"error": f"backend {space.get('backend')!r} does not run in the engine"}
        model, revision, load_seconds = self.get(space)
        pinned = space.get("revision") or ""
        if len(pinned) == 40 and revision != pinned:  # a commit sha must match exactly; a branch resolves to one
            return {"error": f"loaded snapshot {revision!r} is not the pinned revision {space['revision']!r}"}
        rows = [vector.tolist() for vector in model.embed(list(request["texts"]), batch_size=BATCH_SIZE)]
        return {"vectors": rows, "dim": len(rows[0]) if rows else 0, "source_revision": revision,
                "load_seconds": load_seconds}


def serve() -> int:
    idle = float(os.environ.get("HERMES_MEM_ENGINE_IDLE") or 300)
    models = Models()
    while True:
        ready, _, _ = select.select([sys.stdin], [], [], idle)
        if not ready:
            return 0
        line = sys.stdin.readline()
        if not line:
            return 0
        try:
            request = json.loads(line)
            if request.get("protocol") != PROTOCOL:
                reply = {"error": f"protocol {request.get('protocol')!r} != engine protocol {PROTOCOL}; re-run setup"}
            elif request.get("op") == "embed":
                reply = models.embed(request)
            elif request.get("op") == "ping":
                reply = {"ok": True, **probe()}
            else:
                reply = {"error": f"unknown op {request.get('op')!r}"}
        except Exception as exc:  # report, keep serving
            reply = {"error": f"{type(exc).__name__}: {exc}"[:800]}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    if sys.argv[1:] == ["--probe"]:
        print(json.dumps(probe()))
        sys.exit(0)
    sys.exit(serve())
