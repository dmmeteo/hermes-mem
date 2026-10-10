# 0006 BGE-M3 INT8 as the default embedding profile, with a shared model cache

Status: accepted, 2026-10-10.

## Context

MiniLM (`paraphrase-multilingual-MiniLM-L12-v2`, 384 dimensions) was the default. On the paraphrase questions, a literal query reached only hit@1 0.500 on the pricing subset without agent-written variants, so recall in another language or spelling depended on the agent writing good variants. Each Hermes profile also kept its own copy of the model weights under `$HERMES_HOME/hermes-mem/models`, so several profiles on one machine downloaded and stored the same files several times.

The candidates ran in one `scripts/compare_models.py` run at source `49b3bcf`, in Mac Docker Linux ARM64 with 2 CPUs and 4 GiB, all on dense INT8 ONNX. The corpus was the frozen synthetic set (23 episodes, 22 Ukrainian/English questions, including transliterations and distractors). Each candidate embedded the same chunks:

| Profile | dim | semantic hit@1 / MRR, literal query (22) | pricing hit@1, literal (8) | warm query median | cold first query |
|---|---|---|---|---|---|
| `minilm` | 384 | 0.773 / 0.845 | 0.500 | 0.013 s | 0.90 s |
| `bge-m3-int8` | 1024 | 0.955 / 0.977 | 0.875 | 0.052 s | 1.31 s |
| `qwen3-0.6b-int8` | 1024 | 21/22 hit@1 | | 1.065 s | 2.76 s |

All five comparison gates passed: every profile ran, all vectors were real with full coverage, no search degraded, the spaces were distinct, and the pinned revision ran. BGE and Qwen tied on literal hit@1; BGE answered warm queries about 20× faster.

## Decision

- **Default profile `bge-m3-int8`.** The upstream model is [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (MIT; dense, sparse and multi-vector retrieval; 100+ languages; up to 8192 tokens; 1024-dimensional dense vectors; [arXiv:2402.03216](https://arxiv.org/abs/2402.03216)). The weights come from the third-party ONNX export [Xenova/bge-m3](https://huggingface.co/Xenova/bge-m3) (MIT, made for Transformers.js), file `onnx/model_quantized.onnx` (INT8), pinned to revision `4de13258303883538bd53b696b452bf8099f0858`. Pooling is CLS, there are no query or passage prefixes, and only the dense output is used.
- **Pinned, never substituted.** The engine downloads exactly that snapshot and refuses any other. If the model cannot load, search reports `degraded` keyword mode. It never falls back to another model on its own.
- **MiniLM stays** as the `minilm` profile, a control or fallback the operator selects explicitly (`embedding.active: minilm`). `e5-small` also remains. The legacy MVP keys (`provider: fastembed`) still mean MiniLM, so stores configured that way keep their vectors.
- **Spaces stay isolated.** The space id names the model, revision, source file (and with it the precision) and pooling. A query meets only vectors of its own space. fastembed registers custom models by name alone, so the engine refuses a second profile that reuses a model name with another source, file, pooling or dimension, and refuses a custom profile that reuses a built-in name. It never loads one registration under another space id. After the default changes, an existing store re-embeds lazily into the new space. The old space's vectors are kept, `status.profiles` shows `pending` and `failed` for every profile, and search warns while coverage is partial. `hermes hermes-mem index` (or `embed`) finishes the re-embedding in bounded, resumable batches.
- **Shared model cache.** Weights default to `$XDG_CACHE_HOME/hermes-mem/models`, or `~/.cache/hermes-mem/models` when that is unset or relative. Every profile of the same user shares it, and Hugging Face's file locks serialise concurrent downloads. `embedding.cache_dir` overrides it: `~` is expanded, and a relative path is relative to the config file. Resolving the path creates nothing. Episodes, vectors, curated records, the engine runtime and its worker process remain per profile.

## Limits

- The evidence is small, synthetic and hand-labelled. It covers one platform (Linux ARM64) and one precision (dense INT8). It does not show production quality on real history, and it makes no comparison with Hindsight or `session_search`. The Qwen row is only the reference for the tie; its other metrics are not repeated here.
- The ONNX export is third-party. Its fidelity to the upstream PyTorch weights was not measured here.
- BGE vectors are 1024-dimensional (4 KiB each, against 1.5 KiB for MiniLM). The pure-Python scan therefore costs about 2.7× more per chunk, and the model takes more memory in the engine process than MiniLM.
- The first use downloads about 590 MB (`onnx/model_quantized.onnx` is 569,694,530 bytes, SHA-256 `0826f8c1ab9edf1801db86c61919d4d108e8bfc0b809ec823ad366882ff0b77d`, plus the tokenizer files), unless the shared cache already holds the snapshot. Hugging Face's cache stores that blob under its SHA-256, so a cached copy can be checked against it.
