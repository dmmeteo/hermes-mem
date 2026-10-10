# Configuration

Optional `$HERMES_HOME/hermes-mem/config.yaml`. Defaults are shown below. The schema is strict: an unknown key, a wrong type or an out-of-range value stops the plugin from loading, and the error names the key. `status` shows the effective settings.

```yaml
embedding:
  active: bge-m3-int8         # bge-m3-int8 | minilm | e5-small | none | a profile below
  profiles: {}                # extra or overriding profiles, see below
  threads: 1                  # ONNX threads in the engine
  batch_size: 32              # chunks per embedding request
  cache_dir: null             # model weights; default ~/.cache/hermes-mem/models, shared by all profiles
  timeout: 30                 # seconds per request to an OpenAI-compatible endpoint
engine:
  command: null               # argv list replacing the provisioned runtime; never a shell string
  idle_seconds: 300
  timeout: 120                # seconds per engine request; raise it for a model's first use, which downloads it
search:
  limit: 8
  candidates: 40
  context_messages: 2
  context_chars: 1200
  excerpt_chars: 900
  min_similarity: 0.25
  max_variants: 3             # semantic variants used per search
  variant_chars: 300
  conversation_chars: 600
  instructions: null          # replaces the default tool guidance (1-4000 chars)
  instructions_file: null     # or read it from a file, relative to this config
sync:
  max_sessions: 200
  background: true
  chunk_chars: 1400           # 400-8000
processing: {retain: none, compression: none, expiry: none}
```

- **Model profiles.** `bge-m3-int8`, the default, is [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) dense embeddings (1024 dimensions, CLS pooling, no prefixes). It loads from the third-party ONNX INT8 export `Xenova/bge-m3`, file `onnx/model_quantized.onnx`, pinned to revision `4de13258…`. `minilm` is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, fastembed's built-in quantised ONNX export (384 dimensions, about 220 MB), kept as an explicit control or fallback. `e5-small` is `intfloat/multilingual-e5-small` (384 dimensions, fp32, pinned to `614241f6…`, with `query: ` / `passage: ` prefixes). A profile takes `backend` (`fastembed` or `openai`), `model`, `revision`, `query_prefix`, `passage_prefix`, `hf_source`, `model_file`, `pooling`, `dim`, `base_url` and `api_key_env`. A pinned fastembed `revision` needs an explicit `hf_source`, because fastembed itself always fetches the repository head.
- **Default model** ([ADR 0006](adr/0006-bge-m3-default-and-shared-model-cache.md)). In the `compare_models.py` run on 22 synthetic Ukrainian/English questions (Linux ARM64, dense INT8), BGE-M3 reached literal-query semantic hit@1/MRR 0.955/0.977, against 0.773/0.845 for MiniLM. On the eight pricing questions it reached 0.875 without variants, against 0.500. Its warm query median was 0.052 s, against 0.013 s for MiniLM. This is small synthetic evidence, not a benchmark on real history or against Hindsight. The default is never silently replaced: if BGE cannot load, search degrades visibly to keyword mode. Choose MiniLM explicitly with `active: minilm`.
- **Model cache.** `cache_dir` unset means `$XDG_CACHE_HOME/hermes-mem/models`, or `~/.cache/hermes-mem/models` when `XDG_CACHE_HOME` is unset. `~` is expanded, and a relative path is relative to this config file.
- **Embedding spaces.** Each profile is an embedding space identified by backend, model, revision and prefixes. Vectors are stored per space and queries only meet vectors of their own space, even when dimensions match. Switching `active` keeps the old space's vectors, so switching back re-embeds nothing. `status.profiles` shows `current_vectors` and `pending` for every configured profile, including ones never embedded.
- **Legacy MVP keys.** The MVP form (`provider`, `model`, `base_url`, `api_key_env`) still works, and `provider: fastembed` without a model still means MiniLM. It defines a profile named `default` with the same space id, so existing vectors stay valid. `provider: none` means keyword only.
- **Chunk size.** `sync.chunk_chars` sets the size of the verbatim chunks. When it changes, the next sync stores each affected episode as a new version, marks the old one `superseded` with reason `re-chunked at sync.chunk_chars=N`, and reports `rechunked`. The new chunks become `pending` for every space, so stale chunks are never searched. If a chunk's text is identical to one in an earlier version, its vector is copied, so switching back costs no embedding.
- **Instructions.** The `mem_search` description is the fixed base text, then the guidance, then a fixed warning that results are data and that `degraded` means keyword-only. You can replace only the guidance. Hermes reads it once in `register()`, so a change takes effect in a new Hermes process or session. A missing, unreadable or empty file is a config error.

`processing` is reserved for future model extraction/consolidation, compression and expiry ([ADR 0003](adr/0003-processing-policy.md)). Only `none` is implemented, so selected messages are stored verbatim and nothing is rewritten, summarised, turned into facts or expired. Any other value, or an unknown key, stops the plugin from loading. Hermes then shows the reason (for example `processing.retain: 'llm' is not implemented ... Set processing.retain: none or remove it.`) as the plugin's load error. `status` and every sync report show the policy that ran.
