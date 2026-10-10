# Development

Coding agents and contributors start at [AGENTS.md](../AGENTS.md). Every change must pass:

```bash
HERMES_HOST_SOURCE=/path/to/hermes-agent HOST_PYTHON=3.13 bash scripts/verify.sh
```

Scratch goes to `HERMES_MEM_WORK`, or `$TMPDIR/hermes-mem-verify` if that is unset. Models are cached in `$HERMES_MEM_CACHE_HOME/hermes-mem/models`, by default the plugin's own shared cache. The gates are:

- **lint and unit tests**, on Python 3.14 and on the host Python.
- **`engine-setup`**: the real `setup` command.
- **`scripts/host_integration.py`**: real plugin discovery in a temporary `HERMES_HOME`, `hermes plugins list`, the `hermes hermes-mem` CLI, tool dispatch, the post-turn hook, compaction/rewind/delete written through Hermes' own `SessionDB`, and the curated propose/accept round trip persisted in `mem.db`.
- **`scripts/eval_paraphrase.py`**: the real-embedding Ukrainian/English paraphrase eval on the default profile.
- **`scripts/compare_models.py`**: BGE-M3 INT8 and MiniLM, one after the other, on the same corpus and chunks. It covers the original smoke set plus Ukrainian, English and transliterated pricing-policy questions and distractors. It reports per-query ranks, hit@1/5 and MRR for the literal query against the frozen variants, follow-ups with and without context, cold and warm latency, vector size and coverage, the exact space and loaded snapshot, and dataset and source provenance. Its gates check that both models really ran; quality is reported, not gated.
- **`scripts/install_acceptance.py`**: the documented install → `setup` → curated `import` → `index` → tool path on an installed copy, with the default profile and shared cache, run from an unrelated directory. It covers repeat and failed setup, a blocked and a repeated Markdown import, `mem_search` and `mem_curated` dispatch, a second profile that shares the model cache but not the runtime or store, deleting and rebuilding the host venv, and a `--force` reinstall. It checks that host packages are unchanged, that no heavy module enters the Hermes process, and that the SQLite state persists.

To check a candidate Hermes version without touching any live install, run the same script with other selectors:

```bash
python3 scripts/install_acceptance.py --host-source /path/to/candidate-hermes --host-python 3.12 --work /scratch/compat
```

The first run downloads wheels and both models (about 0.8 GB).
