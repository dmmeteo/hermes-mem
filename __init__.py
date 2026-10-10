"""Hermes plugin entry for hermes-mem (standalone kind: explicit tools, CLI and post-turn sync hooks)."""

from __future__ import annotations

from .hermes_mem import cli
from .hermes_mem.app import MemApp
from .hermes_mem.sync import BackgroundSync
from .hermes_mem.tools import SEARCH_SCHEMA, TOOLS, search_schema

TOOLSET = "hermes_mem"


def _bind(app: MemApp, handler):
    def call(args: dict, session_id: str | None = None, **_kwargs) -> str:
        return handler(app, args or {}, session_id)

    call.__name__ = handler.__name__
    return call


def register(ctx) -> None:
    # Side-effect free: no runtime provisioning, model load or engine start happens here.
    # Guidance is read once, so the registered schema stays stable for this process.
    app = MemApp()
    for schema, handler in TOOLS:
        if schema is SEARCH_SCHEMA:
            schema = search_schema(app.config["search"]["guidance"])
        ctx.register_tool(
            name=schema["name"],
            toolset=TOOLSET,
            schema=schema,
            handler=_bind(app, handler),
            description=schema["description"],
        )
    ctx.register_cli_command(
        "hermes-mem",
        "Episode memory: sync, search, status and curated statements",
        cli.setup,
        cli.run,
    )
    if app.config["sync"]["background"]:
        worker = BackgroundSync(app.open_store, app.state_db, app.policy, app.embedder, app.data_dir / "sync.lock")
        app.background = worker

        def after_turn(session_id: str | None = None, **_kwargs) -> None:
            worker.request(session_id)

        ctx.register_hook("post_llm_call", after_turn)
        ctx.register_hook("on_session_end", after_turn)
