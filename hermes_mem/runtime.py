"""The external embedding engine: a separately owned venv and a resident worker subprocess.

Hermes' interpreter never imports numpy/ONNX/fastembed. `hermes hermes-mem setup` provisions
`<data_dir>/runtime/engine-<lock hash>` from engine/requirements.lock with the host's base
interpreter (not the host venv), so replacing the Hermes venv leaves it intact. The worker
speaks one JSON object per line on stdin/stdout, loads each model once, and exits by itself
after `engine.idle_seconds` without requests. Discovery and status never provision anything.
"""

from __future__ import annotations

import hashlib
import json
import os
import selectors
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = PLUGIN_ROOT / "engine"
ENGINE_SCRIPT = ENGINE_DIR / "hermes_mem_engine.py"
LOCK_FILE = ENGINE_DIR / "requirements.lock"
PROTOCOL = 1


class RuntimeMissing(RuntimeError):
    """No usable engine runtime; the message says how to provision one."""


def lock_hash() -> str:
    return hashlib.sha256(LOCK_FILE.read_bytes()).hexdigest()[:16]


@dataclass
class EngineRuntime:
    data_dir: Path
    command: list[str] | None = None  # operator-provided argv; replaces the provisioned runtime
    idle_seconds: float = 300.0
    timeout: float = 120.0
    threads: int = 1
    cache_dir: str | None = None

    @property
    def root(self) -> Path:
        return self.data_dir / "runtime"

    @property
    def venv(self) -> Path:
        return self.root / f"engine-{lock_hash()}"

    def argv(self) -> list[str]:
        if self.command:
            return list(self.command)
        if not LOCK_FILE.exists():
            raise RuntimeMissing(f"this plugin copy has no {LOCK_FILE}; reinstall the complete plugin (engine/ included)")
        python = self.venv / "bin" / "python"
        if not (self.venv / ".ready").exists() or not python.exists():
            raise RuntimeMissing(
                f"embedding engine runtime is not provisioned at {self.venv}; run `hermes hermes-mem setup` "
                "(semantic search is degraded to keyword until then)"
            )
        return [str(python), "-I", str(ENGINE_SCRIPT)]

    def diagnose(self) -> dict:
        """Side-effect free: never creates, downloads or starts anything."""
        if not LOCK_FILE.exists():
            return {"lock": str(LOCK_FILE), "state": "broken", "command_override": bool(self.command),
                    "detail": f"this plugin copy has no {LOCK_FILE.name}; reinstall the complete plugin (engine/ included)"}
        out: dict = {"lock": str(LOCK_FILE), "lock_hash": lock_hash(), "command_override": bool(self.command)}
        try:
            argv = self.argv()
        except RuntimeMissing as exc:
            out.update(state="missing", detail=str(exc))
            stale = sorted(p.name for p in self.root.glob("engine-*") if p != self.venv) if self.root.exists() else []
            if stale:
                out["other_runtimes"] = stale
            return out
        executable = Path(argv[0]) if os.sep in argv[0] else shutil.which(argv[0])
        if not executable or not Path(executable).exists():
            out.update(state="broken", detail=f"engine executable {argv[0]!r} does not exist")
            return out
        out.update(state="provisioned", executable=str(executable))
        if not self.command:
            out["runtime"] = json.loads((self.venv / ".ready").read_text() or "{}")
        return out


def base_python() -> str:
    """The interpreter a venv should be built from: the real base, never the host's venv shim."""
    return os.path.realpath(getattr(sys, "_base_executable", None) or sys.executable)


def setup_runtime(rt: EngineRuntime, *, python: str | None = None, uv: str | None = None, force: bool = False) -> dict:
    """Provision the locked engine runtime. Idempotent; a failure leaves any existing runtime untouched."""
    target = rt.venv
    if (target / ".ready").exists() and not force:
        return {"status": "unchanged", "runtime": str(target), "lock_hash": lock_hash()}
    python = python or base_python()
    rt.root.mkdir(parents=True, exist_ok=True)
    staging = rt.root / f".staging-{os.getpid()}-{int(time.time())}"
    uv = uv if uv is not None else shutil.which("uv")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
    try:
        if uv:
            steps = [
                [uv, "venv", "-q", "--python", python, str(staging)],
                [uv, "pip", "install", "-q", "--require-hashes", "--python", str(staging / "bin/python"), "-r", str(LOCK_FILE)],
            ]
        else:
            steps = [
                [python, "-m", "venv", str(staging)],
                [str(staging / "bin/python"), "-m", "pip", "install", "-q", "--require-hashes", "-r", str(LOCK_FILE)],
            ]
        for step in steps:
            done = subprocess.run(step, env=env, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"{Path(step[0]).name} {step[1]} failed ({done.returncode}): {done.stderr.strip()[-800:]}")
        probe = subprocess.run(
            [str(staging / "bin/python"), "-I", str(ENGINE_SCRIPT), "--probe"], env=env, capture_output=True, text=True
        )
        if probe.returncode != 0:
            raise RuntimeError(f"engine probe failed: {probe.stderr.strip()[-800:]}")
        info = json.loads(probe.stdout)
        info.update(lock_hash=lock_hash(), built_from=python, installer="uv" if uv else "pip", built_at=time.time())
        (staging / ".ready").write_text(json.dumps(info))
        if target.exists():
            shutil.rmtree(target)  # only reached with force=True, after the replacement succeeded
        os.replace(staging, target)
    except BaseException as exc:
        shutil.rmtree(staging, ignore_errors=True)
        if isinstance(exc, Exception):
            return {"status": "failed", "error": str(exc), "runtime": str(target),
                    "kept_existing": (target / ".ready").exists()}
        raise
    return {"status": "provisioned", "runtime": str(target), **info}


class EngineClient:
    """One resident worker per Hermes process, started on first use and restarted after a crash."""

    def __init__(self, rt: EngineRuntime):
        self.rt = rt
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def _start(self) -> subprocess.Popen:
        argv = self.rt.argv()
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        env["HERMES_MEM_ENGINE_IDLE"] = str(self.rt.idle_seconds)
        env["HERMES_MEM_ENGINE_THREADS"] = str(self.rt.threads)
        if self.rt.cache_dir:
            env["HERMES_MEM_ENGINE_CACHE"] = self.rt.cache_dir
        try:
            return subprocess.Popen(
                argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                env=env, text=True, bufsize=1, cwd=str(self.rt.data_dir) if self.rt.data_dir.exists() else None,
            )
        except OSError as exc:
            raise RuntimeMissing(f"cannot start embedding engine {argv[0]!r}: {exc}") from exc

    def call(self, request: dict) -> dict:
        from .embed import EmbeddingUnavailable

        with self._lock:
            for attempt in (1, 2):
                if self._proc is None or self._proc.poll() is not None:
                    try:
                        self._proc = self._start()
                    except RuntimeMissing as exc:
                        raise EmbeddingUnavailable(str(exc)) from None
                try:
                    reply = self._roundtrip({"protocol": PROTOCOL, **request})
                except (BrokenPipeError, EOFError) as exc:
                    self.close()
                    if attempt == 2:
                        raise EmbeddingUnavailable(f"embedding engine exited: {exc}") from None
                    continue  # idle exit or crash: restart once
                except TimeoutError:
                    self.close()
                    raise EmbeddingUnavailable(f"embedding engine timed out after {self.rt.timeout:.0f}s") from None
                if "error" in reply:
                    raise EmbeddingUnavailable(f"embedding engine: {reply['error']}")
                return reply
        raise AssertionError("unreachable")

    def _roundtrip(self, request: dict) -> dict:
        proc = self._proc
        assert proc and proc.stdin and proc.stdout
        proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        proc.stdin.flush()
        with selectors.DefaultSelector() as sel:
            sel.register(proc.stdout, selectors.EVENT_READ)
            if not sel.select(self.rt.timeout):
                raise TimeoutError
        line = proc.stdout.readline()
        if not line:
            raise EOFError("no reply")
        try:
            return json.loads(line)
        except ValueError:
            raise EOFError("malformed reply") from None

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.stdin.close()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
            proc.wait()
