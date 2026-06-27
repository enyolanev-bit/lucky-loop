from __future__ import annotations
import json, shlex, subprocess, sys, time
from pathlib import Path
from .schemas import ActualResult


def tail(s: str, n: int = 2000) -> str:
    return s[-n:] if s else ""


def normalize_command(command: str) -> str:
    parts = shlex.split(command)
    if parts and parts[0] == "python":
        parts[0] = sys.executable
        return shlex.join(parts)
    return command


def execute(command: str, cwd: Path, timeout: int = 120) -> ActualResult:
    t0 = time.perf_counter()
    proc = subprocess.run(normalize_command(command), shell=True, cwd=str(cwd), text=True, capture_output=True, timeout=timeout)
    runtime = time.perf_counter() - t0
    stdout, stderr = proc.stdout, proc.stderr
    raw = {}
    status = "success" if proc.returncode == 0 else "failed"
    try:
        raw = json.loads(stdout[stdout.find("{"):])
    except Exception:
        raw = {"returncode": proc.returncode}
    return ActualResult(
        status=raw.get("status", status),
        accuracy=raw.get("accuracy"),
        f1=raw.get("f1"),
        runtime_seconds=raw.get("runtime_seconds", round(runtime, 4)),
        stdout_tail=tail(stdout),
        stderr_tail=tail(stderr),
        raw=raw,
    )
