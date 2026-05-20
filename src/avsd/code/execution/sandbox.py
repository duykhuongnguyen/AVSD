import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence


_BWRAP_ENV = "CODE_AVSD_BWRAP"
_BWRAP_FALLBACKS = ("/usr/bin/bwrap", "/bin/bwrap")


def check_sandbox_available(raise_on_error: bool = False) -> bool:
    bwrap_path = _find_bwrap()
    if bwrap_path is None and raise_on_error:
        override = os.environ.get(_BWRAP_ENV)
        override_text = f"{_BWRAP_ENV}={override!r}, " if override else ""
        raise RuntimeError(
            "bubblewrap (bwrap) is required for the default code sandbox. "
            f"Could not find an executable bwrap ({override_text}PATH={os.environ.get('PATH', '')!r}; "
            f"also checked {', '.join(_BWRAP_FALLBACKS)}). "
            f"Set {_BWRAP_ENV}=/path/to/bwrap if your launcher hides it from PATH. "
            "Use --unsafe-subprocess-sandbox only for local tests."
        )
    return bwrap_path is not None


def run_python_in_sandbox(
    code: str,
    stdin: str,
    *,
    timeout_s: float = 2.0,
    max_output_chars: int = 4000,
    memory_limit_mb: int = 512,
    unsafe_subprocess_sandbox: bool = False,
) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="avsd_code_") as tmp:
        tmp_path = Path(tmp)
        program = tmp_path / "main.py"
        program.write_text(code, encoding="utf-8")
        cmd = _unsafe_cmd(program) if unsafe_subprocess_sandbox else _bwrap_cmd(tmp_path)
        proc = subprocess.run(
            cmd,
            input=stdin,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            cwd=str(tmp_path),
            start_new_session=True,
            preexec_fn=_limit_resources(memory_limit_mb),
        )
        proc.stdout = proc.stdout[:max_output_chars]
        proc.stderr = proc.stderr[:max_output_chars]
        return proc


def _unsafe_cmd(program: Path) -> list[str]:
    return [sys.executable, str(program)]


def _bwrap_cmd(tmp_path: Path) -> list[str]:
    bwrap_path = _find_bwrap()
    if bwrap_path is None:
        check_sandbox_available(raise_on_error=True)
        raise AssertionError("unreachable")
    return [
        bwrap_path,
        "--die-with-parent",
        "--new-session",
        "--unshare-net",
        "--ro-bind",
        "/",
        "/",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/tmp/work",
        "--bind",
        str(tmp_path),
        "/tmp/work",
        "--chdir",
        "/tmp/work",
        "--setenv",
        "PYTHONIOENCODING",
        "utf-8",
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        sys.executable,
        "/tmp/work/main.py",
    ]


def _find_bwrap() -> str | None:
    override = os.environ.get(_BWRAP_ENV)
    if override:
        return override if _is_executable(override) else None
    from_path = shutil.which("bwrap")
    if from_path:
        return from_path
    for candidate in _BWRAP_FALLBACKS:
        if _is_executable(candidate):
            return candidate
    return None


def _is_executable(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK)


def _limit_resources(memory_limit_mb: int):
    if os.name != "posix":
        return None

    def set_limits() -> None:
        try:
            import resource

            memory_bytes = int(memory_limit_mb * 1024 * 1024)
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            cpu_seconds = 30
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        except Exception:
            pass

    return set_limits


def kill_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
