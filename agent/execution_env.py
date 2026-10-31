"""Run LLM-generated pandas code in a restricted subprocess.

Defense in depth, NOT a true sandbox: a fresh `python -I` interpreter with an
empty environment, a runtime audit hook (no network, subprocesses, writes, or
reads outside the Python install), a memory cap (Linux), restricted imports and
builtins, and a wall-clock timeout. Before exposing this to untrusted users,
run the worker inside Docker (--network none, read-only FS, non-root).

The file is also its own worker: the parent launches `python -I execution_env.py`,
sends a pickled payload on stdin, and reads one JSON line from stdout.
"""

from __future__ import annotations

import builtins
import contextlib
import io
import json
import os
import pickle
import re
import subprocess
import sys
import sysconfig
import tempfile
import traceback

import numpy as np
import pandas as pd

try:  # For Unix
    import resource
except ImportError:
    resource = None

LLM_FILENAME = "<llm_generated_code>" #Fake file name
STARTUP_GRACE = 10  # extra seconds for interpreter + pandas startup

MAX_ROWS = 200
MAX_STDOUT = 10_000

ALLOWED_IMPORTS = {"pandas", "numpy", "math", "datetime", "re"} # We can add more to expand the usage
SAFE_BUILTINS = (
    "abs all any bool dict enumerate filter float int isinstance len list map "
    "max min print range reversed round set sorted str sum tuple zip "
    "Exception ValueError KeyError TypeError IndexError ZeroDivisionError"
).split()
BLOCKED_EVENTS = (
    "socket.", "subprocess.", "os.system", "os.exec", "os.spawn",
    "os.posix_spawn", "os.fork", "os.forkpty", "os.kill", "os.remove",
    "os.rename", "os.rmdir", "os.mkdir", "os.chmod", "os.chown", "os.link",
    "os.symlink", "os.truncate", "os.putenv", "os.unsetenv", "os.chdir",
    "shutil.", "ctypes.dlopen", "ctypes.call_function", "pty.",
    "urllib.Request", "http.client.connect", "smtplib.", "ftplib.",
    "telnetlib.", "webbrowser.", "sqlite3.connect", "winreg.", "tempfile.mk",
) # no-no commands

# default return
def _fail(message: str, stdout: str = "") -> dict:
    return {"success": False, "result": None, "result_type": None,
            "stdout": stdout, "error": message}

# Parent side (App part)

# Clean the LLM output
def clean_code(text: str) -> str:
    """Pull plain Python out of an LLM reply that may contain markdown fences."""
    fenced = re.search(r"```(?:python|py)?[ \t]*\n(.*?)```", text or "", re.DOTALL)
    text = fenced.group(1) if fenced else (text or "")
    return text.replace("```python", "").replace("```", "").strip()

# Set up the subprocess env to run the LLM code
def execute_code(code_string: str, df: pd.DataFrame, timeout: int = 10,
                 memory_mb: int = 1024, max_rows: int = MAX_ROWS) -> dict:
    """Run code with `df`, `pd`, `np` available; the code must set `result`.

    Returns {success, result, result_type ("table"|"scalar"|"object"), stdout, error}.
    DataFrames and Series come back as a list of row dicts.
    """
    code = clean_code(code_string)
    if not code:
        return _fail("Execution Error: no code was provided.")
    if df is None or df.empty:
        return _fail("Execution Error: the dataset DataFrame is empty or None.")

    payload = pickle.dumps({"code": code, "df": df, "memory_mb": memory_mb, "max_rows": max_rows})

    # Set the child environment 
    env = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1"
        }
    if os.name == "nt":  # Python on Windows
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")

    with tempfile.TemporaryDirectory() as workdir: #Make a "fake" ttemp cwd to run
        try:
            proc = subprocess.run(
                [sys.executable, "-I", os.path.abspath(__file__)],
                input=payload, capture_output=True, env=env, cwd=workdir,
                timeout=timeout + STARTUP_GRACE,
            )
        except subprocess.TimeoutExpired:
            return _fail(f"Execution Error: code timed out after {timeout} seconds "
                         "(possible infinite loop). Use vectorized pandas operations.")

    try:
        return json.loads(proc.stdout.decode("utf-8", "replace").strip().splitlines()[-1])
    except (IndexError, ValueError):
        stderr = proc.stderr.decode("utf-8", "replace").strip()[-300:]
        return _fail(f"Execution Error: worker crashed (exit code {proc.returncode}). {stderr}")

# Worker side (runs in the child interpreter that we previously open from parent)
def _limit_memory(memory_mb: int) -> None:
    """Cap address space at current usage + memory_mb. Linux only; skipped elsewhere."""
    try:
        with open("/proc/self/statm") as fh:
            current = int(fh.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
        limit = current + memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (OSError, ValueError, AttributeError):
        pass

def _install_audit_hook() -> None:
    """Check the restricted file commands, access control (write/read/execute)."""
    roots = tuple(
        os.path.normcase(os.path.realpath(p))
        for p in {
            sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix,
            *sysconfig.get_paths().values(),
            "/usr/lib", "/usr/lib64", "/lib", "/lib64", "/usr/share/zoneinfo",
            "/etc/localtime", "/dev/urandom", "/dev/null",
        }
        if p and os.path.exists(p)
    )
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND

    def hook(event: str, args: tuple) -> None:
        if event == "open":
            path, mode, flags = args
            if isinstance(path, int):
                return
            if (isinstance(flags, int) and flags & write_flags) or (
                isinstance(mode, str) and any(c in mode for c in "wax+")
                ):
                raise PermissionError("Blocked by sandbox: writing files")
            real = os.path.normcase(os.path.realpath(os.fsdecode(path)))
            if not any(real == r or real.startswith(r + os.sep) for r in roots):
                raise PermissionError(f"Blocked by sandbox: reading {path!r}")
        elif event.startswith(BLOCKED_EVENTS):
            raise PermissionError(f"Blocked by sandbox: {event}")

    sys.addaudithook(hook)  # cannot be removed once installed

def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Check the allowed import libs."""
    if level != 0 or name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(f"Import of '{name}' is not allowed. pd, np and df are already "
                          f"available. Allowed modules: {', '.join(sorted(ALLOWED_IMPORTS))}.")
    return builtins.__import__(name, globals, locals, fromlist, level)

def _json_default(obj):
    """Handle unknown type."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (np.ndarray, pd.Index, pd.api.extensions.ExtensionArray)):
        return obj.tolist()  # .unique() returns an ExtensionArray in recent pandas
    return str(obj)  # Timestamp, Decimal, set, ...

def _serialize(value, max_rows: int):
    """Return (json_safe_value, result_type)."""
    if isinstance(value, pd.Series):
        value = value.to_frame(name=value.name if value.name is not None else "value")
    if isinstance(value, pd.DataFrame):
        if not isinstance(value.index, pd.RangeIndex):  # keep groupby labels
            try:
                value = value.reset_index()
            except ValueError:
                value = value.reset_index(drop=True)
        records = value.head(max_rows).to_json(orient="records", date_format="iso", default_handler=str)
        return json.loads(records), "table"

    safe = json.loads(json.dumps(value, default=_json_default))
    if isinstance(safe, float) and safe != safe:
        safe = None
    scalar = safe is None or isinstance(safe, (bool, int, float, str))
    return safe, "scalar" if scalar else "object"

def _format_error(exc: BaseException, code: str, df) -> str:
    """Error raise formatting for feedback: type, message, offending line, columns."""
    lineno = exc.lineno if isinstance(exc, SyntaxError) and exc.filename == LLM_FILENAME else None
    for frame in traceback.extract_tb(exc.__traceback__):
        if frame.filename == LLM_FILENAME:
            lineno = frame.lineno
    lines = code.splitlines()
    message = f"{type(exc).__name__}: {exc}"
    if lineno and 1 <= lineno <= len(lines):
        message += f"\n  at line {lineno}: {lines[lineno - 1].strip()}"
    if isinstance(exc, (KeyError, AttributeError)) and df is not None:
        message += f"\nAvailable columns: {list(df.columns)}"
    return message

def _run(code: str, df: pd.DataFrame, stdout: io.StringIO, max_rows: int) -> dict:
    """Execute code with all restrictions applied."""
    builtins_ = {
        name: getattr(builtins, name) for name in SAFE_BUILTINS
        }
    builtins_["__import__"] = _restricted_import
    # One namespace (not separate globals/locals) so comprehensions and defs see top-level names.
    env = {
        "__builtins__": builtins_,
        "__name__": "__sandbox__",
        "pd": pd,
        "np": np,
        "df": df
        }

    with contextlib.redirect_stdout(stdout):
        exec(compile(code, LLM_FILENAME, "exec"), env)

    if "result" not in env:
        raise NameError("the code ran but did not define 'result'. Assign your final answer to `result`.")
    result, kind = _serialize(env["result"], max_rows)
    return {"success": True, "result": result, "result_type": kind,
            "stdout": stdout.getvalue()[:MAX_STDOUT], "error": None}

# Worker entry point!
def _worker() -> None:
    out = sys.stdout.buffer
    stdout, code, df = io.StringIO(), "", None
    try:
        payload = pickle.load(sys.stdin.buffer)  # the payload that was written by parent
        code, df = payload["code"], payload["df"] # Extract payload now!!!
        os.environ.clear() # clean the env that we created for test run
        _limit_memory(payload["memory_mb"])
        _install_audit_hook()  # last step before running LLM code -> Lock all dangerous commands :)
        outcome = _run(code, df, stdout, payload["max_rows"])
    except BaseException as exc:  # includes MemoryError and SystemExit
        outcome = _fail(_format_error(exc, code, df), stdout.getvalue()[:MAX_STDOUT])
    out.write(json.dumps(outcome, default=str).encode("utf-8") + b"\n")
    out.flush()

if __name__ == "__main__":
    _worker()