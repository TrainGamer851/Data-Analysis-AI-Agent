"""Worker that runs the LLM-generated pandas code.

It is started by execution_env.py as:
    - a local `python -I` subprocess 
    Or - inside the Docker container. 
    
(!) The app never imports it. 
    Idea of this Protocol: 
        read one pickled payload from stdin,
        -> write exactly one JSON line to stdout.

It must stay self-contained (standard library + pandas + numpy only): 
    inside the container there is no other project code.
"""

from __future__ import annotations

import builtins
import contextlib
import io
import json
import os
import pickle
import sys
import sysconfig
import traceback

import numpy as np
import pandas as pd

try:  # For Unix
    import resource
except ImportError:
    resource = None

LLM_FILENAME = "<llm_generated_code>" #Fake file name

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