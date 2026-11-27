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
import math
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

ALLOWED_IMPORTS = {"pandas", "numpy", "math", "datetime", "re"} # We can add more to expand the usage if change, also change in prompts.py
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
    if "pyarrow" in sys.modules:
        return
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

def _count_inversions(values):
    """Count strict inversions using merge sort."""

    size = len(values)

    if size <= 1:
        return 0

    middle = size // 2

    left = values[:middle]
    right = values[middle:]

    count = (
        _count_inversions(left)
        + _count_inversions(right)
    )

    i = 0
    j = 0
    k = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            values[k] = left[i]
            i += 1
        else:
            values[k] = right[j]
            j += 1

            count += len(left) - i

        k += 1

    while i < len(left):
        values[k] = left[i]
        i += 1
        k += 1

    while j < len(right):
        values[k] = right[j]
        j += 1
        k += 1

    return count


def _count_tied_pairs(values):
    """Count pairs sharing an identical value."""

    counts = {}

    for value in values:
        counts[value] = counts.get(value, 0) + 1

    return sum(
        count * (count - 1) // 2
        for count in counts.values()
    )


def _kendall_tau_b(x, y):
    """Calculate Kendall tau-b with tie correction."""

    n = len(x)

    if n < 2:
        return float("nan")

    total_pairs = n * (n - 1) // 2

    ties_x = _count_tied_pairs(x)
    ties_y = _count_tied_pairs(y)

    pairs = sorted(zip(x, y))

    ties_both = _count_tied_pairs(pairs)

    sorted_y = [
        pair[1]
        for pair in pairs
    ]

    discordant = _count_inversions(sorted_y)

    comparable_pairs = (
        total_pairs
        - ties_x
        - ties_y
        + ties_both
    )

    numerator = (
        comparable_pairs
        - 2 * discordant
    )

    denominator = math.sqrt(
        (total_pairs - ties_x)
        * (total_pairs - ties_y)
    )

    if denominator == 0:
        return float("nan")

    return numerator / denominator


def safe_correlation(
    left,
    right,
    method="pearson",
):
    """Calculate correlation without requiring SciPy."""

    if not isinstance(method, str):
        raise TypeError(
            "Correlation method must be a string."
        )

    method = method.strip().lower()

    if method not in {
        "pearson",
        "spearman",
        "kendall",
    }:
        raise ValueError(
            "Supported correlation methods: "
            "pearson, spearman, kendall."
        )

    if not isinstance(left, pd.Series):
        raise TypeError(
            "Left correlation input must be a pandas Series."
        )

    if not isinstance(right, pd.Series):
        raise TypeError(
            "Right correlation input must be a pandas Series."
        )

    if not left.index.is_unique or not right.index.is_unique:
        raise ValueError(
            "Correlation inputs require unique indexes."
        )

    left, right = left.align(
        right,
        join="inner",
    )

    left = pd.to_numeric(
        left,
        errors="raise",
    )

    right = pd.to_numeric(
        right,
        errors="raise",
    )

    if (
        pd.api.types.is_complex_dtype(left.dtype)
        or pd.api.types.is_complex_dtype(right.dtype)
    ):
        raise ValueError(
            "Complex correlation inputs are unsupported."
        )

    x = left.to_numpy(
        dtype=float,
        na_value=np.nan,
    )

    y = right.to_numpy(
        dtype=float,
        na_value=np.nan,
    )

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return float("nan")

    if (
        np.all(x == x[0])
        or np.all(y == y[0])
    ):
        return float("nan")

    if method == "pearson":
        return float(
            pd.Series(x).corr(
                pd.Series(y),
                method="pearson",
            )
        )

    if method == "spearman":
        ranked_x = pd.Series(x).rank(
            method="average"
        )

        ranked_y = pd.Series(y).rank(
            method="average"
        )

        return float(
            ranked_x.corr(
                ranked_y,
                method="pearson",
            )
        )

    return float(
        _kendall_tau_b(
            x.tolist(),
            y.tolist(),
        )
    )

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
        "df": df,
        "safe_correlation": safe_correlation,
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
        payload = pickle.load(sys.stdin.buffer)
        code, df = payload["code"], payload["df"]

        # Preserve the Windows system directory required by SSL.
        system_root = os.environ.get("SYSTEMROOT") if os.name == "nt" else None

        # Remove environment variables to prevent credential exposure.
        os.environ.clear()

        if system_root:
            os.environ["SYSTEMROOT"] = system_root

        _limit_memory(payload["memory_mb"])
        _install_audit_hook()
        outcome = _run(code, df, stdout, payload["max_rows"])

    except BaseException as exc:
        outcome = _fail(
            _format_error(exc, code, df),
            stdout.getvalue()[:MAX_STDOUT],
        )

    out.write(json.dumps(outcome, default=str).encode("utf-8") + b"\n")
    out.flush()

if __name__ == "__main__":
    _worker()