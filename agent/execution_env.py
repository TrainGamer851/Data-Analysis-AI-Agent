"""Run LLM-generated pandas code in an isolated worker (sandbox_worker.py).

Now become the core of a sandbox.

execute_code() sends the code and the DataFrame 
    to a worker process and returns the answer as a dict.
    
The SANDBOX_BACKEND environment variable picks where the worker runs:

    local (default) The previous version that implemented
        using purely subprocess isolation and restrictions:
            a fresh `python -I` subprocess with an empty environment.
            Defense in depth, NOT a true sandbox. (Using mainly for testing development.)
    docker (new)
        Same worker in an additional layer - locked-down container:
            no network, read-only filesystem, non-root, dropped capabilities, memory/CPU/process limits.
        The image is built automatically on first use.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import subprocess
import sys
import tempfile

import pandas as pd

STARTUP_GRACE = 10  # extra seconds for interpreter + pandas startup
MAX_ROWS = 200

_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_worker.py")

# default return
def _fail(message: str) -> dict:
    return {
        "success": False,
        "result": None,
        "result_type": None,
        "stdout": "",
        "error": message
        }

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

    Returns{
        success,
        result,
        result_type ("table"|"scalar"|"object"),
        stdout,
        error
        }.
    DataFrames and Series come back as a list of row dicts.
    """
    code = clean_code(code_string)
    if not code:
        return _fail("Execution Error: no code was provided.")
    if df is None or df.empty:
        return _fail("Execution Error: the dataset DataFrame is empty or None.")

    payload = pickle.dumps({
        "code": code,
        "df": df,
        "memory_mb": memory_mb,
        "max_rows": max_rows
        })

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