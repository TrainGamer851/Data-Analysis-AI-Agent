"""Run LLM-generated pandas code in an isolated worker (sandbox_worker.py).

Now become the core of a sandbox.

execute_code() sends the code and the DataFrame 
    to a worker process and returns the answer as a dict.
    
The SANDBOX_BACKEND environment variable picks where the worker runs:

    local (default) The previous version that implemented
        using purely subprocess isolation and restrictions:
            a fresh `python -I` subprocess with an empty environment.
            Defense in depth, NOT a true sandbox. (Using mainly for testing development.)
    docker (new - required installation of Docker Desktop)
        Same worker in an additional layer - locked-down container:
            no network, read-only filesystem, non-root, dropped capabilities, memory/CPU/process limits.
        The image is built automatically on first use.
"""

from __future__ import annotations

import contextlib
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
import uuid
from importlib.metadata import version

import pandas as pd

STARTUP_GRACE = 10  # extra seconds for interpreter + pandas startup
MAX_ROWS = 200

IMAGE = "data-agent-sandbox"
 
_CURR = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_CURR, "sandbox_worker.py")
_DOCKERFILE = os.path.join(os.path.dirname(_CURR), "docker", "Dockerfile")
_image_ready = False  # checked once per process

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

# Docker helpers
def docker_available() -> bool:
    """True <- the Docker client is installed and the daemon is running."""
    try:
        r = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"],
                           capture_output=True, timeout=15)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False
 
def prepare_sandbox() -> str | None:
    """Check sandbox image exists and matches the installed pandas.
 
    Builds it if needed (very slow for first build).
    Returns an error message, or None when ready.
    execute_code() calls it automatically;
        (intended to) call it at app startup to avoid a slow first question.
    """
    global _image_ready
    if _image_ready:
        return None
    try:
        # If our image exists
        # Read "pandas" label stored in the image
        r = subprocess.run(["docker", "image", "inspect", "-f", "{{.Config.Labels.pandas}}", IMAGE],
                           capture_output=True, timeout=30)
        if r.returncode == 0 and r.stdout.decode().strip() == pd.__version__:
            _image_ready = True
            return None
        # Missing/wrong pandas version: re-build to correct
        with tempfile.TemporaryDirectory() as empty_context:  # open empty Docker
            build = subprocess.run(
                ["docker", "build", "-t", IMAGE, "-f", _DOCKERFILE,
                 "--build-arg", f"PANDAS_VERSION={pd.__version__}",
                 "--build-arg", f"NUMPY_VERSION={version('numpy')}", empty_context],
                capture_output=True, timeout=900)
    except OSError as exc:
        return f"Execution Error: could not run Docker. Is it installed and on PATH? ({exc})"
    except subprocess.TimeoutExpired:
        return "Execution Error: building the sandbox image timed out."
    if build.returncode != 0:
        tail = build.stderr.decode("utf-8", "replace").strip()[-400:]
        return f"Execution Error: could not build the sandbox image. Is Docker running? {tail}"
    _image_ready = True
    return None
 
def _docker_base(name: str, memory_mb: int) -> list[str]:
    """`docker run` flags for the locked-down container (image and command are appended)."""
    limit = f"{memory_mb + 512}m"  # headroom for the Python + pandas baseline
    return [
        "docker", "run", "--rm", "-i", "--name", name,
        "--network", "none", # stop network activity
        "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
        "--memory", limit, "--memory-swap", limit,  # fixed mem cap, no swap
        "--cpus", "1", "--pids-limit", "64",  # limited CPU and fork-bomb
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--user", "65534:65534",  # any, but not root
        "-e", "OMP_NUM_THREADS=1", "-e", "OPENBLAS_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1",
        # only worker script is shared with the host, in read-only
        "--mount", f"type=bind,source={_WORKER},target=/app/sandbox_worker.py,readonly",
    ]

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

    docker = os.environ.get("SANDBOX_BACKEND", "local").lower() == "docker"
    name = f"sandbox-{uuid.uuid4().hex[:12]}" # Tracking this docker env

    if docker:
        error = prepare_sandbox()
        if error:
            return _fail(error)
        cmd = _docker_base(name, memory_mb) + [IMAGE, "python", "-I", "/app/sandbox_worker.py"]
        env = None  # the docker client needs PATH etc.; the container only gets the -e flags
    else:
        cmd = [sys.executable, "-I", _WORKER]
        # Set the child environment 
        env = {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1"
            }
        if os.name == "nt":  # Python on Windows
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")

    payload = pickle.dumps({
        "code": code,
        "df": df,
        "memory_mb": memory_mb,
        "max_rows": max_rows
        })

    with tempfile.TemporaryDirectory() as workdir: #Make a "fake" ttemp cwd to run
        try:
            proc = subprocess.run(
                cmd,
                input=payload,
                capture_output=True,
                env=env,
                cwd=workdir,
                timeout=timeout + STARTUP_GRACE + (10 if docker else 0),
            )
        except subprocess.TimeoutExpired:
            if docker: #kill docker
                with contextlib.suppress(Exception):
                    subprocess.run(["docker", "kill", name],
                        capture_output=True,
                        timeout=10
                        )
            return _fail(f"Execution Error: code timed out after {timeout} seconds "
                         "(possible infinite loop). Use vectorized pandas operations.")
        except OSError as exc:
            return _fail(f"Execution Error: could not start the sandbox ({exc})")

    try:
        return json.loads(proc.stdout.decode("utf-8", "replace").strip().splitlines()[-1])
    except (IndexError, ValueError):
        stderr = proc.stderr.decode("utf-8", "replace").strip()[-400:]
        return _fail(f"Execution Error: worker crashed (exit code {proc.returncode}). {stderr}")