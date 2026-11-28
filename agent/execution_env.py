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
from agent.debug_logger import debug_log, DebugTimer

STARTUP_GRACE = 10  # extra seconds for interpreter + pandas startup
MAX_ROWS = 200  # change this please change also in prompts.py

IMAGE = "data-agent-sandbox"

_CURR = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_CURR, "sandbox_worker.py")
_DOCKERFILE = os.path.join(os.path.dirname(_CURR), "docker", "Dockerfile")
_image_ready = False  # checked once per process


# default return
def _fail(message: str) -> dict:
    debug_log(
        "SANDBOX ERROR",
        message,
    )

    return {
        "success": False,
        "result": None,
        "result_type": None,
        "stdout": "",
        "error": message,
    }


# Parent side (App part)


# Clean the LLM output Pull plain Python out of an LLM reply that may contain markdown fences.
def clean_code(text: str) -> str:
    fenced = re.search(r"```(?:python|py)?[ \t]*\n(.*?)```", text or "", re.DOTALL)
    text = fenced.group(1) if fenced else (text or "")
    return text.replace("```python", "").replace("```", "").strip()


# Docker helpers True when Docker is installed and its daemon is running.
def docker_available() -> bool:

    timer = DebugTimer("DOCKER")

    debug_log(
        "DOCKER",
        "Checking Docker daemon availability",
    )

    try:
        r = subprocess.run(
            [
                "docker",
                "version",
                "--format",
                "{{.Server.Version}}",
            ],
            capture_output=True,
            timeout=15,
        )

        available = r.returncode == 0

        debug_log(
            "DOCKER",
            f"Docker available: {available}",
        )

        return available

    except (OSError, subprocess.TimeoutExpired) as exc:
        debug_log(
            "DOCKER ERROR",
            f"Availability check failed: {type(exc).__name__}",
        )

        return False

    finally:
        timer.finish("Docker availability check finished")


# Check sandbox image exists and matches installed pandas.
def prepare_sandbox() -> str | None:

    global _image_ready

    timer = DebugTimer("DOCKER IMAGE")

    debug_log(
        "DOCKER IMAGE",
        f"Preparing sandbox image: {IMAGE}",
    )

    # Call `debug_log` when `_image_ready`.
    if _image_ready:
        debug_log(
            "DOCKER IMAGE",
            "Image already verified in this process",
        )

        return None

    try:
        debug_log(
            "DOCKER IMAGE",
            "Inspecting existing sandbox image",
        )

        debug_log(
            "DOCKER IMAGE",
            f"Required pandas version: {pd.__version__}",
        )

        r = subprocess.run(
            [
                "docker",
                "image",
                "inspect",
                "-f",
                "{{.Config.Labels.pandas}}",
                IMAGE,
            ],
            capture_output=True,
            timeout=30,
        )

        image_version = r.stdout.decode().strip()

        debug_log(
            "DOCKER IMAGE",
            f"Inspection exit code: {r.returncode}",
        )

        # Call `debug_log` when `r.returncode == 0`.
        if r.returncode == 0:
            debug_log(
                "DOCKER IMAGE",
                f"Installed image pandas version: {image_version}",
            )

        # Update `_image_ready` when `r.returncode == 0 and image_version == pd.__version__`.
        if r.returncode == 0 and image_version == pd.__version__:
            _image_ready = True

            timer.finish("Existing sandbox image verified")

            return None

        debug_log(
            "DOCKER IMAGE",
            "Image missing or version mismatch",
        )

        debug_log(
            "DOCKER IMAGE",
            "Starting sandbox image build",
        )

        with tempfile.TemporaryDirectory(
            dir=os.environ.get("ANALYSIS_TEMP_DIR")
        ) as empty_context:

            build_timer = DebugTimer("DOCKER BUILD")

            build = subprocess.run(
                [
                    "docker",
                    "build",
                    "-t",
                    IMAGE,
                    "-f",
                    _DOCKERFILE,
                    "--build-arg",
                    f"PANDAS_VERSION={pd.__version__}",
                    "--build-arg",
                    f"NUMPY_VERSION={version('numpy')}",
                    empty_context,
                ],
                capture_output=True,
                timeout=900,
            )

            build_timer.finish("Docker build command finished")

    except OSError as exc:
        debug_log(
            "DOCKER ERROR",
            "Unable to execute Docker command",
        )

        return (
            "Execution Error: could not run Docker. "
            f"Is it installed and on PATH? ({exc})"
        )

    except subprocess.TimeoutExpired:
        debug_log(
            "DOCKER ERROR",
            "Docker image preparation timed out",
        )

        return "Execution Error: building the sandbox image timed out."

    # Call `debug_log` when `build.returncode != 0`.
    if build.returncode != 0:
        debug_log(
            "DOCKER ERROR",
            f"Image build exit code: {build.returncode}",
        )

        tail = build.stderr.decode("utf-8", "replace").strip()[-400:]

        return (
            "Execution Error: could not build the sandbox image. "
            f"Is Docker running? {tail}"
        )

    _image_ready = True

    timer.finish("Sandbox image prepared successfully")

    return None


# docker run` flags for the locked-down container (image and command are appended).
def _docker_base(name: str, memory_mb: int) -> list[str]:
    limit = f"{memory_mb + 512}m"  # headroom for the Python + pandas baseline
    return [
        "docker",
        "run",
        "--rm",
        "-i",
        "--name",
        name,
        "--network",
        "none",  # stop network activity
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=16m",
        "--memory",
        limit,
        "--memory-swap",
        limit,  # fixed mem cap, no swap
        "--cpus",
        "1",
        "--pids-limit",
        "64",  # limited CPU and fork-bomb
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        "65534:65534",  # any, but not root
        "-e",
        "OMP_NUM_THREADS=1",
        "-e",
        "OPENBLAS_NUM_THREADS=1",
        "-e",
        "MKL_NUM_THREADS=1",
        # only worker script is shared with the host, in read-only
        "--mount",
        f"type=bind,source={_WORKER},target=/app/sandbox_worker.py,readonly",
    ]


# Set up the subprocess env to run the LLM code Run code with `df`, `pd`, `np` available; the code must set `result`.
def execute_code(
    code_string: str,
    df: pd.DataFrame,
    timeout: int = 10,
    memory_mb: int = 1024,
    max_rows: int = MAX_ROWS,
) -> dict:
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
    timer = DebugTimer("EXECUTION")

    debug_log(
        "EXECUTION",
        "Starting sandbox execution pipeline",
    )

    debug_log(
        "EXECUTION",
        "Cleaning generated Python code",
    )

    code = clean_code(code_string)

    debug_log(
        "EXECUTION",
        f"Cleaned code characters: {len(code)}",
    )

    # Return the selected result when `not code`.
    if not code:
        return _fail("Execution Error: no code was provided.")
    # Return the selected result when `df is None or df.empty`.
    if df is None or df.empty:
        return _fail("Execution Error: the dataset DataFrame is empty or None.")

    docker = os.environ.get("SANDBOX_BACKEND", "local").lower() == "docker"

    debug_log(
        "EXECUTION",
        f"Backend selected: {'docker' if docker else 'local'}",
    )

    debug_log(
        "EXECUTION",
        f"Dataset: {df.shape[0]:,} rows | " f"{df.shape[1]} columns",
    )

    debug_log(
        "EXECUTION",
        f"Worker memory limit: {memory_mb} MB",
    )

    debug_log(
        "EXECUTION",
        f"Maximum returned table rows: {max_rows}",
    )

    job_id = os.environ.get("ANALYSIS_JOB_ID", "")

    # Update `job_id` when `not re.fullmatch('[a-f0-9]{32}', job_id)`.
    if not re.fullmatch(r"[a-f0-9]{32}", job_id):
        job_id = ""

    # Update `name` when `job_id`.
    if job_id:
        name = f"sandbox-{job_id}-{uuid.uuid4().hex[:12]}"
    # Update `name` when no earlier condition matches.
    else:
        name = f"sandbox-{uuid.uuid4().hex[:12]}"

    # Call `debug_log` when `docker`.
    if docker:
        debug_log(
            "DOCKER",
            f"Preparing container: {name}",
        )

        error = prepare_sandbox()
        # Return the selected result when `error`.
        if error:
            return _fail(error)
        cmd = _docker_base(name, memory_mb)

        debug_log(
            "DOCKER",
            "Container security configuration prepared",
        )

        debug_log(
            "DOCKER",
            "Network disabled | Read-only filesystem | Non-root",
        )

        debug_log(
            "DOCKER",
            "CPU limit: 1 | Process limit: 64",
        )

        debug_log(
            "DOCKER",
            f"Container memory limit: {memory_mb + 512} MB",
        )

        # Update `cmd` when `job_id`.
        if job_id:
            cmd += ["--label", f"data-analysis-job={job_id}"]

        cmd += [IMAGE, "python", "-I", "/app/sandbox_worker.py"]
        env = None  # the docker client needs PATH etc.; the container only gets the -e flags
    # Update `cmd` when no earlier condition matches.
    else:
        cmd = [sys.executable, "-I", _WORKER]
        # Set the child environment
        env = {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        # Update `env['SYSTEMROOT']` when `os.name == 'nt'`.
        if os.name == "nt":  # Python on Windows
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")

    payload_timer = DebugTimer("PAYLOAD")

    debug_log(
        "PAYLOAD",
        "Serializing code and dataset for sandbox worker",
    )

    payload = pickle.dumps(
        {"code": code, "df": df, "memory_mb": memory_mb, "max_rows": max_rows}
    )

    debug_log(
        "PAYLOAD",
        f"Serialized payload size: " f"{len(payload) / 1024:,.2f} KB",
    )

    payload_timer.finish("Payload serialization completed")

    effective_timeout = timeout + STARTUP_GRACE + (10 if docker else 0)

    debug_log(
        "EXECUTION",
        f"Code timeout: {timeout}s | " f"Effective wall timeout: {effective_timeout}s",
    )

    with tempfile.TemporaryDirectory(
        dir=os.environ.get("ANALYSIS_TEMP_DIR")
    ) as workdir:
        try:
            execution_timer = DebugTimer("SANDBOX WORKER")

            debug_log(
                "SANDBOX WORKER",
                "Launching isolated execution worker",
            )

            proc = subprocess.run(
                cmd,
                input=payload,
                capture_output=True,
                env=env,
                cwd=workdir,
                timeout=effective_timeout,
            )

            execution_timer.finish("Sandbox worker exited")

            debug_log(
                "SANDBOX WORKER",
                f"Exit code: {proc.returncode}",
            )

            debug_log(
                "SANDBOX WORKER",
                f"Standard output bytes: {len(proc.stdout)}",
            )

            debug_log(
                "SANDBOX WORKER",
                f"Standard error bytes: {len(proc.stderr)}",
            )
        except subprocess.TimeoutExpired:
            debug_log(
                "SANDBOX TIMEOUT",
                f"Worker exceeded {effective_timeout}s wall limit",
            )

            # Call `debug_log` when `docker`.
            if docker:  # kill docker
                debug_log(
                    "DOCKER",
                    f"Attempting to kill timed-out container: {name}",
                )
                with contextlib.suppress(Exception):
                    subprocess.run(
                        ["docker", "kill", name], capture_output=True, timeout=10
                    )
            return _fail(
                f"Execution Error: code timed out after {timeout} seconds "
                f"(wall limit {effective_timeout}s including startup). "
                f"Use vectorized pandas operations."
            )
        except OSError as exc:
            return _fail(f"Execution Error: could not start the sandbox ({exc})")

    try:
        debug_log(
            "RESULT",
            "Decoding sandbox worker JSON response",
        )

        result = json.loads(
            proc.stdout.decode("utf-8", "replace").strip().splitlines()[-1]
        )

        debug_log(
            "RESULT",
            f"Sandbox success: {result.get('success')}",
        )

        debug_log(
            "RESULT",
            f"Result type: {result.get('result_type')}",
        )

        # Call `debug_log` when `result.get('result_type') == 'table'`.
        if result.get("result_type") == "table":
            debug_log(
                "RESULT",
                f"Returned table rows: " f"{len(result.get('result') or [])}",
            )

        timer.finish("Sandbox execution pipeline finished")

        return result

    except (IndexError, ValueError):
        debug_log(
            "RESULT ERROR",
            "Unable to decode sandbox worker response",
        )

        stderr = proc.stderr.decode("utf-8", "replace").strip()[-400:]

        timer.finish("Sandbox execution pipeline failed")

        return _fail(
            f"Execution Error: worker crashed "
            f"(exit code {proc.returncode}). {stderr}"
        )
