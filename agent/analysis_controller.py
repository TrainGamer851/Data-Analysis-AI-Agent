"""Manage cancellable analysis subprocesses."""

import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path


ANALYSIS_TIMEOUT_SECONDS = 120
MAX_CONCURRENT_ANALYSES = 2
POLL_INTERVAL_SECONDS = 0.1

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)

_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_ANALYSES)


def error_response(message):
    return {
        "answer": None,
        "table": None,
        "error": message,
    }


class AnalysisJob:
    """Represent one independently cancellable analysis."""

    def __init__(self):
        self.job_id = uuid.uuid4().hex
        self.cancel_event = threading.Event()

    def cancel(self):
        self.cancel_event.set()

    @property
    def cancelled(self):
        return self.cancel_event.is_set()


def terminate_process(process):
    """Terminate the analysis process and its subprocesses."""

    if process.poll() is not None:
        return

    try:
        if os.name == "nt":
            subprocess.run(
                [
                    "taskkill",
                    "/PID",
                    str(process.pid),
                    "/T",
                    "/F",
                ],
                capture_output=True,
                timeout=10,
            )
        else:
            os.killpg(process.pid, signal.SIGKILL)

        process.wait(timeout=5)

    except (OSError, subprocess.TimeoutExpired):
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def cleanup_docker(job_id):
    """Remove Docker containers belonging only to this job."""

    try:
        result = subprocess.run(
            [
                "docker",
                "ps",
                "-aq",
                "--filter",
                f"label=data-analysis-job={job_id}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )

        container_ids = result.stdout.split()

        if container_ids:
            subprocess.run(
                ["docker", "rm", "-f", *container_ids],
                capture_output=True,
                timeout=10,
                check=True,
            )

    except (OSError, subprocess.SubprocessError):
        logger.exception(
            "Docker cleanup failed for job %s",
            job_id,
        )


def run_controlled_analysis(
    csv_path,
    question,
    job=None,
    explain=True,
    timeout=ANALYSIS_TIMEOUT_SECONDS,
):
    """Run analysis with cancellation and an overall deadline."""

    if job is None:
        job = AnalysisJob()

    if job.cancelled:
        return error_response("Analysis stopped.")

    if not _SLOTS.acquire(blocking=False):
        return error_response(
            "The analysis service is busy. "
            "Please try again in a moment."
        )

    process = None
    deadline = time.monotonic() + timeout

    try:
        with tempfile.TemporaryDirectory(
            prefix="analysis-"
        ) as directory:

            request_path = Path(directory) / "request.json"
            result_path = Path(directory) / "result.json"

            request_path.write_text(
                json.dumps({
                    "csv_path": str(Path(csv_path).resolve()),
                    "question": question,
                    "explain": explain,
                }),
                encoding="utf-8",
            )

            env = os.environ.copy()
            env["ANALYSIS_JOB_ID"] = job.job_id

            process_options = {}

            if os.name == "nt":
                process_options["creationflags"] = (
                    subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:
                process_options["start_new_session"] = True

            if job.cancelled:
                return error_response("Analysis stopped.")

            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "agent.analysis_controller",
                    str(request_path),
                    str(result_path),
                ],
                cwd=str(PROJECT_ROOT),
                env=env,
                **process_options,
            )

            try:
                while True:
                    if job.cancelled:
                        return error_response(
                            "Analysis stopped."
                        )

                    if time.monotonic() >= deadline:
                        return error_response(
                            f"Analysis timed out after "
                            f"{timeout} seconds."
                        )

                    if process.poll() is not None:
                        break

                    job.cancel_event.wait(
                        POLL_INTERVAL_SECONDS
                    )

                if job.cancelled:
                    return error_response("Analysis stopped.")

                if process.returncode != 0:
                    return error_response(
                        "The analysis process failed. "
                        "Please try again."
                    )

                if not result_path.exists():
                    return error_response(
                        "The analysis returned no result."
                    )

                return json.loads(
                    result_path.read_text(encoding="utf-8")
                )

            finally:
                if process.poll() is None:
                    terminate_process(process)

                # Clean up any containers left by this job.
                cleanup_docker(job.job_id)

    except Exception:
        logger.exception("Analysis controller failed")

        return error_response(
            "An unexpected analysis error occurred. "
            "Please try again."
        )

    finally:
        _SLOTS.release()


def _worker(request_path, result_path):
    """Execute the existing agent inside the child process."""

    from agent.agent_runner import run_analysis

    request = json.loads(
        Path(request_path).read_text(encoding="utf-8")
    )

    result = run_analysis(
        request["csv_path"],
        request["question"],
        explain=request["explain"],
    )

    Path(result_path).write_text(
        json.dumps(result, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "Expected request and result file paths."
        )

    _worker(sys.argv[1], sys.argv[2])