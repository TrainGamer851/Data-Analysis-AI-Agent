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
from agent.debug_logger import debug_log, DebugTimer
from agent.time_budget import DEADLINE_ENV
from agent.conversation_context import (
    build_conversation_context,
)

ANALYSIS_TIMEOUT_SECONDS = 120
MAX_CONCURRENT_ANALYSES = 2
POLL_INTERVAL_SECONDS = 0.1

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)

_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_ANALYSES)
_RUNNING_JOBS = set()
_RUNNING_CONDITION = threading.Condition()
_SHUTTING_DOWN = False


# Build the standard response object for an analysis error.
def error_response(message):
    return {
        "answer": None,
        "table": None,
        "error": message,
    }


class AnalysisJob:
    """Represent one independently cancellable analysis."""

    # Initialize the object and store its starting state.
    def __init__(self):
        self.job_id = uuid.uuid4().hex
        self.cancel_event = threading.Event()

    # Request cancellation of the analysis job.
    def cancel(self):
        self.cancel_event.set()

    # Return whether cancellation has been requested for the job.
    @property
    def cancelled(self):
        return self.cancel_event.is_set()


# Cancel active analyses and wait for their cleanup.
def shutdown_analysis_jobs(timeout=60):

    global _SHUTTING_DOWN

    deadline = time.monotonic() + timeout

    with _RUNNING_CONDITION:
        _SHUTTING_DOWN = True

        # Process each `job` from `_RUNNING_JOBS`.
        for job in _RUNNING_JOBS:
            job.cancel()

        # Repeat the block while `_RUNNING_JOBS` is true.
        while _RUNNING_JOBS:
            remaining = deadline - time.monotonic()

            # Call `logger.warning` when `remaining <= 0`.
            if remaining <= 0:
                logger.warning(
                    "Shutdown timed out with %d active jobs",
                    len(_RUNNING_JOBS),
                )
                return False

            _RUNNING_CONDITION.wait(timeout=remaining)

    return True


# Terminate the analysis process and its subprocesses.
def terminate_process(process):

    # Return without a value when `process.poll() is not None`.
    if process.poll() is not None:
        return

    try:
        # Call `subprocess.run` when `os.name == 'nt'`.
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
        # Call `os.killpg` when no earlier condition matches.
        else:
            os.killpg(process.pid, signal.SIGKILL)

        process.wait(timeout=5)

    except (OSError, subprocess.TimeoutExpired):
        # Call `process.kill` when `process.poll() is None`.
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


# Remove Docker containers belonging only to this job.
def cleanup_docker(job_id):

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

        debug_log(
            "DOCKER",
            f"Remaining job containers: {len(container_ids)}",
            job_id=job_id,
        )

        # Call `subprocess.run` when `container_ids`.
        if container_ids:
            subprocess.run(
                ["docker", "rm", "-f", *container_ids],
                capture_output=True,
                timeout=10,
                check=True,
            )

    except (OSError, subprocess.SubprocessError):
        debug_log(
            "ERROR",
            "Docker cleanup failed",
            job_id=job_id,
        )

        logger.exception(
            "Docker cleanup failed for job %s",
            job_id,
        )


# Run analysis with cancellation and an overall deadline.
def run_controlled_analysis(
    csv_path,
    question,
    job=None,
    explain=True,
    timeout=ANALYSIS_TIMEOUT_SECONDS,
    context=None,
):
    """Run analysis with cancellation and an overall deadline."""

    # Update `job` when `job is None`.
    if job is None:
        job = AnalysisJob()

    timer = DebugTimer(
        "CONTROLLER",
        job_id=job.job_id,
    )

    debug_log(
        "CONTROLLER",
        f"Analysis requested | Timeout: {timeout}s",
        job_id=job.job_id,
    )

    with _RUNNING_CONDITION:
        # Return the selected result when `_SHUTTING_DOWN`.
        if _SHUTTING_DOWN:
            return error_response("The analysis service is shutting down.")

        # Return the selected result when `job.cancelled`.
        if job.cancelled:
            return error_response("Analysis stopped.")

        # Return the selected result when `not _SLOTS.acquire(blocking=False)`.
        if not _SLOTS.acquire(blocking=False):
            return error_response(
                "The analysis service is busy. " "Please try again in a moment."
            )

        _RUNNING_JOBS.add(job)

    debug_log(
        "CONTROLLER",
        "Analysis slot acquired and job registered",
        job_id=job.job_id,
    )

    process = None
    deadline = time.monotonic() + timeout

    try:
        with tempfile.TemporaryDirectory(prefix="analysis-") as directory:

            request_path = Path(directory) / "request.json"
            result_path = Path(directory) / "result.json"

            request_path.write_text(
                json.dumps(
                    {
                        "csv_path": str(Path(csv_path).resolve()) if csv_path else None,
                        "question": question,
                        "explain": explain,
                        "context": build_conversation_context(context),
                    }
                ),
                encoding="utf-8",
            )

            debug_log(
                "CONTROLLER",
                "Worker request JSON created",
                job_id=job.job_id,
            )

            debug_log(
                "CONTEXT",
                f"Serialized messages: " f"{len(build_conversation_context(context))}",
                job_id=job.job_id,
            )

            env = os.environ.copy()
            env["ANALYSIS_JOB_ID"] = job.job_id

            # Share the controller's absolute deadline with the worker.
            env[DEADLINE_ENV] = str(deadline)

            debug_log(
                "BUDGET",
                f"Worker time remaining: "
                f"{max(0.0, deadline - time.monotonic()):.2f}s",
                job_id=job.job_id,
            )

            sandbox_temp = Path(directory) / "sandbox"
            sandbox_temp.mkdir()

            env["ANALYSIS_TEMP_DIR"] = str(sandbox_temp)

            process_options = {}

            # Update `process_options['creationflags']` when `os.name == 'nt'`.
            if os.name == "nt":
                process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            # Update `process_options['start_new_session']` when no earlier condition matches.
            else:
                process_options["start_new_session"] = True

            # Return the selected result when `job.cancelled`.
            if job.cancelled:
                return error_response("Analysis stopped.")

            debug_log(
                "PROCESS",
                "Launching isolated analysis subprocess",
                job_id=job.job_id,
            )

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

            debug_log(
                "PROCESS",
                "Analysis subprocess launched",
                job_id=job.job_id,
            )

            debug_log(
                "CONTROLLER",
                "Waiting for worker result",
                job_id=job.job_id,
            )

            try:
                # Repeat the block while `True` is true.
                while True:
                    # Call `debug_log` when `job.cancelled`.
                    if job.cancelled:
                        debug_log(
                            "CANCEL",
                            "Cancellation detected by controller",
                            job_id=job.job_id,
                        )

                        return error_response("Analysis stopped.")

                    # Call `debug_log` when `time.monotonic() >= deadline`.
                    if time.monotonic() >= deadline:
                        debug_log(
                            "TIMEOUT",
                            f"Analysis exceeded {timeout}s",
                            job_id=job.job_id,
                        )

                        return error_response(
                            f"Analysis timed out after " f"{timeout} seconds."
                        )

                    # Call `debug_log` when `process.poll() is not None`.
                    if process.poll() is not None:
                        debug_log(
                            "PROCESS",
                            f"Worker exited with code " f"{process.returncode}",
                            job_id=job.job_id,
                        )

                        break

                    job.cancel_event.wait(POLL_INTERVAL_SECONDS)

                # Return the selected result when `job.cancelled`.
                if job.cancelled:
                    return error_response("Analysis stopped.")

                # Return the selected result when `process.returncode != 0`.
                if process.returncode != 0:
                    return error_response(
                        "The analysis process failed. " "Please try again."
                    )

                # Return the selected result when `not result_path.exists()`.
                if not result_path.exists():
                    return error_response("The analysis returned no result.")

                debug_log(
                    "RESULT",
                    "Reading worker result JSON",
                    job_id=job.job_id,
                )

                result = json.loads(result_path.read_text(encoding="utf-8"))

                debug_log(
                    "RESULT",
                    (
                        "Worker returned an error"
                        if result.get("error")
                        else "Worker result received successfully"
                    ),
                    job_id=job.job_id,
                )

                return result

            finally:
                debug_log(
                    "CLEANUP",
                    "Starting subprocess cleanup",
                    job_id=job.job_id,
                )

                try:
                    # Call `debug_log` when `process.poll() is None`.
                    if process.poll() is None:
                        debug_log(
                            "PROCESS",
                            "Terminating active worker",
                            job_id=job.job_id,
                        )

                        terminate_process(process)

                        debug_log(
                            "PROCESS",
                            "Worker terminated",
                            job_id=job.job_id,
                        )

                finally:
                    debug_log(
                        "DOCKER",
                        "Checking for remaining job containers",
                        job_id=job.job_id,
                    )

                    cleanup_docker(job.job_id)

                    debug_log(
                        "CLEANUP",
                        "Docker cleanup attempt finished",
                        job_id=job.job_id,
                    )

    except Exception:
        logger.exception("Analysis controller failed")

        return error_response(
            "An unexpected analysis error occurred. " "Please try again."
        )

    finally:
        with _RUNNING_CONDITION:
            _SLOTS.release()

            _RUNNING_JOBS.discard(job)

            _RUNNING_CONDITION.notify_all()

        debug_log(
            "CLEANUP",
            "Analysis slot released and job unregistered",
            job_id=job.job_id,
        )

        timer.finish("Controller lifecycle finished")


# Execute the existing agent inside the child process.
def _worker(request_path, result_path):

    timer = DebugTimer("WORKER")

    debug_log(
        "WORKER",
        f"Analysis worker started | PID: {os.getpid()}",
    )

    import_timer = DebugTimer("IMPORT")

    debug_log(
        "WORKER",
        "Importing analysis agent",
    )

    debug_log(
        "IMPORT",
        "Loading essential analysis dependencies",
    )

    from agent.agent_runner import run_analysis

    import_timer.finish("Essential analysis dependencies loaded")

    debug_log(
        "IMPORT",
        f"mini-swe-agent loaded: " f"{'minisweagent' in sys.modules}",
    )

    debug_log(
        "IMPORT",
        f"LLM client loaded: " f"{'agent.llm_client' in sys.modules}",
    )

    debug_log(
        "IMPORT",
        f"Sandbox environment loaded: " f"{'agent.execution_env' in sys.modules}",
    )

    request = json.loads(Path(request_path).read_text(encoding="utf-8"))

    debug_log(
        "WORKER",
        "Request JSON loaded",
    )

    debug_log(
        "CONTEXT",
        f"Worker received " f"{len(request.get('context') or [])} messages",
    )

    debug_log(
        "WORKER",
        "Starting agent analysis",
    )

    result = run_analysis(
        request["csv_path"],
        request["question"],
        explain=request["explain"],
        context=request.get("context"),
    )

    debug_log(
        "IMPORT SUMMARY",
        f"mini-swe-agent loaded: " f"{'minisweagent' in sys.modules}",
    )

    debug_log(
        "IMPORT SUMMARY",
        f"LLM client loaded: " f"{'agent.llm_client' in sys.modules}",
    )

    debug_log(
        "IMPORT SUMMARY",
        f"Sandbox environment loaded: " f"{'agent.execution_env' in sys.modules}",
    )

    debug_log(
        "WORKER",
        "Agent returned an error" if result.get("error") else "Agent returned a result",
    )

    Path(result_path).write_text(
        json.dumps(result, default=str),
        encoding="utf-8",
    )

    debug_log(
        "WORKER",
        "Result JSON written successfully",
    )

    timer.finish("Worker execution finished")


# Evaluate the nested condition when `__name__ == '__main__'`.
if __name__ == "__main__":
    # Raise `SystemExit` when `len(sys.argv) != 3`.
    if len(sys.argv) != 3:
        raise SystemExit("Expected request and result file paths.")

    _worker(sys.argv[1], sys.argv[2])
