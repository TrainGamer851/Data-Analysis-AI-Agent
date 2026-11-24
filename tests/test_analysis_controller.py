"""Tests for analysis cancellation and timeout management."""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from agent import analysis_controller as controller
from agent.analysis_controller import (
    AnalysisJob,
    run_controlled_analysis,
)


@pytest.fixture
def fake_worker(monkeypatch):
    """Replace the analysis worker with a controlled subprocess."""

    original_popen = subprocess.Popen
    processes = []
    started = threading.Event()

    config = {
        "delay": 0,
        "returncode": 0,
    }

    def launch(args, **kwargs):
        if "agent.analysis_controller" not in args:
            return original_popen(args, **kwargs)

        result_path = args[-1]

        script = (
            "import json, time, sys; "
            "from pathlib import Path; "
            "time.sleep(float(sys.argv[2])); "
            "Path(sys.argv[1]).write_text("
            "json.dumps({'answer': '300.0', "
            "'table': None, 'error': None}), "
            "encoding='utf-8'); "
            "sys.exit(int(sys.argv[3]))"
        )

        process = original_popen(
            [
                sys.executable,
                "-c",
                script,
                result_path,
                str(config["delay"]),
                str(config["returncode"]),
            ],
            **kwargs,
        )

        processes.append(process)
        started.set()

        return process

    monkeypatch.setattr(controller.subprocess, "Popen", launch)

    # Avoid calling Docker during these controller unit tests.
    monkeypatch.setattr(
        controller,
        "cleanup_docker",
        lambda job_id: None,
    )

    yield config, processes, started

    # Prevent leftover test processes.
    for process in processes:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "orders.csv"
    path.write_text("Sales\n100\n200\n", encoding="utf-8")
    return str(path)


# ------------------------------------------------------------
# Job state
# ------------------------------------------------------------

def test_job_initially_not_cancelled():
    job = AnalysisJob()

    assert job.cancelled is False


def test_job_can_be_cancelled():
    job = AnalysisJob()
    job.cancel()

    assert job.cancelled is True


def test_jobs_have_unique_ids():
    first = AnalysisJob()
    second = AnalysisJob()

    assert first.job_id != second.job_id


# ------------------------------------------------------------
# Successful analysis
# ------------------------------------------------------------

def test_successful_analysis(fake_worker, csv_file):
    result = run_controlled_analysis(
        csv_file,
        "What are total sales?",
        timeout=10,
    )

    assert result == {
        "answer": "300.0",
        "table": None,
        "error": None,
    }


# ------------------------------------------------------------
# Cancellation
# ------------------------------------------------------------

def test_cancel_before_start(fake_worker, csv_file):
    _, processes, _ = fake_worker

    job = AnalysisJob()
    job.cancel()

    result = run_controlled_analysis(
        csv_file,
        "What are total sales?",
        job=job,
    )

    assert result["answer"] is None
    assert result["error"] == "Analysis stopped."
    assert processes == []


def test_cancel_running_analysis(fake_worker, csv_file):
    config, processes, started = fake_worker
    config["delay"] = 30

    job = AnalysisJob()
    results = []

    thread = threading.Thread(
        target=lambda: results.append(
            run_controlled_analysis(
                csv_file,
                "What are total sales?",
                job=job,
                timeout=60,
            )
        )
    )

    thread.start()

    try:
        assert started.wait(timeout=10)
    finally:
        job.cancel()
        thread.join(timeout=20)

    assert not thread.is_alive()
    assert results[0]["error"] == "Analysis stopped."
    assert processes[0].poll() is not None


# ------------------------------------------------------------
# Timeout
# ------------------------------------------------------------

def test_analysis_timeout(fake_worker, csv_file):
    config, processes, _ = fake_worker
    config["delay"] = 30

    result = run_controlled_analysis(
        csv_file,
        "What are total sales?",
        timeout=1,
    )

    assert result["answer"] is None
    assert "timed out" in result["error"]
    assert processes[0].poll() is not None


# ------------------------------------------------------------
# Process failure
# ------------------------------------------------------------

def test_worker_failure(fake_worker, csv_file):
    config, _, _ = fake_worker
    config["returncode"] = 1

    result = run_controlled_analysis(
        csv_file,
        "What are total sales?",
        timeout=10,
    )

    assert result["answer"] is None
    assert "process failed" in result["error"]


# ------------------------------------------------------------
# Slot management
# ------------------------------------------------------------

def test_slot_released_after_timeout(
    fake_worker,
    csv_file,
    monkeypatch,
):
    config, _, _ = fake_worker

    # Use one slot so a leaked slot is immediately detectable.
    monkeypatch.setattr(
        controller,
        "_SLOTS",
        threading.BoundedSemaphore(1),
    )

    config["delay"] = 30

    first = run_controlled_analysis(
        csv_file,
        "First question",
        timeout=1,
    )

    assert "timed out" in first["error"]

    config["delay"] = 0

    second = run_controlled_analysis(
        csv_file,
        "Second question",
        timeout=10,
    )

    assert second["error"] is None
    assert second["answer"] == "300.0"


def test_slot_released_after_cancellation(
    fake_worker,
    csv_file,
    monkeypatch,
):
    config, _, started = fake_worker

    monkeypatch.setattr(
        controller,
        "_SLOTS",
        threading.BoundedSemaphore(1),
    )

    config["delay"] = 30

    job = AnalysisJob()
    results = []

    thread = threading.Thread(
        target=lambda: results.append(
            run_controlled_analysis(
                csv_file,
                "First question",
                job=job,
                timeout=60,
            )
        )
    )

    thread.start()

    try:
        assert started.wait(timeout=10)
    finally:
        job.cancel()
        thread.join(timeout=20)

    assert not thread.is_alive()
    assert results[0]["error"] == "Analysis stopped."

    config["delay"] = 0

    second = run_controlled_analysis(
        csv_file,
        "Second question",
        timeout=10,
    )

    assert second["error"] is None
    assert second["answer"] == "300.0"

# ------------------------------------------------------------
# Docker cleanup
# ------------------------------------------------------------

def test_docker_cleanup_removes_only_matching_containers(monkeypatch):
    job = AnalysisJob()
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)

        if args[1] == "ps":
            return subprocess.CompletedProcess(
                args,
                0,
                stdout="container123\ncontainer456\n",
            )

        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(
        controller.subprocess,
        "run",
        fake_run,
    )

    controller.cleanup_docker(job.job_id)

    assert calls == [
        [
            "docker",
            "ps",
            "-aq",
            "--filter",
            f"label=data-analysis-job={job.job_id}",
        ],
        [
            "docker",
            "rm",
            "-f",
            "container123",
            "container456",
        ],
    ]


def test_docker_cleanup_without_containers(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)

        return subprocess.CompletedProcess(
            args,
            0,
            stdout="",
        )

    monkeypatch.setattr(
        controller.subprocess,
        "run",
        fake_run,
    )

    controller.cleanup_docker(AnalysisJob().job_id)

    assert len(calls) == 1
    assert calls[0][1] == "ps"


def test_docker_cleanup_handles_unavailable_docker(monkeypatch):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("Docker is unavailable")

    monkeypatch.setattr(
        controller.subprocess,
        "run",
        fake_run,
    )

    # Cleanup should not crash the application.
    controller.cleanup_docker(AnalysisJob().job_id)


def test_docker_cleanup_called_after_timeout(fake_worker, csv_file, monkeypatch):
    config, _, _ = fake_worker
    config["delay"] = 30

    job = AnalysisJob()
    cleaned_jobs = []

    monkeypatch.setattr(
        controller,
        "cleanup_docker",
        cleaned_jobs.append,
    )

    result = run_controlled_analysis(
        csv_file,
        "What are total sales?",
        job=job,
        timeout=1,
    )

    assert "timed out" in result["error"]
    assert cleaned_jobs == [job.job_id]


def test_docker_cleanup_called_after_cancellation(
    fake_worker,
    csv_file,
    monkeypatch,
):
    config, _, started = fake_worker
    config["delay"] = 30

    job = AnalysisJob()
    cleaned_jobs = []
    results = []

    monkeypatch.setattr(
        controller,
        "cleanup_docker",
        cleaned_jobs.append,
    )

    thread = threading.Thread(
        target=lambda: results.append(
            run_controlled_analysis(
                csv_file,
                "What are total sales?",
                job=job,
                timeout=60,
            )
        )
    )

    thread.start()

    try:
        assert started.wait(timeout=10)
    finally:
        job.cancel()
        thread.join(timeout=30)

    assert not thread.is_alive()
    assert results[0]["error"] == "Analysis stopped."
    assert cleaned_jobs == [job.job_id]