import threading

import pytest

from agent import analysis_controller as controller


@pytest.fixture(autouse=True)
def reset_shutdown_state(monkeypatch):
    monkeypatch.setattr(
        controller,
        "_SHUTTING_DOWN",
        False,
    )

    monkeypatch.setattr(
        controller,
        "_RUNNING_JOBS",
        set(),
    )


def test_shutdown_without_active_jobs():
    result = controller.shutdown_analysis_jobs(
        timeout=1
    )

    assert result is True
    assert controller._SHUTTING_DOWN is True


def test_shutdown_rejects_new_analysis():
    controller.shutdown_analysis_jobs(timeout=1)

    result = controller.run_controlled_analysis(
        "unused.csv",
        "What are the total sales?",
    )

    assert "shutting down" in result["error"]


def test_shutdown_cancels_active_job():
    job = controller.AnalysisJob()

    with controller._RUNNING_CONDITION:
        controller._RUNNING_JOBS.add(job)

    def finish_job():
        job.cancel_event.wait(timeout=2)

        with controller._RUNNING_CONDITION:
            controller._RUNNING_JOBS.discard(job)

            controller._RUNNING_CONDITION.notify_all()

    thread = threading.Thread(target=finish_job)
    thread.start()

    try:
        result = controller.shutdown_analysis_jobs(
            timeout=2
        )

        assert result is True
        assert job.cancelled is True
        assert not controller._RUNNING_JOBS

    finally:
        thread.join(timeout=3)


def test_shutdown_timeout():
    job = controller.AnalysisJob()

    with controller._RUNNING_CONDITION:
        controller._RUNNING_JOBS.add(job)

    result = controller.shutdown_analysis_jobs(
        timeout=0.1
    )

    assert result is False
    assert job.cancelled is True


def test_shutdown_can_be_called_twice():
    assert controller.shutdown_analysis_jobs(
        timeout=1
    ) is True

    assert controller.shutdown_analysis_jobs(
        timeout=1
    ) is True