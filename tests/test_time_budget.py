import time

import pytest

from agent.time_budget import (
    DEADLINE_ENV,
    AnalysisBudgetExceeded,
    remaining_seconds,
    request_timeout,
    code_request_timeout,
    explanation_request_timeout,
)


# Clear deadline.
@pytest.fixture(autouse=True)
def clear_deadline(monkeypatch):
    monkeypatch.delenv(DEADLINE_ENV, raising=False)


# Set the remaining time budget for the test.
@pytest.fixture
def set_remaining(monkeypatch):
    # Configure the timer with the supplied test duration.
    def configure(seconds):
        monkeypatch.setattr(
            "agent.time_budget.time.monotonic",
            lambda: 1000.0,
        )

        monkeypatch.setenv(
            DEADLINE_ENV,
            str(1000.0 + seconds),
        )

    return configure


# Verify that no controller deadline.
def test_no_controller_deadline():
    assert remaining_seconds() == float("inf")


# Verify that default request timeout.
def test_default_request_timeout():
    assert request_timeout(120) == 120


# Verify that remaining time.
def test_remaining_time(set_remaining):
    set_remaining(60)

    assert remaining_seconds() == 60


# Verify that expired deadline.
def test_expired_deadline(set_remaining):
    set_remaining(-10)

    assert remaining_seconds() == 0


# Verify that code request maximum.
def test_code_request_maximum(set_remaining):
    set_remaining(120)

    assert code_request_timeout() == 75


# Verify that code request preserves reserves.
def test_code_request_preserves_reserves(set_remaining):
    set_remaining(80)

    assert code_request_timeout() == 45


# Verify that code request rejected when time is low.
def test_code_request_rejected_when_time_is_low(
    set_remaining,
):
    set_remaining(38)

    with pytest.raises(AnalysisBudgetExceeded):
        code_request_timeout()


# Verify that explanation request maximum.
def test_explanation_request_maximum(set_remaining):
    set_remaining(60)

    assert explanation_request_timeout() == 15


# Verify that explanation uses remaining time.
def test_explanation_uses_remaining_time(set_remaining):
    set_remaining(12)

    assert explanation_request_timeout() == 7


# Verify that explanation skipped when time is low.
def test_explanation_skipped_when_time_is_low(
    set_remaining,
):
    set_remaining(8)

    with pytest.raises(AnalysisBudgetExceeded):
        explanation_request_timeout()


# Verify that deadline decreases.
def test_deadline_decreases(monkeypatch):
    monkeypatch.setenv(
        DEADLINE_ENV,
        str(time.monotonic() + 60),
    )

    first = remaining_seconds()
    second = remaining_seconds()

    assert second <= first


# Verify that invalid deadline.
def test_invalid_deadline(monkeypatch):
    monkeypatch.setenv(DEADLINE_ENV, "nan")

    with pytest.raises(ValueError):
        remaining_seconds()
