
"""Tests for coordinated analysis time budgets."""

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


@pytest.fixture(autouse=True)
def clear_deadline(monkeypatch):
    monkeypatch.delenv(DEADLINE_ENV, raising=False)


@pytest.fixture
def set_remaining(monkeypatch):
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


def test_no_controller_deadline():
    assert remaining_seconds() == float("inf")


def test_default_request_timeout():
    assert request_timeout(120) == 120


def test_remaining_time(set_remaining):
    set_remaining(60)

    assert remaining_seconds() == 60


def test_expired_deadline(set_remaining):
    set_remaining(-10)

    assert remaining_seconds() == 0


def test_code_request_maximum(set_remaining):
    set_remaining(120)

    assert code_request_timeout() == 75


def test_code_request_preserves_reserves(set_remaining):
    set_remaining(80)

    assert code_request_timeout() == 45


def test_code_request_rejected_when_time_is_low(
    set_remaining,
):
    set_remaining(38)

    with pytest.raises(AnalysisBudgetExceeded):
        code_request_timeout()


def test_explanation_request_maximum(set_remaining):
    set_remaining(60)

    assert explanation_request_timeout() == 15


def test_explanation_uses_remaining_time(set_remaining):
    set_remaining(12)

    assert explanation_request_timeout() == 7


def test_explanation_skipped_when_time_is_low(
    set_remaining,
):
    set_remaining(8)

    with pytest.raises(AnalysisBudgetExceeded):
        explanation_request_timeout()


def test_deadline_decreases(monkeypatch):
    monkeypatch.setenv(
        DEADLINE_ENV,
        str(time.monotonic() + 60),
    )

    first = remaining_seconds()
    second = remaining_seconds()

    assert second <= first


def test_invalid_deadline(monkeypatch):
    monkeypatch.setenv(DEADLINE_ENV, "nan")

    with pytest.raises(ValueError):
        remaining_seconds()
