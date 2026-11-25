
"""Shared time budgets for controlled analysis jobs."""

import math
import os
import time


DEADLINE_ENV = "ANALYSIS_DEADLINE_MONOTONIC"

CODE_REQUEST_LIMIT = 75.0
EXPLANATION_REQUEST_LIMIT = 15.0

SANDBOX_RESERVE = 30.0
RESULT_RESERVE = 5.0
MIN_REQUEST_TIME = 5.0


class AnalysisBudgetExceeded(TimeoutError):
    """Not enough analysis time remains for another operation."""


def remaining_seconds():
    """Return remaining controlled analysis time."""

    raw_deadline = os.getenv(DEADLINE_ENV)

    if raw_deadline is None:
        return math.inf

    deadline = float(raw_deadline)

    if not math.isfinite(deadline):
        raise ValueError("Invalid analysis deadline.")

    return max(0.0, deadline - time.monotonic())


def request_timeout(
    limit,
    reserve=0.0,
    minimum=MIN_REQUEST_TIME,
):
    """Calculate a request timeout without consuming reserved time."""

    available = remaining_seconds() - reserve
    timeout = min(float(limit), available)

    if timeout < minimum:
        raise AnalysisBudgetExceeded(
            "Insufficient analysis time remains "
            "for another model request."
        )

    return timeout


def code_request_timeout():
    """Leave time for sandbox execution and result delivery."""

    return request_timeout(
        CODE_REQUEST_LIMIT,
        reserve=SANDBOX_RESERVE + RESULT_RESERVE,
    )


def explanation_request_timeout():
    """Leave time for returning an already calculated result."""

    return request_timeout(
        EXPLANATION_REQUEST_LIMIT,
        reserve=RESULT_RESERVE,
    )
