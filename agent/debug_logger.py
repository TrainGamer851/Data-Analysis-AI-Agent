import os
import re
import threading

from datetime import datetime
from time import perf_counter

_print_lock = threading.Lock()


# Print a timestamped execution-stage message.
def debug_log(stage, message, *, job_id=None):

    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    job_id = job_id or os.environ.get("ANALYSIS_JOB_ID")

    job_label = f" [JOB:{str(job_id)[:8]}]" if job_id else ""

    # Remove terminal control characters.
    safe_message = re.sub(
        r"[\x00-\x08\x0b-\x1f\x7f]",
        "",
        str(message),
    )

    with _print_lock:
        print(
            f"[{timestamp}]{job_label} " f"[{stage.upper()}] {safe_message}",
            flush=True,
        )


class DebugTimer:
    """Measure execution duration for an analysis stage."""

    # Initialize the object and store its starting state.
    def __init__(self, stage, *, job_id=None):
        self.stage = stage
        self.job_id = job_id
        self.started = perf_counter()

    # Return the elapsed time since the timer started.
    def elapsed(self):
        return perf_counter() - self.started

    # Stop the timer and record its final duration.
    def finish(self, message="Completed"):
        duration = self.elapsed()

        debug_log(
            self.stage,
            f"{message} | Duration: {duration:.2f}s",
            job_id=self.job_id,
        )

        return duration
