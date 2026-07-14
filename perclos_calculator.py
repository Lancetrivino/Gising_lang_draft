"""
PERCLOSCalculator - maintains a rolling window of eye-closed/open frames and
computes PERCLOS (percentage of eye closure), per the GisingLang thesis
(Chapter 3, "Eye Aspect Ratio and PERCLOS Algorithm").

    "The PERCLOS calculator receives this binary [eye_closed] signal and
    maintains a rolling deque of (timestamp, is_closed) pairs, pruning
    entries older than the 60-second window. The PERCLOS value is computed
    as the proportion of closed frames within the current window. When
    PERCLOS exceeds 0.20 (20 percent), the DrowsinessClassifier marks the
    PERCLOS channel as breached."

This module has no camera or hardware dependency - it consumes the
eye_closed booleans produced by EARCalculator, so it's fully unit testable
with synthetic timestamps (see tests/test_perclos_calculator.py). Corresponds
to test cases UT-02 and UT-03 in the Software Testing document.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

# Thesis Chapter 3: 60-second rolling window.
DEFAULT_WINDOW_SECONDS = 60.0

# Thesis Chapter 3, Three-Output Notification Logic: "PERCLOS >= 0.20" is the
# exact boundary used by the DrowsinessClassifier's breach decision.
DEFAULT_BREACH_THRESHOLD = 0.20


@dataclass
class PERCLOSResult:
    perclos: float
    breached: bool
    sample_count: int


class PERCLOSCalculator:
    """Rolling-window PERCLOS (percentage of eye closure) calculator."""

    def __init__(
        self,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        breach_threshold: float = DEFAULT_BREACH_THRESHOLD,
    ):
        self.window_seconds = window_seconds
        self.breach_threshold = breach_threshold
        # Each entry is (timestamp_seconds, is_closed).
        self._samples: deque[tuple[float, bool]] = deque()

    def _prune(self, current_timestamp: float) -> None:
        cutoff = current_timestamp - self.window_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def update(self, timestamp: float, is_closed: bool) -> PERCLOSResult:
        """
        Feed one new frame's eye-state into the rolling window.

        timestamp: seconds, monotonically increasing (e.g. time.monotonic()
            in production; a plain float in tests).
        is_closed: EARCalculator's per-frame eye_closed flag.
        """
        self._samples.append((timestamp, is_closed))
        self._prune(timestamp)

        sample_count = len(self._samples)
        if sample_count == 0:
            perclos = 0.0
        else:
            closed_count = sum(1 for _, closed in self._samples if closed)
            perclos = closed_count / sample_count

        return PERCLOSResult(
            perclos=perclos,
            breached=perclos >= self.breach_threshold,
            sample_count=sample_count,
        )

    def reset(self) -> None:
        """Clears the rolling window, e.g. after a device restart."""
        self._samples.clear()
