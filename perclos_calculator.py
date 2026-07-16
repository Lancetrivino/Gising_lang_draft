"""
PERCLOSCalculator - maintains a rolling window of eye-closed/open frames and
computes PERCLOS (percentage of eye closure), per the GisingLang thesis
(Chapter 3, "Eye Aspect Ratio and PERCLOS Algorithm").

FAST-RELEASE NOTE: real-world testing showed LAYER1 "sticking" for a long
time after the driver reopened their eyes. This is a natural consequence of
a 60-second rolling window - closed-eye samples from a deliberate eyes-shut
test don't vanish the instant the driver opens their eyes again, they only
age out of the window over time, so the raw windowed ratio can stay above
breach_threshold for close to the full window length even with eyes wide
open the whole time. That's mathematically correct as a *trend* metric
(that's the point of PERCLOS), but a poor fit for gating a real-time alert
that should stop promptly once the driver is clearly alert again.

release_seconds adds hysteresis: the windowed ratio still governs when the
breach turns ON (unchanged - a single blink still can't trigger it), but
once the driver's eyes have been continuously open for release_seconds, the
breach is force-cleared immediately rather than waiting for the window to
decay. The raw `perclos` value returned is always the true windowed ratio
(still correct for logging/thesis data) - only `breached` is affected.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

DEFAULT_WINDOW_SECONDS = 60.0
DEFAULT_BREACH_THRESHOLD = 0.20

# How long the driver's eyes must be continuously open before an active
# breach is force-cleared, instead of waiting for the 60s window to decay.
DEFAULT_RELEASE_SECONDS = 1.5


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
        release_seconds: float = DEFAULT_RELEASE_SECONDS,
    ):
        self.window_seconds = window_seconds
        self.breach_threshold = breach_threshold
        self.release_seconds = release_seconds
        self._samples: deque = deque()
        self._open_streak_start = None

    def _prune(self, current_timestamp: float) -> None:
        cutoff = current_timestamp - self.window_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def update(self, timestamp: float, is_closed: bool) -> PERCLOSResult:
        self._samples.append((timestamp, is_closed))
        self._prune(timestamp)

        sample_count = len(self._samples)
        if sample_count == 0:
            perclos = 0.0
        else:
            closed_count = sum(1 for _, closed in self._samples if closed)
            perclos = closed_count / sample_count

        breached = perclos >= self.breach_threshold

        if is_closed:
            self._open_streak_start = None
        else:
            if self._open_streak_start is None:
                self._open_streak_start = timestamp
            open_streak = timestamp - self._open_streak_start
            if breached and open_streak >= self.release_seconds:
                # Eyes have clearly been open long enough - don't leave the
                # alert hanging on stale closed-eye history still sitting in
                # the window.
                breached = False

        return PERCLOSResult(
            perclos=perclos,
            breached=breached,
            sample_count=sample_count,
        )

    def reset(self) -> None:
        self._samples.clear()
        self._open_streak_start = None