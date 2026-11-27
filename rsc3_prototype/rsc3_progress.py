"""One status line for the encoder and decoder.

The line is redrawn at most 60 times a second. Faster than that just floods the
terminal, and sleeping to invent updates would slow the codec, so a slow frame
simply updates when it finishes. stderr is not a terminal (a pipe, a capture,
the round-trip test) and the line is not drawn at all.
"""
from __future__ import annotations

import sys
import time


def _span(seconds):
    seconds = max(0.0, seconds)
    # Whole seconds hide a half-finished run behind "0s left". Tenths until 10s.
    if seconds < 10:
        return f"{seconds:0.1f}s"
    seconds = int(seconds + 0.5)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class Bar:
    """A single-line meter. `step` is safe to call once per frame."""

    HZ = 60.0
    WIDTH = 28

    def __init__(self, total, label, stream=None, enabled=None, clock=None):
        self.total = max(int(total), 1)
        self.done = 0
        self.label = label
        self.stream = sys.stderr if stream is None else stream
        self.clock = time.monotonic if clock is None else clock
        if enabled is None:
            enabled = getattr(self.stream, "isatty", lambda: False)()
        self.enabled = bool(enabled)
        self.t0 = self.clock()
        self.next_t = self.t0
        self._shown = 0
        self._closed = False
        if self.enabled:
            self._paint(force=True)

    def set_label(self, label):
        self.label = label
        if self.enabled:
            self._paint(force=True)

    def pump(self):
        """Redraw the elapsed time without counting a frame.

        The rate solve prices a handful of windows and does not know how many
        are left, so it cannot move the percentage. Leaving the line untouched
        holds the clock at the moment the label changed, which reads as a hang.
        """
        if self.enabled and not self._closed:
            self._paint(force=True)

    def add_total(self, n):
        """More work showed up (another rate-check pass). The percent steps back
        by that amount instead of sitting at 100% while the new pass runs."""
        self.total += max(int(n), 0)
        if self.enabled:
            self._paint(force=True)

    def step(self, n=1):
        self.done += int(n)
        if self.enabled:
            self._paint()

    def close(self):
        if self._closed:
            return
        self._closed = True
        if not self.enabled:
            return
        self.done = max(self.done, self.total)
        self._paint(force=True)
        self.stream.write("\n")
        self.stream.flush()

    def _paint(self, force=False):
        now = self.clock()
        if not force and now < self.next_t:
            return
        self.next_t = now + 1.0 / self.HZ
        frac = min(self.done / self.total, 1.0) if self.total else 1.0
        filled = int(round(frac * self.WIDTH))
        if self.done > 0 and filled == 0:
            filled = 1
        if self.done >= self.total:
            filled = self.WIDTH
        bar = "#" * filled + "-" * (self.WIDTH - filled)
        elapsed = now - self.t0
        if 0 < self.done < self.total and elapsed >= 0.3:
            left = elapsed * (self.total - self.done) / self.done
            extra = f"  {_span(elapsed)}  {_span(left)} left"
        else:
            extra = f"  {_span(elapsed)}"
        text = f"{self.label}  [{bar}] {frac * 100:5.1f}%{extra}"
        pad = max(0, self._shown - len(text))
        self.stream.write("\r" + text + " " * pad)
        self.stream.flush()
        self._shown = len(text)
