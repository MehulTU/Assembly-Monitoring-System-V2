"""
decision/event_detector.py
================================
SUBSYSTEM: Decision (SS5) — File 5 System Architecture

This is NOT image processing. This is logic.

OpenCV (Processing) tells us: "motion detected"
Decision asks:               "does this motion mean something happened?"

STATE MACHINE:
  The system is always in one of two states:
    IDLE   → nothing is happening
    ACTIVE → assembly activity is occurring

  Transitions:
    IDLE   → ACTIVE  when motion is detected for N consecutive frames
    ACTIVE → IDLE    when no motion for N frames AND idle lasted > min duration

  This uses temporal filtering to avoid false events from brief flickers.

IMPORTANT NOTE on Pick vs Place:
  With motion detection only (V1), we CANNOT distinguish Pick from Place.
  Both look like "motion in the ROI". They are logged as "ACTIVE (Pick/Place)".
  Distinguishing Pick from Place requires object recognition (YOLO — V3).
  This is a documented V1 limitation, not a bug.
"""

import time
from dataclasses import dataclass
from config.settings import (
    TEMPORAL_FILTER_FRAMES,
    IDLE_MIN_DURATION_SEC,
    ACTIVE_MIN_DURATION_SEC,
)


@dataclass
class AssemblyEvent:
    """
    Represents one detected assembly event.
    Passed to Analytics (SS6) and Storage (SS8).
    """
    event_type:   str    # "ACTIVE (Pick/Place)" or "IDLE"
    start_time:   float  # Unix timestamp (seconds)
    end_time:     float  # Unix timestamp (seconds)
    duration_sec: float  # end_time - start_time


class EventDetector:
    """
    State machine that converts per-frame motion signals
    into meaningful assembly events.

    Maps to Decision subsystem (SS5) in the system architecture.
    """

    IDLE   = "IDLE"
    ACTIVE = "ACTIVE"

    def __init__(self, on_event_detected=None):
        """
        Args:
            on_event_detected: optional callback function called whenever
                               a complete event is detected.
                               Signature: callback(event: AssemblyEvent)
        """
        self._state             = self.IDLE
        self._state_start_time  = time.time()
        self._motion_streak     = 0    # consecutive frames with motion
        self._still_streak      = 0    # consecutive frames without motion
        self._on_event          = on_event_detected
        self.current_state      = self.IDLE
        self.current_duration   = 0.0

    def update(self, motion_detected: bool, is_warmup: bool = False):
        """
        Feed one frame's motion result into the state machine.

        Args:
            motion_detected: True if Processing found motion this frame
            is_warmup:       True during background warmup — no events logged

        Returns:
            AssemblyEvent if a complete event was just finished, else None.
        """
        now = time.time()

        # Update streak counters
        if motion_detected:
            self._motion_streak += 1
            self._still_streak   = 0
        else:
            self._still_streak  += 1
            self._motion_streak  = 0

        completed_event = None

        if not is_warmup:
            # ── Transition: IDLE → ACTIVE ─────────────────────────────────
            if (self._state == self.IDLE
                    and self._motion_streak >= TEMPORAL_FILTER_FRAMES):
                self._state           = self.ACTIVE
                self._state_start_time = now

            # ── Transition: ACTIVE → IDLE ─────────────────────────────────
            elif (self._state == self.ACTIVE
                  and self._still_streak >= TEMPORAL_FILTER_FRAMES):
                duration = now - self._state_start_time
                if duration >= ACTIVE_MIN_DURATION_SEC:
                    completed_event = AssemblyEvent(
                        event_type   = "ACTIVE (Pick/Place)",
                        start_time   = self._state_start_time,
                        end_time     = now,
                        duration_sec = duration,
                    )
                    if self._on_event:
                        self._on_event(completed_event)
                self._state            = self.IDLE
                self._state_start_time = now

        # Update public attributes for display
        self.current_state    = self._state
        self.current_duration = now - self._state_start_time

        return completed_event
