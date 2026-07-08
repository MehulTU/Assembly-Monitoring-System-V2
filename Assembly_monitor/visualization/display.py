"""
visualization/display.py
================================
SUBSYSTEM: Visualization (SS9) — File 5 System Architecture

Draws the live display overlay on each frame — state, metrics, ROI,
bounding boxes, warmup indicator, and motion score bar.

This subsystem ONLY displays. It never changes any data.
That is the correct separation of concerns.
"""

import cv2
import numpy as np


# Colour palette (BGR format for OpenCV)
COLOR_ACTIVE  = (0,  165, 255)   # orange — assembly activity
COLOR_IDLE    = (180, 180, 180)  # grey — idle
COLOR_WARMUP  = (50,  200, 255)  # yellow — warming up
COLOR_ROI     = (0,  200,  80)   # green — ROI boundary
COLOR_BOX     = (0,  100, 255)   # blue — motion bounding box
COLOR_WHITE   = (255, 255, 255)
COLOR_BLACK   = (0,   0,   0)


class LiveDisplay:
    """
    Renders all visual overlays onto the camera frame.
    Maps to Visualization subsystem (SS9) in the system architecture.
    """

    def __init__(self, roi_points=None):
        self._roi_points = roi_points

    def render(self, frame, motion_result, event_state, event_duration,
               metrics_summary, is_warmup, warmup_remaining):
        """
        Render all overlays onto a copy of the frame.

        Args:
            frame:             Raw BGR frame from camera
            motion_result:     MotionResult from Processing (SS4)
            event_state:       Current state string from Decision (SS5)
            event_duration:    Seconds in current state
            metrics_summary:   Dict from Analytics (SS6) MetricsCalculator
            is_warmup:         True during background warmup
            warmup_remaining:  Frames remaining in warmup

        Returns:
            display_frame: frame with all overlays drawn
        """
        display = frame.copy()

        # ── ROI boundary ──────────────────────────────────────────────────
        if self._roi_points is not None:
            cv2.polylines(display, [self._roi_points], True, COLOR_ROI, 2)

        # ── Motion bounding boxes ─────────────────────────────────────────
        for (x, y, w, h) in motion_result.bounding_boxes:
            cv2.rectangle(display, (x, y), (x+w, y+h), COLOR_BOX, 2)

        # ── Status panel (top-left) ───────────────────────────────────────
        if is_warmup:
            state_color = COLOR_WARMUP
            state_text  = f"WARMING UP ({warmup_remaining} frames)"
        else:
            state_color = COLOR_ACTIVE if "ACTIVE" in event_state else COLOR_IDLE
            state_text  = event_state

        self._draw_panel(display, state_text, state_color,
                         event_duration, motion_result.motion_score)

        # ── Metrics panel (bottom-left) ───────────────────────────────────
        self._draw_metrics(display, metrics_summary, frame.shape)

        return display

    def _draw_panel(self, frame, state_text, color, duration, score):
        """Draw the main state panel at the top of the frame."""
        # Background rectangle
        cv2.rectangle(frame, (8, 8), (320, 90), COLOR_BLACK, -1)
        cv2.rectangle(frame, (8, 8), (320, 90), color, 2)

        # State text
        cv2.putText(frame, state_text, (14, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # Duration
        cv2.putText(frame, f"Duration: {duration:.1f}s", (14, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

        # Motion score bar
        bar_width = int(score * 2.8)  # scale 0-100% to 0-280px
        bar_width = min(bar_width, 280)
        cv2.rectangle(frame, (14, 68), (294, 80), (40, 40, 40), -1)
        cv2.rectangle(frame, (14, 68), (14 + bar_width, 80), color, -1)
        cv2.putText(frame, f"Motion: {score:.1f}%", (14, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_WHITE, 1)

    def _draw_metrics(self, frame, metrics, frame_shape):
        """Draw running metrics at the bottom of the frame."""
        h = frame_shape[0]
        y_start = h - 75
        cv2.rectangle(frame, (8, y_start), (320, h - 8), COLOR_BLACK, -1)
        cv2.rectangle(frame, (8, y_start), (320, h - 8), (60, 60, 60), 1)

        lines = [
            f"Events: {metrics.get('total_active_events', 0)} active  "
            f"{metrics.get('total_idle_events', 0)} idle",
            f"Avg step time:  {metrics.get('avg_step_time_sec', 0):.2f}s",
            f"Avg cycle time: {metrics.get('avg_cycle_time_sec', 0):.2f}s",
            f"Last step time: {metrics.get('last_step_time_sec', 0):.2f}s",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (14, y_start + 16 + i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_WHITE, 1)
