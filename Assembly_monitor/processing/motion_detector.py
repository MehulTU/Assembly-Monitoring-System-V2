"""
processing/motion_detector.py
================================
SUBSYSTEM: Processing (SS4) — File 5 System Architecture

Full motion detection pipeline in one clean class.
Steps inside (in order):
  1. Crop to ROI
  2. Grayscale conversion
  3. Gaussian blur (noise reduction)
  4. Background subtraction
  5. Threshold → binary mask
  6. Morphological opening (removes noise dots)
  7. Contour detection
  8. Motion score calculation (% of ROI moving)

OUTPUT:
  MotionResult object containing:
    - motion_detected (bool)
    - motion_score (float, 0–100%)
    - contours list
    - processed frame for display
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List
from config.settings import (
    BLUR_KERNEL_SIZE, MOTION_THRESHOLD, MORPH_KERNEL_SIZE,
    MIN_CONTOUR_AREA, MOTION_SCORE_THRESHOLD,
    BACKGROUND_WARMUP_FRAMES
)


@dataclass
class MotionResult:
    """
    Result object returned by MotionDetector.process().
    All downstream subsystems (Decision, Analytics) use this.
    """
    motion_detected: bool          # True = motion above threshold
    motion_score:    float         # % of ROI pixels that are moving (0.0–100.0)
    contours:        List          = field(default_factory=list)  # list of motion contours
    bounding_boxes:  List          = field(default_factory=list)  # [(x,y,w,h), ...]
    binary_mask:     object        = None   # the cleaned binary motion mask (for debug display)
    is_warmup:       bool          = False  # True = background still warming up, don't log events


class MotionDetector:
    """
    Full processing pipeline from raw frame to motion score.
    Maps to Processing subsystem (SS4) in the system architecture.
    """

    def __init__(self, roi_points=None):
        """
        Args:
            roi_points: numpy array of 4 corner points defining the ROI,
                        from calibration/roi_config.json.
                        If None, the full frame is used.
        """
        self._roi_points    = roi_points
        self._roi_mask      = None
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=False
        )
        self._morph_kernel  = cv2.getStructuringElement(
            cv2.MORPH_RECT, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
        )
        self._frame_count   = 0    # counts frames for warmup tracking
        self._roi_area      = None # pixel area of the ROI (set on first frame)

    def process(self, frame) -> MotionResult:
        """
        Run the full processing pipeline on one frame.

        Args:
            frame: BGR numpy array from camera

        Returns:
            MotionResult
        """
        self._frame_count += 1
        still_warming_up = self._frame_count <= BACKGROUND_WARMUP_FRAMES

        # ── Step 1: Build ROI mask on first frame ────────────────────────
        if self._roi_mask is None:
            self._roi_mask = self._build_roi_mask(frame.shape)
            self._roi_area = max(1, int(cv2.countNonZero(self._roi_mask)))

        # ── Step 2: Grayscale ─────────────────────────────────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Step 3: Gaussian blur (reduces sensor noise) ──────────────────
        blurred = cv2.GaussianBlur(gray, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)

        # ── Step 4: Background subtraction ───────────────────────────────
        fg_mask = self._bg_subtractor.apply(blurred)

        # ── Step 5: Apply ROI mask (ignore everything outside ROI) ────────
        fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=self._roi_mask)

        # ── Step 6: Threshold → clean binary mask ─────────────────────────
        _, binary = cv2.threshold(fg_mask, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)

        # ── Step 7: Morphological opening (erode then dilate) ─────────────
        # This removes tiny noise dots while preserving real motion regions.
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self._morph_kernel)

        # ── Step 8: Contour detection ─────────────────────────────────────
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out tiny contours below minimum area
        valid_contours  = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
        bounding_boxes  = [cv2.boundingRect(c) for c in valid_contours]

        # ── Step 9: Motion score ──────────────────────────────────────────
        # Count white pixels (moving pixels) as % of ROI area
        moving_pixels = cv2.countNonZero(binary)
        motion_score  = (moving_pixels / self._roi_area) * 100.0
        motion_detected = motion_score >= MOTION_SCORE_THRESHOLD

        return MotionResult(
            motion_detected = motion_detected and not still_warming_up,
            motion_score    = motion_score,
            contours        = valid_contours,
            bounding_boxes  = bounding_boxes,
            binary_mask     = binary,
            is_warmup       = still_warming_up,
        )

    def _build_roi_mask(self, frame_shape) -> np.ndarray:
        """Create a binary mask from ROI corner points."""
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        if self._roi_points is not None:
            cv2.fillPoly(mask, [self._roi_points], 255)
        else:
            mask[:, :] = 255   # full frame if no ROI defined
        return mask
