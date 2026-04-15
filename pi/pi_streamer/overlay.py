"""OpenCV drawing for detection + pose overlays.

Draws bounding boxes, COCO-pose skeletons, keypoints, object-detection
bboxes, and an FPS counter directly onto a BGR numpy array in-place.
Designed to run on every video frame while inference results lag 1-2
frames behind.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Sequence

import numpy as np

from .hand_types import HandLandmarks
from .inference import (
    COCO_SKELETON,
    KEYPOINT_SCORE_THRESHOLD,
    HandInferenceResult,
    InferenceResult,
    ObjectDetection,
    ObjectInferenceResult,
    PersonDetection,
)

LOG = logging.getLogger("pi_streamer.overlay")

# Colors in BGR order (picamera2 RGB888 format is actually BGR in memory,
# and OpenCV tuples are BGR too, so everything lines up).
BBOX_COLOR = (80, 220, 120)            # pose bbox — lime green
LABEL_BG_COLOR = (0, 0, 0)
LABEL_TEXT_COLOR = (255, 255, 255)
SKELETON_COLOR = (230, 150, 40)        # pose skeleton — bright blue
KEYPOINT_COLOR_HIGH = (255, 240, 50)   # keypoint hi-conf — cyan
KEYPOINT_COLOR_LOW = (180, 180, 180)
FPS_COLOR = (200, 255, 200)
OBJECT_BBOX_COLOR = (0, 140, 255)      # object bbox — vivid orange
OBJECT_LABEL_BG = (0, 90, 180)         # darker orange for label bg

# Hand skeleton rendering. Pale cyan-green chosen to be distinct from
# the pose skeleton (bright blue) and pose keypoints (cyan/yellow) so
# hand lines are unambiguous even when the wrist sits on top of a pose
# keypoint. Thickness matches pose lines.
HAND_COLOR_BGR = (0, 255, 200)
HAND_KPT_RADIUS = 3
HAND_LINE_THICKNESS = 2
# Lowered from 0.30 because when presence used min(fc2, fc4), the
# broadcast score often dropped below 0.30 on confident-but-uncertain
# hands. The decoder now uses max() instead; 0.15 stays as a safety net
# so marginal presence still renders something rather than nothing.
HAND_MIN_KPT_SCORE = 0.15

# 21-point MediaPipe hand topology — 21 edges forming the 5 fingers plus
# the palm base. Wrist is index 0; indices 1-4 are the thumb; 5-8 index
# finger; 9-12 middle; 13-16 ring; 17-20 pinky. The (0, 17) edge closes
# the palm base so the hand renders as a closed polygon at the wrist.
HAND_SKELETON = (
    # thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # middle finger
    (5, 9), (9, 10), (10, 11), (11, 12),
    # ring finger
    (9, 13), (13, 14), (14, 15), (15, 16),
    # pinky
    (13, 17), (17, 18), (18, 19), (19, 20),
    # palm base
    (0, 17),
)

# yolo26m returns class 0 (person) in the same frame where yolov8m_pose
# already has the person bbox. Suppress the duplicate in overlay so pose
# remains the authoritative source for people.
SUPPRESS_OBJECT_CLASSES = frozenset({0})


class OverlayDrawer:
    """Stateful overlay drawer that tracks FPS and last-known inference."""

    def __init__(self, frame_width: int, frame_height: int) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._frame_count = 0
        self._fps_window_start = time.monotonic()
        self._fps_window_frames = 0
        self._current_fps = 0.0
        self._last_inference_frame_id = -1

    def draw(
        self,
        frame: np.ndarray,
        inference: Optional[InferenceResult],
        stream_active: bool,
        objects: Optional[ObjectInferenceResult] = None,
        hands: Optional[HandInferenceResult] = None,
    ) -> None:
        """Draw all overlays in-place on ``frame``.

        ``inference`` carries the latest pose result (persons + keypoints);
        ``objects`` carries the latest yolo26m object-detection result;
        ``hands`` carries the latest 21-point hand-landmark result from
        :class:`pi_streamer.inference.HailoHandInference`. Any of them
        may be ``None`` if the corresponding worker has not produced a
        result yet. Class 0 (person) in ``objects`` is suppressed to
        avoid duplicate bboxes on top of the pose person. Hand skeletons
        are drawn last so the finger joints sit on top of pose keypoints
        when a wrist is occluded.
        """
        import cv2  # lazy import for dev machines without cv2

        self._frame_count += 1
        self._fps_window_frames += 1
        now = time.monotonic()
        elapsed = now - self._fps_window_start
        if elapsed >= 1.0:
            self._current_fps = self._fps_window_frames / elapsed
            self._fps_window_frames = 0
            self._fps_window_start = now

        # Draw objects first so pose overlays sit on top.
        if objects is not None:
            for obj in objects.objects:
                if obj.class_id in SUPPRESS_OBJECT_CLASSES:
                    continue
                self._draw_object(cv2, frame, obj)

        if inference is not None:
            self._last_inference_frame_id = inference.frame_id
            for person in inference.persons:
                self._draw_person(cv2, frame, person)

        # Hand skeletons on top of body pose: when the wrist is
        # mid-motion, the hand's 0-landmark (also "wrist") sits on top
        # of pose keypoint 9 or 10. Drawing hands last means the finger
        # joints remain visible over the pose keypoint circle.
        if hands is not None:
            for hand in hands.hands:
                self._draw_hand(cv2, frame, hand)

        self._draw_hud(cv2, frame, inference, stream_active, objects, hands)

    # ---- person drawing ----------------------------------------------------

    def _draw_person(self, cv2, frame: np.ndarray, person: PersonDetection) -> None:
        x1 = int(round(person.x1))
        y1 = int(round(person.y1))
        x2 = int(round(person.x2))
        y2 = int(round(person.y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), BBOX_COLOR, 2)

        label = f"person {int(round(person.score * 100))}%"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        label_y = max(y1 - 6, th + 4)
        cv2.rectangle(
            frame,
            (x1, label_y - th - baseline - 2),
            (x1 + tw + 4, label_y + 2),
            LABEL_BG_COLOR,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 2, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            LABEL_TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )

        # Skeleton first (so keypoints draw on top)
        for start, end in COCO_SKELETON:
            kp_a = person.keypoints[start]
            kp_b = person.keypoints[end]
            if kp_a.score < KEYPOINT_SCORE_THRESHOLD or kp_b.score < KEYPOINT_SCORE_THRESHOLD:
                continue
            cv2.line(
                frame,
                (int(round(kp_a.x)), int(round(kp_a.y))),
                (int(round(kp_b.x)), int(round(kp_b.y))),
                SKELETON_COLOR,
                2,
                cv2.LINE_AA,
            )

        for kp in person.keypoints:
            if kp.score < KEYPOINT_SCORE_THRESHOLD:
                continue
            color = KEYPOINT_COLOR_HIGH if kp.score > 0.6 else KEYPOINT_COLOR_LOW
            cv2.circle(
                frame,
                (int(round(kp.x)), int(round(kp.y))),
                3,
                color,
                -1,
                cv2.LINE_AA,
            )

    # ---- hand drawing ------------------------------------------------------

    def _draw_hand(
        self, cv2, frame: np.ndarray, hand: HandLandmarks
    ) -> None:
        """Draw a 21-point MediaPipe-style hand skeleton in-place.

        Each of the 20 edges in ``HAND_SKELETON`` becomes an antialiased
        line; each landmark above ``HAND_MIN_KPT_SCORE`` becomes a filled
        circle. Because ``HailoHandInference`` broadcasts the overall
        presence score across all 21 landmarks, in practice the whole
        hand either draws or it doesn't — matching MediaPipe's own
        rendering convention where hand landmarks are all-or-none.
        """
        kpts = hand.kpts
        scores = hand.kpt_scores

        # Skeleton lines
        for i, j in HAND_SKELETON:
            if scores[i] < HAND_MIN_KPT_SCORE or scores[j] < HAND_MIN_KPT_SCORE:
                continue
            pi = (int(round(float(kpts[i, 0]))), int(round(float(kpts[i, 1]))))
            pj = (int(round(float(kpts[j, 0]))), int(round(float(kpts[j, 1]))))
            cv2.line(
                frame,
                pi,
                pj,
                HAND_COLOR_BGR,
                HAND_LINE_THICKNESS,
                cv2.LINE_AA,
            )

        # Keypoint circles on top
        for k in range(kpts.shape[0]):
            if scores[k] < HAND_MIN_KPT_SCORE:
                continue
            cv2.circle(
                frame,
                (int(round(float(kpts[k, 0]))), int(round(float(kpts[k, 1])))),
                HAND_KPT_RADIUS,
                HAND_COLOR_BGR,
                -1,
                cv2.LINE_AA,
            )

    # ---- object drawing ----------------------------------------------------

    def _draw_object(
        self, cv2, frame: np.ndarray, obj: ObjectDetection
    ) -> None:
        x1 = int(round(obj.x1))
        y1 = int(round(obj.y1))
        x2 = int(round(obj.x2))
        y2 = int(round(obj.y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), OBJECT_BBOX_COLOR, 2)

        label = f"{obj.class_name} {int(round(obj.score * 100))}%"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
        )
        label_y = max(y1 - 6, th + 4)
        cv2.rectangle(
            frame,
            (x1, label_y - th - baseline - 2),
            (x1 + tw + 4, label_y + 2),
            OBJECT_LABEL_BG,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 2, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            LABEL_TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )

    # ---- HUD ---------------------------------------------------------------

    def _draw_hud(
        self,
        cv2,
        frame: np.ndarray,
        inference: Optional[InferenceResult],
        stream_active: bool,
        objects: Optional[ObjectInferenceResult] = None,
        hands: Optional[HandInferenceResult] = None,
    ) -> None:
        hud_lines = [
            f"RPi Pose+Obj+Hand  {self._current_fps:.1f} fps",
        ]
        if inference is not None:
            hud_lines.append(
                f"pose f{inference.frame_id}  {inference.latency_ms:.0f}ms  p={len(inference.persons)}"
            )
        else:
            hud_lines.append("pose: waiting...")
        if objects is not None:
            visible = sum(
                1 for o in objects.objects
                if o.class_id not in SUPPRESS_OBJECT_CLASSES
            )
            hud_lines.append(
                f"obj  f{objects.frame_id}  {objects.latency_ms:.0f}ms  o={visible}"
            )
        else:
            hud_lines.append("obj: waiting...")
        if hands is not None:
            hud_lines.append(
                f"hand f{hands.frame_id}  {hands.latency_ms:.0f}ms  h={len(hands.hands)}/{hands.num_crops}"
            )
        hud_lines.append("STREAMING" if stream_active else "IDLE (no client)")

        x = 12
        y = 22
        for line in hud_lines:
            cv2.putText(
                frame,
                line,
                (x + 1, y + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                LABEL_BG_COLOR,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                FPS_COLOR,
                1,
                cv2.LINE_AA,
            )
            y += 22

    @property
    def current_fps(self) -> float:
        return self._current_fps
