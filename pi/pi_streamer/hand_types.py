"""Frame-space hand-landmark dataclasses.

Separated from ``inference.py`` to avoid circular imports: both
``tracker.py`` (for ``HandTracker``) and ``overlay.py`` (for ``_draw_hand``)
need to import these types, and ``inference.py`` already imports
``tracker.py``. Keeping the dataclasses in a third module that has no
dependencies on the other streamer modules breaks the cycle.

The 21-point topology follows MediaPipe Hands:

    0  wrist
    1-4   thumb  (CMC, MCP, IP, TIP)
    5-8   index  (MCP, PIP, DIP, TIP)
    9-12  middle (MCP, PIP, DIP, TIP)
    13-16 ring   (MCP, PIP, DIP, TIP)
    17-20 pinky  (MCP, PIP, DIP, TIP)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NUM_HAND_KEYPOINTS = 21


@dataclass(frozen=True)
class HandLandmarks:
    """21 hand landmarks in frame pixel coordinates.

    Produced by ``HailoHandInference`` after dequantizing the
    ``hand_landmark_lite`` HEF output and back-projecting through the
    ``CropInfo.affine`` that gave rise to the input crop. Consumed by
    ``HandTracker`` (which produces a smoothed instance of the same
    dataclass) and finally by ``OverlayDrawer._draw_hand``.

    Fields:
      kpts        : shape (21, 2) float32, frame pixel (x, y) per landmark.
      kpt_scores  : shape (21,)  float32, per-landmark visibility. The
                    ``hand_landmark_lite`` model does not expose per-point
                    confidence, so this is populated with the overall
                    ``presence`` value broadcast across all 21 landmarks.
                    Kept as an array (rather than a scalar) so the same
                    rendering path as ``PersonDetection`` can gate each
                    point against a threshold.
      presence    : overall hand presence score 0..1.
      person_id   : stable identity from the upstream pose tracker; paired
                    with ``side`` to form the ``(person_id, side)`` key
                    used by ``HandTracker``.
      side        : ``"left"`` or ``"right"``. Source of truth is the wrist
                    index from the pose detector (``LEFT_WRIST``/``RIGHT_WRIST``)
                    — the model's own handedness output is intentionally
                    ignored because it is unreliable in top-down crops.
    """

    kpts: np.ndarray
    kpt_scores: np.ndarray
    presence: float
    person_id: int
    side: str
