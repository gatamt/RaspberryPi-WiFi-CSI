"""Temporal trackers for pose and object detections.

Eliminates visible flicker in the baked-in overlays by matching
detections across frames (greedy IOU) and smoothing bbox coordinates
(plus per-keypoint coordinates for pose) with an exponential moving
average. A short "confirmation" hysteresis (must be seen in N consecutive
updates before being rendered) suppresses one-frame false positives;
a longer "coasting" window (remains visible for M seconds after the last
detection) rides out short dropouts.

Two concrete trackers share a small set of helpers but are kept as
separate classes because their payload shapes diverge (object has class
hysteresis, person has 17 keypoints with visibility):

  - :class:`ObjectTracker` smooths bbox + score, with class-switching
    hysteresis based on a rolling window of recent ``class_id`` values.
  - :class:`PersonTracker` smooths bbox + 17 keypoint (x, y, score)
    triples, with visibility-gated keypoint smoothing — invisible
    keypoints hold their last position while their score decays toward
    the visibility threshold so the overlay eventually stops drawing
    a stale limb.

Both trackers smooth in **letterbox coordinates** (the native output
space of the YOLO decoders) rather than frame coordinates, so that the
clipping in :meth:`HailoPoseInference._transform_raw_detection` /
:meth:`HailoObjectInference._transform_raw_detection` does not bias the
EMA inward when an object is partially off-screen.

Time-based miss decay (``max_miss_age_s``) instead of frame-based misses
keeps coasting behavior stable even when the HailoRT scheduler time-
slices unevenly between the two workers — the tracker doesn't care
whether the worker ran 18 Hz or 25 Hz this second.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .hand_types import HandLandmarks, NUM_HAND_KEYPOINTS
from .postprocess import Keypoint as _Keypoint, RawDetection
from .postprocess_yolo26 import COCO_CLASSES, RawObjectDetection


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _bbox_iou(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """Intersection-over-union of two xyxy bboxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _ema(old: float, new: float, alpha: float) -> float:
    """Exponential moving average. ``alpha`` is the weight on the new value."""
    return (1.0 - alpha) * old + alpha * new


def _greedy_match(
    track_bboxes: List[Tuple[float, float, float, float]],
    det_bboxes: List[Tuple[float, float, float, float]],
    iou_threshold: float,
    track_class_ids: Optional[List[int]] = None,
    det_class_ids: Optional[List[int]] = None,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Greedy IOU matching with optional class-equality tiebreaker.

    Builds an IOU matrix and picks the highest-IOU pair repeatedly,
    masking out used tracks/detections. When ``track_class_ids`` and
    ``det_class_ids`` are provided, ties are broken in favor of
    same-class matches by adding a tiny bonus to the IOU score, which
    keeps couch/bed-style overlapping detections stable.

    Returns ``(matches, unmatched_track_idx, unmatched_det_idx)`` where
    ``matches`` is a list of ``(track_idx, det_idx)`` pairs.
    """
    n_tracks = len(track_bboxes)
    n_dets = len(det_bboxes)
    if n_tracks == 0 or n_dets == 0:
        return ([], list(range(n_tracks)), list(range(n_dets)))

    t = np.array(track_bboxes, dtype=np.float32)  # (N, 4)
    d = np.array(det_bboxes, dtype=np.float32)    # (M, 4)
    x1 = np.maximum(t[:, None, 0], d[None, :, 0])
    y1 = np.maximum(t[:, None, 1], d[None, :, 1])
    x2 = np.minimum(t[:, None, 2], d[None, :, 2])
    y2 = np.minimum(t[:, None, 3], d[None, :, 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area_t = (t[:, 2] - t[:, 0]) * (t[:, 3] - t[:, 1])
    area_d = (d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1])
    union = area_t[:, None] + area_d[None, :] - inter
    iou_matrix = np.where(union > 0, inter / union, np.float32(0)).astype(np.float32)

    # Class tiebreaker: same-class pairs get a small bonus that only matters
    # when two pairs are within ~1e-3 IOU of each other.
    if track_class_ids is not None and det_class_ids is not None:
        t_cls = np.array(track_class_ids)
        d_cls = np.array(det_class_ids)
        same_class = (t_cls[:, None] == d_cls[None, :])
        bonus = same_class & (iou_matrix >= iou_threshold)
        iou_matrix[bonus] += 1e-3

    matches: List[Tuple[int, int]] = []
    used_tracks: set = set()
    used_dets: set = set()

    flat = iou_matrix.flatten()
    order = np.argsort(-flat)  # descending
    for idx in order:
        score = flat[idx]
        if score < iou_threshold:
            break
        i, j = divmod(int(idx), n_dets)
        if i in used_tracks or j in used_dets:
            continue
        matches.append((i, j))
        used_tracks.add(i)
        used_dets.add(j)

    unmatched_tracks = [i for i in range(n_tracks) if i not in used_tracks]
    unmatched_dets = [j for j in range(n_dets) if j not in used_dets]
    return matches, unmatched_tracks, unmatched_dets


# ---------------------------------------------------------------------------
# Lifecycle state shared between tracker types
# ---------------------------------------------------------------------------


@dataclass
class _TrackLifecycle:
    """Hits/misses/confirmation state shared by every tracked object.

    Stored as a separate dataclass so the same lifecycle logic
    (``update_on_match`` / ``update_on_miss`` / ``should_confirm`` /
    ``should_drop``) is reused by both ``_ObjectTrack`` and
    ``_PersonTrack``. The reasoning: when a lifecycle bug surfaces,
    it should be fixed in one place.
    """

    track_id: int
    age: int = 0
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    last_update_s: float = 0.0
    prev_cx: float = 0.0  # stored for future adaptive-alpha; unused in v1
    prev_cy: float = 0.0

    def update_on_match(self, now_s: float) -> None:
        self.age += 1
        self.hits += 1
        self.misses = 0
        self.last_update_s = now_s

    def update_on_miss(self) -> None:
        self.age += 1
        self.misses += 1

    def should_confirm(self, min_hits: int) -> bool:
        return self.hits >= min_hits

    def should_drop(self, now_s: float, max_miss_age_s: float) -> bool:
        return (now_s - self.last_update_s) >= max_miss_age_s


# ---------------------------------------------------------------------------
# Object tracker
# ---------------------------------------------------------------------------


@dataclass
class _ObjectTrack:
    """Internal tracker state for one yolo26 detection across time.

    Coordinates are in **letterbox space** (0..640 typically), not frame
    space. Conversion to frame space happens after the tracker, in
    :meth:`HailoObjectInference._transform_raw_detection`.
    """

    life: _TrackLifecycle
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    recent_classes: Deque[int]


class ObjectTracker:
    """Temporal tracker for yolo26m object detections."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        smoothing_alpha: float = 0.4,
        min_hits_to_confirm: int = 3,
        max_miss_age_s: float = 0.4,
        class_window: int = 5,
        class_flip_count: int = 4,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.smoothing_alpha = smoothing_alpha
        self.min_hits_to_confirm = min_hits_to_confirm
        self.max_miss_age_s = max_miss_age_s
        self.class_window = class_window
        self.class_flip_count = class_flip_count
        self._next_id = 1
        self._tracks: List[_ObjectTrack] = []

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 1

    def update(
        self,
        detections: List[RawObjectDetection],
        now_s: float,
    ) -> List[RawObjectDetection]:
        """Match, smooth, age. Returns confirmed tracks as raw detections.

        The returned ``RawObjectDetection`` objects carry smoothed bbox
        coordinates and a stable ``class_id``; tentative tracks (still in
        the confirmation hysteresis) are not returned, so the overlay
        never sees a one-frame false positive.
        """
        track_bboxes = [(t.x1, t.y1, t.x2, t.y2) for t in self._tracks]
        det_bboxes = [(d.x1, d.y1, d.x2, d.y2) for d in detections]
        track_classes = [t.class_id for t in self._tracks]
        det_classes = [d.class_id for d in detections]

        matches, unmatched_t, unmatched_d = _greedy_match(
            track_bboxes,
            det_bboxes,
            iou_threshold=self.iou_threshold,
            track_class_ids=track_classes,
            det_class_ids=det_classes,
        )

        a = self.smoothing_alpha
        for ti, di in matches:
            track = self._tracks[ti]
            det = detections[di]
            track.x1 = _ema(track.x1, det.x1, a)
            track.y1 = _ema(track.y1, det.y1, a)
            track.x2 = _ema(track.x2, det.x2, a)
            track.y2 = _ema(track.y2, det.y2, a)
            track.score = _ema(track.score, det.score, a)
            track.recent_classes.append(det.class_id)
            # Class-switch hysteresis: flip only if the modal class in the
            # rolling window has been seen ``class_flip_count`` times.
            counts = Counter(track.recent_classes)
            modal_class, modal_count = counts.most_common(1)[0]
            if (
                modal_class != track.class_id
                and modal_count >= self.class_flip_count
            ):
                track.class_id = modal_class
            track.life.update_on_match(now_s)

        for ti in unmatched_t:
            self._tracks[ti].life.update_on_miss()

        for di in unmatched_d:
            det = detections[di]
            self._tracks.append(
                _ObjectTrack(
                    life=_TrackLifecycle(
                        track_id=self._next_id,
                        last_update_s=now_s,
                    ),
                    x1=det.x1,
                    y1=det.y1,
                    x2=det.x2,
                    y2=det.y2,
                    score=det.score,
                    class_id=det.class_id,
                    recent_classes=deque([det.class_id], maxlen=self.class_window),
                )
            )
            self._next_id += 1

        # Drop tracks that have not been updated within max_miss_age_s.
        # Anything that has missed at least once and is past the age limit
        # is removed; never-missed tracks always survive this pass.
        self._tracks = [
            t for t in self._tracks
            if t.life.misses == 0 or not t.life.should_drop(now_s, self.max_miss_age_s)
        ]

        # Promote tentative tracks once they have enough hits.
        out: List[RawObjectDetection] = []
        for track in self._tracks:
            if not track.life.confirmed and track.life.should_confirm(
                self.min_hits_to_confirm
            ):
                track.life.confirmed = True
            if track.life.confirmed:
                out.append(
                    RawObjectDetection(
                        x1=track.x1,
                        y1=track.y1,
                        x2=track.x2,
                        y2=track.y2,
                        score=track.score,
                        class_id=track.class_id,
                    )
                )
        return out


# ---------------------------------------------------------------------------
# Person tracker
# ---------------------------------------------------------------------------


@dataclass
class _PersonTrack:
    life: _TrackLifecycle
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    kpts_x: np.ndarray  # shape (17,)
    kpts_y: np.ndarray
    kpts_score: np.ndarray


class PersonTracker:
    """Temporal tracker for yolov8 pose detections with 17 keypoints.

    Smoothing strategy:

      - **bbox**: standard EMA with ``smoothing_alpha_bbox`` (slightly
        more aggressive than the object tracker because pose bboxes are
        usually larger and steadier — flicker comes from keypoints).
      - **per-keypoint EMA**: each of 17 keypoints gets its own EMA
        with ``smoothing_alpha_kpt``. The smoothed score is what
        ``overlay.py`` checks against ``KEYPOINT_SCORE_THRESHOLD``.
      - **visibility gating**: if the new raw keypoint score is below
        ``kpt_visible_threshold``, the position is **not** updated
        (held at last visible value), but the score is decayed by
        ``kpt_score_decay`` so the keypoint eventually fades from the
        overlay rather than appearing frozen.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        smoothing_alpha_bbox: float = 0.4,
        smoothing_alpha_kpt: float = 0.6,
        kpt_visible_threshold: float = 0.3,
        kpt_score_decay: float = 0.85,
        min_hits_to_confirm: int = 2,
        max_miss_age_s: float = 0.3,
        num_keypoints: int = 17,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.smoothing_alpha_bbox = smoothing_alpha_bbox
        self.smoothing_alpha_kpt = smoothing_alpha_kpt
        self.kpt_visible_threshold = kpt_visible_threshold
        self.kpt_score_decay = kpt_score_decay
        self.min_hits_to_confirm = min_hits_to_confirm
        self.max_miss_age_s = max_miss_age_s
        self.num_keypoints = num_keypoints
        self._next_id = 1
        self._tracks: List[_PersonTrack] = []

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 1

    def update(
        self,
        detections: List[RawDetection],
        now_s: float,
    ) -> List[RawDetection]:
        track_bboxes = [(t.x1, t.y1, t.x2, t.y2) for t in self._tracks]
        det_bboxes = [(d.x1, d.y1, d.x2, d.y2) for d in detections]
        # No class hysteresis for person tracker — only one class.
        matches, unmatched_t, unmatched_d = _greedy_match(
            track_bboxes,
            det_bboxes,
            iou_threshold=self.iou_threshold,
        )

        ab = self.smoothing_alpha_bbox
        ak = self.smoothing_alpha_kpt
        for ti, di in matches:
            track = self._tracks[ti]
            det = detections[di]
            track.x1 = _ema(track.x1, det.x1, ab)
            track.y1 = _ema(track.y1, det.y1, ab)
            track.x2 = _ema(track.x2, det.x2, ab)
            track.y2 = _ema(track.y2, det.y2, ab)
            track.score = _ema(track.score, det.score, ab)

            det_x = np.array([kp.x for kp in det.keypoints], dtype=np.float32)
            det_y = np.array([kp.y for kp in det.keypoints], dtype=np.float32)
            det_s = np.array([kp.score for kp in det.keypoints], dtype=np.float32)
            visible = det_s >= self.kpt_visible_threshold

            track.kpts_x[visible] += ak * (det_x[visible] - track.kpts_x[visible])
            track.kpts_y[visible] += ak * (det_y[visible] - track.kpts_y[visible])
            track.kpts_score[visible] += ak * (det_s[visible] - track.kpts_score[visible])

            inv = ~visible
            track.kpts_score[inv] *= self.kpt_score_decay
            floor_val = np.float32(self.kpt_visible_threshold * 0.5)
            track.kpts_score[inv] = np.maximum(track.kpts_score[inv], floor_val)

            track.life.update_on_match(now_s)

        for ti in unmatched_t:
            self._tracks[ti].life.update_on_miss()
            # Decay all keypoint scores when the whole person is missed,
            # so a fully-occluded person fades naturally.
            self._tracks[ti].kpts_score *= self.kpt_score_decay

        for di in unmatched_d:
            det = detections[di]
            kpts_x = np.array([kp.x for kp in det.keypoints], dtype=np.float32)
            kpts_y = np.array([kp.y for kp in det.keypoints], dtype=np.float32)
            kpts_s = np.array([kp.score for kp in det.keypoints], dtype=np.float32)
            self._tracks.append(
                _PersonTrack(
                    life=_TrackLifecycle(
                        track_id=self._next_id,
                        last_update_s=now_s,
                    ),
                    x1=det.x1,
                    y1=det.y1,
                    x2=det.x2,
                    y2=det.y2,
                    score=det.score,
                    kpts_x=kpts_x,
                    kpts_y=kpts_y,
                    kpts_score=kpts_s,
                )
            )
            self._next_id += 1

        self._tracks = [
            t for t in self._tracks
            if t.life.misses == 0 or not t.life.should_drop(now_s, self.max_miss_age_s)
        ]

        out: List[RawDetection] = []
        for track in self._tracks:
            if not track.life.confirmed and track.life.should_confirm(
                self.min_hits_to_confirm
            ):
                track.life.confirmed = True
            if track.life.confirmed:
                kps = tuple(
                    _Keypoint(
                        x=float(track.kpts_x[k]),
                        y=float(track.kpts_y[k]),
                        score=float(track.kpts_score[k]),
                    )
                    for k in range(self.num_keypoints)
                )
                out.append(
                    RawDetection(
                        x1=track.x1,
                        y1=track.y1,
                        x2=track.x2,
                        y2=track.y2,
                        score=track.score,
                        keypoints=kps,
                    )
                )
        return out


# ---------------------------------------------------------------------------
# Hand tracker
# ---------------------------------------------------------------------------


@dataclass
class _HandTrack:
    """Internal hand-tracker state keyed by ``(person_id, side)``.

    Coordinates live in **frame pixel space**, not crop or letterbox
    space: each crop has its own affine and is re-extracted every frame,
    so there is no stable reference frame other than the full video
    frame. EMA smoothing in frame space is correct because the landmark
    positions we're averaging are all projected to the same coordinate
    system across frames.
    """

    life: _TrackLifecycle
    kpts: np.ndarray        # shape (21, 2) float32, frame pixel coords
    kpt_scores: np.ndarray  # shape (21,)  float32
    presence: float


class HandTracker:
    """Temporal smoother for 21-point hand landmarks.

    Identity is keyed by ``(person_id, side)`` tuple, where ``person_id``
    comes from the upstream pose tracker and ``side`` is ``"left"`` or
    ``"right"`` from the wrist anchor in
    :class:`pi_streamer.hand_roi.WristAnchoredROISource`. Because the
    identity comes from an external source, this tracker does *not* do
    IOU matching — it's a per-key smoother.

    Smoothing strategy:

      - **Per-landmark EMA** on (x, y) and presence score with
        ``smoothing_alpha`` (slightly lower default than PersonTracker's
        keypoint alpha because hand landmark model output is less noisy
        than pose-head keypoint output).
      - **Presence gating** on the render side: ``kpt_scores`` is populated
        uniformly with the tracked presence so overlay.py's standard
        ``score >= threshold`` check falls through naturally.
      - **Confirmation hysteresis** (min 2 consecutive hits) suppresses
        single-frame false positives.
      - **Time-based miss decay**: if a ``(person_id, side)`` key is not
        updated for ``max_miss_age_s``, the track is dropped so a stale
        hand does not keep rendering after a person walks out of frame.
    """

    def __init__(
        self,
        smoothing_alpha: float = 0.5,
        min_hits_to_confirm: int = 2,
        max_miss_age_s: float = 0.25,
    ) -> None:
        self.smoothing_alpha = smoothing_alpha
        self.min_hits_to_confirm = min_hits_to_confirm
        self.max_miss_age_s = max_miss_age_s
        self._next_id = 1
        self._tracks: Dict[Tuple[int, str], _HandTrack] = {}

    def reset(self) -> None:
        self._tracks = {}
        self._next_id = 1

    def update(
        self,
        hands: List[HandLandmarks],
        now_s: float,
    ) -> List[HandLandmarks]:
        """Update tracker state with this frame's raw hands.

        Returns a list of *confirmed* smoothed ``HandLandmarks`` (only
        tracks that passed the confirmation hysteresis). Tentative tracks
        are held internally but not returned, so the overlay never sees
        a one-frame false positive.
        """
        a = self.smoothing_alpha

        # Step 1: ingest this frame's hands into the keyed registry.
        seen_keys: set = set()
        for hand in hands:
            key = (hand.person_id, hand.side)
            seen_keys.add(key)
            track = self._tracks.get(key)
            if track is None:
                # New track: initialize from raw.
                self._tracks[key] = _HandTrack(
                    life=_TrackLifecycle(
                        track_id=self._next_id,
                        last_update_s=now_s,
                    ),
                    kpts=np.array(hand.kpts, dtype=np.float32, copy=True),
                    kpt_scores=np.array(
                        hand.kpt_scores, dtype=np.float32, copy=True
                    ),
                    presence=float(hand.presence),
                )
                self._next_id += 1
                continue

            # Existing track: EMA the 21 landmarks and the presence score.
            track.kpts += a * (hand.kpts.astype(np.float32, copy=False) - track.kpts)
            track.kpt_scores += a * (hand.kpt_scores.astype(np.float32, copy=False) - track.kpt_scores)
            track.presence = _ema(track.presence, float(hand.presence), a)
            track.life.update_on_match(now_s)

        # Step 2: age any tracks not seen this frame.
        for key, track in self._tracks.items():
            if key not in seen_keys:
                track.life.update_on_miss()

        # Step 3: drop tracks past their max miss age. Never-missed tracks
        # always survive (misses==0 gate).
        to_drop = [
            key for key, t in self._tracks.items()
            if t.life.misses > 0 and t.life.should_drop(now_s, self.max_miss_age_s)
        ]
        for key in to_drop:
            del self._tracks[key]

        # Step 4: emit confirmed tracks as smoothed HandLandmarks.
        out: List[HandLandmarks] = []
        for (person_id, side), track in self._tracks.items():
            if not track.life.confirmed and track.life.should_confirm(
                self.min_hits_to_confirm
            ):
                track.life.confirmed = True
            if not track.life.confirmed:
                continue
            out.append(
                HandLandmarks(
                    kpts=track.kpts.copy(),
                    kpt_scores=track.kpt_scores.copy(),
                    presence=track.presence,
                    person_id=person_id,
                    side=side,
                )
            )
        return out


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


def _self_test() -> None:
    """Lightweight self-tests with no Hailo dependency.

    Run with::

        python -m pi_streamer.tracker
    """

    # ----- _bbox_iou -----
    assert abs(_bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 1e-6
    assert abs(_bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) - 0.0) < 1e-6
    # Two 10x10 boxes overlapping in 5x5 → inter=25, area=100 each, union=175
    iou = _bbox_iou((0, 0, 10, 10), (5, 5, 15, 15))
    assert abs(iou - 25.0 / 175.0) < 1e-6, f"iou={iou}"

    # ----- _greedy_match -----
    matches, ut, ud = _greedy_match([], [], 0.3)
    assert matches == [] and ut == [] and ud == []

    matches, ut, ud = _greedy_match(
        [(0, 0, 10, 10), (100, 100, 120, 120)],
        [(1, 1, 11, 11), (101, 101, 121, 121), (200, 200, 210, 210)],
        iou_threshold=0.3,
    )
    assert (0, 0) in matches and (1, 1) in matches
    assert ut == [] and ud == [2]

    # Class tiebreaker: equal IOU, same-class pair wins
    matches, _, _ = _greedy_match(
        [(0, 0, 10, 10)],
        [(0, 0, 10, 10), (0, 0, 10, 10)],
        iou_threshold=0.3,
        track_class_ids=[5],
        det_class_ids=[7, 5],
    )
    # Both candidates have IOU 1.0; the same-class one (idx=1) should win.
    assert matches == [(0, 1)], f"matches={matches}"

    # ----- _TrackLifecycle -----
    life = _TrackLifecycle(track_id=1, last_update_s=0.0)
    life.update_on_match(now_s=0.1)
    life.update_on_match(now_s=0.2)
    life.update_on_match(now_s=0.3)
    assert life.hits == 4 and life.misses == 0
    assert life.should_confirm(min_hits=3)
    assert not life.should_drop(now_s=0.4, max_miss_age_s=0.5)
    life.update_on_miss()
    assert life.misses == 1
    assert life.should_drop(now_s=1.0, max_miss_age_s=0.5)

    # ----- ObjectTracker end-to-end -----
    tracker = ObjectTracker(
        iou_threshold=0.3,
        smoothing_alpha=0.5,
        min_hits_to_confirm=3,
        max_miss_age_s=0.5,
        class_window=5,
        class_flip_count=4,
    )

    # Stable bbox over 5 frames at the same coords; should become confirmed
    # after the 3rd update and remain stable thereafter.
    detection = RawObjectDetection(x1=100, y1=100, x2=200, y2=200, score=0.8, class_id=15)
    out = tracker.update([detection], now_s=0.0)
    assert out == [], f"frame 1: tentative, expected empty, got {out}"
    out = tracker.update([detection], now_s=0.05)
    assert out == [], f"frame 2: still tentative, got {out}"
    out = tracker.update([detection], now_s=0.10)
    assert len(out) == 1, f"frame 3: should be confirmed, got {len(out)}"
    confirmed = out[0]
    assert confirmed.class_id == 15
    assert abs(confirmed.x1 - 100) < 0.5 and abs(confirmed.x2 - 200) < 0.5

    # Jittered detection: tracker should smooth toward the average position
    jittered = RawObjectDetection(x1=110, y1=100, x2=210, y2=200, score=0.8, class_id=15)
    out = tracker.update([jittered], now_s=0.15)
    assert len(out) == 1
    # With alpha=0.5, x1 should be (100 + 110)/2 = 105, not jumped to 110.
    assert abs(out[0].x1 - 105.0) < 0.5, f"x1={out[0].x1}"

    # Class-switch hysteresis: 2 frames with class 28 (suitcase) should NOT
    # flip the label (need 4 of 5 in rolling window).
    other_class = RawObjectDetection(x1=110, y1=100, x2=210, y2=200, score=0.8, class_id=28)
    tracker.update([other_class], now_s=0.20)
    out = tracker.update([other_class], now_s=0.25)
    # recent_classes deque is now [15, 15, 15, 28, 28] — modal is 15 with count 3
    assert out[0].class_id == 15, f"expected 15 (no flip), got {out[0].class_id}"
    # Two more class-28 frames push the deque to [15, 28, 28, 28, 28] — modal is 28 with count 4
    tracker.update([other_class], now_s=0.30)
    out = tracker.update([other_class], now_s=0.35)
    assert out[0].class_id == 28, f"expected 28 (flipped), got {out[0].class_id}"

    # Coasting: missing one update should keep the track alive
    out = tracker.update([], now_s=0.40)
    # No matched detections this frame, but the track has misses=1 within
    # max_miss_age_s, so it survives. It is also not returned because it
    # was not updated this frame; that's fine.
    out = tracker.update([other_class], now_s=0.45)
    assert len(out) == 1, "track should be re-matched, not duplicated"

    # Long absence: drop the track
    out = tracker.update([], now_s=2.0)
    out = tracker.update([], now_s=2.5)
    out = tracker.update([], now_s=3.0)
    assert out == [], f"track should be dropped, got {out}"
    assert len(tracker._tracks) == 0

    # ----- PersonTracker end-to-end -----
    ptracker = PersonTracker(
        iou_threshold=0.3,
        smoothing_alpha_bbox=0.5,
        smoothing_alpha_kpt=0.5,
        kpt_visible_threshold=0.3,
        kpt_score_decay=0.85,
        min_hits_to_confirm=2,
        max_miss_age_s=0.3,
    )

    def _mk_person(x_off=0.0, kpt_score=0.9):
        kps = tuple(
            _Keypoint(x=float(50 + x_off + i * 5), y=float(60 + i * 5), score=kpt_score)
            for i in range(17)
        )
        return RawDetection(
            x1=40.0 + x_off, y1=40.0,
            x2=200.0 + x_off, y2=300.0,
            score=0.9, keypoints=kps,
        )

    out = ptracker.update([_mk_person()], now_s=0.0)
    assert out == [], "frame 1: tentative"
    out = ptracker.update([_mk_person()], now_s=0.05)
    assert len(out) == 1, "frame 2: confirmed (min_hits_to_confirm=2)"
    person = out[0]
    assert len(person.keypoints) == 17

    # Jittered person bbox + keypoints — smoothing toward average
    out = ptracker.update([_mk_person(x_off=10.0)], now_s=0.10)
    assert len(out) == 1
    # alpha 0.5: x1 was 40, now 50 → smoothed to 45
    assert abs(out[0].x1 - 45.0) < 0.5

    # Invisible keypoint: position must NOT update, score must decay.
    invisible = _mk_person(x_off=10.0, kpt_score=0.05)
    out = ptracker.update([invisible], now_s=0.15)
    # Bbox is still smoothed normally.
    assert len(out) == 1
    # All 17 keypoint scores should now be decayed (multiplied by 0.85),
    # not EMA'd toward 0.05.
    last_kpt_score = out[0].keypoints[0].score
    # After two visible frames with score=0.9 (smoothed to ~0.7), then decay
    # by 0.85 → ~0.595. Definitely > 0.05 (the invisible reading).
    assert last_kpt_score > 0.4, f"keypoint score={last_kpt_score}"

    # ----- reset() clears state -----
    tracker.reset()
    ptracker.reset()
    assert tracker._tracks == [] and ptracker._tracks == []
    assert tracker._next_id == 1 and ptracker._next_id == 1

    # ----- HandTracker end-to-end -----
    htracker = HandTracker(
        smoothing_alpha=0.5,
        min_hits_to_confirm=2,
        max_miss_age_s=0.25,
    )

    def _mk_hand(person_id: int, side: str, x_off: float = 0.0, presence: float = 0.9):
        kpts = np.array(
            [[100 + x_off + i * 3, 200 + i * 2] for i in range(NUM_HAND_KEYPOINTS)],
            dtype=np.float32,
        )
        scores = np.full((NUM_HAND_KEYPOINTS,), presence, dtype=np.float32)
        return HandLandmarks(
            kpts=kpts,
            kpt_scores=scores,
            presence=presence,
            person_id=person_id,
            side=side,
        )

    # frame 1: tentative (confirmation hysteresis is 2 hits)
    out = htracker.update([_mk_hand(0, "left"), _mk_hand(0, "right")], now_s=0.0)
    assert out == [], f"frame 1: tentative, got {len(out)}"

    # frame 2: confirmed
    out = htracker.update([_mk_hand(0, "left"), _mk_hand(0, "right")], now_s=0.05)
    assert len(out) == 2, f"frame 2: two hands confirmed, got {len(out)}"
    sides = sorted(h.side for h in out)
    assert sides == ["left", "right"], sides

    # frame 3: jittered landmarks — smoothing should average toward prior position
    out = htracker.update(
        [_mk_hand(0, "left", x_off=10.0), _mk_hand(0, "right", x_off=10.0)],
        now_s=0.10,
    )
    left = next(h for h in out if h.side == "left")
    # alpha=0.5: kpts[0,0] was ~100, new is 110 → smoothed to ~105
    assert abs(float(left.kpts[0, 0]) - 105.0) < 0.5, f"x={left.kpts[0, 0]}"

    # frame 4: drop left hand — right persists, left starts aging
    out = htracker.update([_mk_hand(0, "right", x_off=10.0)], now_s=0.15)
    right_kpt0_x = float(next(h for h in out if h.side == "right").kpts[0, 0])
    # right should still be around 107-110 (smoothed)
    assert right_kpt0_x > 100.0, f"right x={right_kpt0_x}"

    # frame 5 (t=0.30): left hand has been missing 0.15s, still under 0.25s max_miss_age
    # → not dropped yet; still in registry as unconfirmed (misses=1)
    out = htracker.update([_mk_hand(0, "right", x_off=10.0)], now_s=0.30)
    assert (0, "left") in htracker._tracks, "left should still be aging, not dropped"

    # frame 6 (t=0.45): left has been missing 0.30s, past 0.25s → dropped
    out = htracker.update([_mk_hand(0, "right", x_off=10.0)], now_s=0.45)
    assert (0, "left") not in htracker._tracks, "left should be dropped after 0.25s"

    # Per-person separation: person 1 left hand should NOT collide with person 0 left.
    out = htracker.update(
        [
            _mk_hand(0, "right", x_off=10.0),
            _mk_hand(1, "left", x_off=500.0),
            _mk_hand(1, "right", x_off=500.0),
        ],
        now_s=0.50,
    )
    keys = set(htracker._tracks.keys())
    assert (0, "right") in keys and (1, "left") in keys and (1, "right") in keys

    htracker.reset()
    assert htracker._tracks == {} and htracker._next_id == 1

    print("tracker self-test PASS")


if __name__ == "__main__":
    _self_test()
