"""Hand region-of-interest (ROI) sources.

Extracts 224×224 crops around hands from an incoming video frame, together
with the per-crop affine that lets :class:`HailoHandInference` back-project
the model's output from crop pixel coordinates to frame pixel coordinates.

Two implementations are planned; only :class:`WristAnchoredROISource`
(Plan A) is provided in this module. A future Plan B will add
``PalmDetectorROISource`` that uses a recompiled ``palm_detection_lite``
HEF to find hand regions directly on the full frame. Both will conform
to the same :class:`HandROISource` ``Protocol`` so the hand inference
worker is agnostic to which source produced its crops.

``WristAnchoredROISource`` builds each crop from COCO-17 body keypoints:

  - reads the ``LEFT_WRIST`` / ``RIGHT_WRIST`` and the matching
    ``LEFT_ELBOW`` / ``RIGHT_ELBOW`` per detected person
  - gates on the wrist and elbow confidence (``min_wrist_score`` /
    ``min_elbow_score``) so low-confidence body joints don't spawn
    garbage crops
  - estimates the forearm vector ``w - e`` and computes the crop center
    as ``wrist + shift_fraction * forearm`` so the crop sits slightly
    past the wrist toward the fingers
  - sizes the crop as ``size_scale * forearm_length`` expanded to a
    square
  - warps the source frame into a 224×224 BGR patch with
    ``cv2.warpAffine``
  - stores the inverse affine (crop→frame) on the resulting
    :class:`CropInfo` so the decoder's output can be back-projected

The heuristic is approximate — the MediaPipe holistic pipeline uses
three MLPs to refine palm center/size/rotation from pose keypoints and
reports IoU improvements from 57% → 63% on rotated hands — but the
simple heuristic is fine for axis-aligned front-facing hands in the
common case. If post-return validation shows poor landmark quality on
tilted hands, a rotated-crop variant (aligning the crop's vertical axis
with the forearm direction) is the natural next step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Literal, Optional, Protocol, Sequence, Tuple

import numpy as np


# COCO-17 keypoint indices used for hand ROI extraction. Must match the
# layout produced by yolov8m_pose (which follows the Ultralytics / COCO
# keypoint order).
COCO_LEFT_SHOULDER = 5
COCO_RIGHT_SHOULDER = 6
COCO_LEFT_ELBOW = 7
COCO_RIGHT_ELBOW = 8
COCO_LEFT_WRIST = 9
COCO_RIGHT_WRIST = 10


@dataclass
class CropInfo:
    """A single 224×224 hand crop ready for hand_landmark_lite inference.

    Produced by :class:`HandROISource.extract_rois`, consumed by
    :class:`pi_streamer.inference.HailoHandInference`. The ``affine``
    field is the **inverse** of the frame→crop warp used by
    ``cv2.warpAffine`` — i.e., it maps crop pixel coordinates back to
    frame pixel coordinates so ``HandLandmarkDecoder`` output can be
    projected onto the full frame.

    ``wrist_xy`` is the original pose wrist position (in frame pixel
    coordinates) that anchored this crop. It is used by the hand
    inference worker's one-shot back-projection sanity check to verify
    that the MediaPipe wrist landmark (index 0) lands near the pose
    wrist that anchored the crop. If that check fails, the affine math
    is wrong and no presence fix will help.
    """

    pixels: np.ndarray                 # shape (224, 224, 3) uint8 BGR
    affine: np.ndarray                 # shape (2, 3) float32, crop→frame
    person_id: int                     # from pose tracker
    side: Literal["left", "right"]
    wrist_xy: Tuple[float, float] = (0.0, 0.0)  # pose wrist anchor, frame px
    source: Literal["wrist", "palm"] = "wrist"


class HandROISource(Protocol):
    """Abstract source of hand crops for a given video frame."""

    def extract_rois(
        self,
        frame: np.ndarray,
        pose_persons: Sequence,
    ) -> List[CropInfo]:
        """Return a list of ``CropInfo`` for the current frame, possibly empty."""
        ...


class WristAnchoredROISource:
    """Plan A: build hand crops from pose wrist+elbow keypoints.

    The current defaults (size_scale=1.80, shift_fraction=0.55,
    min_wrist_score=0.30) came out of tuning after a pass with tighter
    crops failed to render hand skeletons reliably.

    When ``rotate`` is True (the default), the crop's vertical axis
    aligns with the forearm direction (elbow → wrist → fingers), so the
    hand ends up standing up in the 224×224 patch. That matches the
    MediaPipe ``hand_landmark_lite`` training distribution far better
    than an axis-aligned crop — the landmark model was trained on
    palm-centered crops from MediaPipe's own palm detector, which always
    produces oriented boxes. Pass ``rotate=False`` (or
    ``--no-hand-rotate`` on the CLI) to fall back to axis-aligned crops
    for A/B comparison.
    """

    def __init__(
        self,
        target_size: int = 224,
        min_wrist_score: float = 0.30,
        min_elbow_score: float = 0.30,
        size_scale: float = 1.80,
        shift_fraction: float = 0.55,
        max_crops_per_frame: int = 4,
        min_forearm_pixels: float = 12.0,
        rotate: bool = True,
    ) -> None:
        self.target_size = target_size
        self.min_wrist_score = min_wrist_score
        self.min_elbow_score = min_elbow_score
        self.size_scale = size_scale
        self.shift_fraction = shift_fraction
        self.max_crops_per_frame = max_crops_per_frame
        self.min_forearm_pixels = min_forearm_pixels
        self.rotate = rotate

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def extract_rois(
        self,
        frame: np.ndarray,
        pose_persons: Sequence,
    ) -> List[CropInfo]:
        """Build up to ``max_crops_per_frame`` hand crops from ``pose_persons``.

        ``pose_persons`` is the ``InferenceResult.persons`` tuple from
        the latest pose worker result — a sequence of ``PersonDetection``
        with a ``keypoints`` tuple of 17 ``Keypoint`` objects in frame
        pixel coordinates.

        Candidates are scored by ``min(wrist_score, elbow_score)`` and
        only the top ``max_crops_per_frame`` are materialized as crops,
        so a crowd of poorly-detected wrists does not spend the whole
        scheduler budget on bad crops.
        """
        import cv2  # lazy

        candidates = self._candidates(pose_persons)
        if not candidates:
            return []

        # Prefer higher-confidence wrists when there are more than
        # max_crops_per_frame candidates.
        candidates.sort(key=lambda c: c[5], reverse=True)
        candidates = candidates[: self.max_crops_per_frame]

        h, w = frame.shape[:2]
        out: List[CropInfo] = []
        for person_id, side, wrist_xy, elbow_xy, forearm_len, _score in candidates:
            crop_info = self._make_crop(
                cv2_mod=cv2,
                frame=frame,
                frame_w=w,
                frame_h=h,
                wrist_xy=wrist_xy,
                elbow_xy=elbow_xy,
                forearm_len=forearm_len,
                person_id=person_id,
                side=side,
            )
            if crop_info is not None:
                out.append(crop_info)
        return out

    # ------------------------------------------------------------------
    # Candidate selection
    # ------------------------------------------------------------------

    def _candidates(
        self,
        pose_persons: Sequence,
    ) -> List[tuple]:
        """Return ``(person_id, side, wrist, elbow, forearm_len, min_score)`` per candidate."""
        pairs = (
            ("left", COCO_LEFT_WRIST, COCO_LEFT_ELBOW),
            ("right", COCO_RIGHT_WRIST, COCO_RIGHT_ELBOW),
        )
        out: List[tuple] = []
        for person_id, person in enumerate(pose_persons):
            kpts = person.keypoints
            if len(kpts) < 11:  # need at least through COCO index 10 (right_wrist)
                continue
            for side, wrist_idx, elbow_idx in pairs:
                w_kp = kpts[wrist_idx]
                e_kp = kpts[elbow_idx]
                if w_kp.score < self.min_wrist_score:
                    continue
                if e_kp.score < self.min_elbow_score:
                    continue

                wx, wy = float(w_kp.x), float(w_kp.y)
                ex, ey = float(e_kp.x), float(e_kp.y)
                fx = wx - ex
                fy = wy - ey
                forearm_len = float(np.hypot(fx, fy))
                if forearm_len < self.min_forearm_pixels:
                    continue

                min_score = float(min(w_kp.score, e_kp.score))
                out.append(
                    (person_id, side, (wx, wy), (ex, ey), forearm_len, min_score)
                )
        return out

    # ------------------------------------------------------------------
    # Crop materialization
    # ------------------------------------------------------------------

    def _make_crop(
        self,
        cv2_mod,
        frame: np.ndarray,
        frame_w: int,
        frame_h: int,
        wrist_xy: tuple,
        elbow_xy: tuple,
        forearm_len: float,
        person_id: int,
        side: str,
    ) -> Optional[CropInfo]:
        """Warp a ``target_size``×``target_size`` crop from the frame.

        Center and size are derived from the forearm heuristic.
        ``cv2.warpAffine`` handles out-of-bounds reads by filling with
        0 (black border), which is fine for the hand_landmark model —
        its training distribution saw MediaPipe palm-detection crops
        with similar padding.

        If ``self.rotate`` is True (the default), the affine applies a
        rotation so the forearm vector ends up pointing "up" in the crop
        (crop y=0). Otherwise the affine is pure isotropic scale +
        translation, matching the older axis-aligned behaviour.
        """
        wx, wy = wrist_xy
        ex, ey = elbow_xy

        # Crop center: slightly past the wrist toward the fingers.
        cx = wx + self.shift_fraction * (wx - ex)
        cy = wy + self.shift_fraction * (wy - ey)

        # Crop side length (square).
        crop_size = max(
            self.min_forearm_pixels * self.size_scale,
            forearm_len * self.size_scale,
        )
        half = crop_size * 0.5

        scale = float(self.target_size) / float(crop_size)
        if self.rotate:
            # Rotate so the forearm vector (elbow→wrist→fingers) points
            # along the crop's negative-y axis ("up" in image space).
            #
            # The forearm direction in frame space is:
            #     v = (wx - ex, wy - ey)
            # Its angle in atan2 terms (using image-y-down values as if
            # they were math y) is:
            #     theta = atan2(wy - ey, wx - ex)
            # We want v rotated to angle -pi/2 so it points toward y=0
            # (top of image). The rotation amount is:
            #     phi = -pi/2 - theta
            # Applying the standard 2D rotation matrix R(phi) with
            # R(phi) = [[cos, -sin], [sin, cos]] to v yields (0, -|v|),
            # which is purely vertical pointing up in image space.
            #
            # Sign-convention sanity checks (verified in _self_test):
            #   horizontal-right forearm v=(1, 0):  theta=0,    phi=-pi/2
            #   vertical-up forearm v=(0,-1):       theta=-pi/2, phi=0
            #   diagonal up-right v=(1,-1):         theta=-pi/4, phi=-pi/4
            theta = math.atan2(wy - ey, wx - ex)
            phi = -math.pi / 2.0 - theta
            cos_phi = math.cos(phi)
            sin_phi = math.sin(phi)

            a = scale * cos_phi
            b = -scale * sin_phi
            c = scale * sin_phi
            d = scale * cos_phi
            # Translate so (cx, cy) in frame maps to the crop centre.
            tx = self.target_size * 0.5 - (a * cx + b * cy)
            ty = self.target_size * 0.5 - (c * cx + d * cy)
            frame_to_crop = np.array(
                [[a, b, tx],
                 [c, d, ty]],
                dtype=np.float32,
            )
        else:
            # Legacy axis-aligned path. Kept so --no-hand-rotate can A/B
            # against the pre-rotation behaviour.
            frame_to_crop = np.array(
                [
                    [scale, 0.0, self.target_size * 0.5 - scale * cx],
                    [0.0, scale, self.target_size * 0.5 - scale * cy],
                ],
                dtype=np.float32,
            )

        crop_px = cv2_mod.warpAffine(
            frame,
            frame_to_crop,
            (self.target_size, self.target_size),
            flags=cv2_mod.INTER_LINEAR,
            borderMode=cv2_mod.BORDER_CONSTANT,
            borderValue=0,
        )
        if crop_px.shape != (self.target_size, self.target_size, 3):
            return None

        # Inverse: crop→frame (2x3). For rotation+uniform-scale+translation,
        # cv2.invertAffineTransform still gives the exact closed-form inverse.
        crop_to_frame = cv2_mod.invertAffineTransform(frame_to_crop).astype(
            np.float32
        )

        # Reject degenerate crops that are entirely outside the frame.
        # We use the axis-aligned bounding box of the unrotated crop for
        # this check. With rotation, the AABB of the rotated crop is at
        # least as large, so a true "entirely off-screen" case is still
        # caught — the rotated crop just ends up with slightly more
        # black border near the edges, which is acceptable.
        if (cx + half) < 0 or (cy + half) < 0:
            return None
        if (cx - half) > frame_w or (cy - half) > frame_h:
            return None

        return CropInfo(
            pixels=crop_px,
            affine=crop_to_frame,
            person_id=person_id,
            side=side,  # type: ignore[arg-type]
            wrist_xy=(float(wx), float(wy)),
            source="wrist",
        )


# ---------------------------------------------------------------------------
# Self-test (run with ``python -m pi_streamer.hand_roi``)
# ---------------------------------------------------------------------------


def _self_test() -> None:
    """Lightweight self-test without a Hailo device.

    Uses small synthetic pose structs so we don't have to import the real
    ``Keypoint`` / ``PersonDetection`` dataclasses — the ROI source only
    touches ``.keypoints[i].x/.y/.score`` which is duck-typed.
    """
    from dataclasses import dataclass as _dc

    @_dc
    class _Kpt:
        x: float
        y: float
        score: float

    @_dc
    class _Person:
        keypoints: list

    def mk_person(wrist_l, elbow_l, wrist_r, elbow_r, score=0.9):
        kpts = [_Kpt(0.0, 0.0, 0.0)] * 17
        kpts[COCO_LEFT_WRIST] = _Kpt(*wrist_l, score)
        kpts[COCO_LEFT_ELBOW] = _Kpt(*elbow_l, score)
        kpts[COCO_RIGHT_WRIST] = _Kpt(*wrist_r, score)
        kpts[COCO_RIGHT_ELBOW] = _Kpt(*elbow_r, score)
        return _Person(keypoints=kpts)

    # Self-test pins the legacy size_scale=1.1 / shift_fraction=0.35 AND
    # rotate=False so the hardcoded expected crop centers below do not
    # drift every time the production defaults are retuned. Rotation
    # correctness is covered by the dedicated tests 8-11 further down.
    src = WristAnchoredROISource(
        target_size=224,
        min_wrist_score=0.3,
        min_elbow_score=0.3,
        size_scale=1.1,
        shift_fraction=0.35,
        max_crops_per_frame=4,
        rotate=False,
    )

    # Synthetic 1280x720 frame with known pattern: each 20x20 patch filled
    # with (row % 256, col % 256, 128) so we can verify the warp lands on
    # the right pixels.
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    for y in range(720):
        for x in range(1280):
            frame[y, x, 0] = (x // 8) & 0xFF
            frame[y, x, 1] = (y // 8) & 0xFF
            frame[y, x, 2] = 128

    # ---- test 1: one person, both hands visible ----
    person = mk_person(
        wrist_l=(300, 400), elbow_l=(250, 450),
        wrist_r=(800, 400), elbow_r=(750, 450),
    )
    rois = src.extract_rois(frame, [person])
    assert len(rois) == 2, f"expected 2 crops, got {len(rois)}"
    sides = sorted(r.side for r in rois)
    assert sides == ["left", "right"], sides
    for roi in rois:
        assert roi.pixels.shape == (224, 224, 3), roi.pixels.shape
        assert roi.pixels.dtype == np.uint8
        assert roi.affine.shape == (2, 3), roi.affine.shape
        assert roi.person_id == 0

    # ---- test 2: affine round-trip ----
    # Pick one crop's affine and verify the inverse of the inverse returns
    # approximately the original (cx, cy) when we project crop center
    # (112, 112) back to frame. With the expected crop geometry:
    #   forearm_len = sqrt(50^2 + 50^2) ≈ 70.71
    #   size = 70.71 * 1.1 ≈ 77.78
    #   scale = 224 / 77.78 ≈ 2.88
    #   shift = 0.35 * (wrist - elbow) = 0.35 * (50, -50) = (17.5, -17.5)
    #   center_l = (300 + 17.5, 400 - 17.5) = (317.5, 382.5)
    left_roi = next(r for r in rois if r.side == "left")
    crop_center = np.array([[112.0], [112.0], [1.0]], dtype=np.float32)
    frame_point = left_roi.affine @ crop_center
    assert abs(frame_point[0, 0] - 317.5) < 0.5, f"frame_x={frame_point[0, 0]}"
    assert abs(frame_point[1, 0] - 382.5) < 0.5, f"frame_y={frame_point[1, 0]}"

    # ---- test 3: low-confidence wrist gated out ----
    person_bad = mk_person(
        wrist_l=(300, 400), elbow_l=(250, 450),
        wrist_r=(800, 400), elbow_r=(750, 450),
        score=0.1,
    )
    rois = src.extract_rois(frame, [person_bad])
    assert rois == [], f"expected no crops for low-confidence person, got {len(rois)}"

    # ---- test 4: min_forearm gate ----
    person_close = mk_person(
        wrist_l=(300, 400), elbow_l=(302, 402),  # forearm ≈ 2.8 pixels
        wrist_r=(800, 400), elbow_r=(801, 401),  # forearm ≈ 1.4 pixels
    )
    rois = src.extract_rois(frame, [person_close])
    assert rois == [], f"expected no crops for tiny forearms, got {len(rois)}"

    # ---- test 5: max_crops_per_frame caps the output ----
    # Pinned to rotate=False so test only checks the count, not geometry.
    src_capped = WristAnchoredROISource(
        max_crops_per_frame=2,
        min_wrist_score=0.3,
        min_elbow_score=0.3,
        rotate=False,
    )
    people = [
        mk_person((100, 100), (80, 80), (200, 100), (220, 80)),
        mk_person((100, 200), (80, 180), (200, 200), (220, 180)),
    ]
    rois = src_capped.extract_rois(frame, people)
    assert len(rois) == 2, f"expected 2 crops (capped), got {len(rois)}"

    # ---- test 6: crop entirely off-screen is rejected ----
    off_person = mk_person(
        wrist_l=(-500, -500), elbow_l=(-520, -520),
        wrist_r=(-500, -500), elbow_r=(-520, -520),
    )
    rois = src.extract_rois(frame, [off_person])
    assert rois == [], f"expected no crops for off-screen hand, got {len(rois)}"

    # ---- test 7: empty input ----
    assert src.extract_rois(frame, []) == []

    # ------------------------------------------------------------------
    # Rotation tests
    # ------------------------------------------------------------------
    # These tests verify the math in _make_crop's rotate branch by
    # constructing synthetic poses with known forearm orientations,
    # extracting the ROI, and decomposing the resulting frame→crop
    # affine to read off the rotation angle and scale. We then
    # round-trip the crop centre (112, 112) through the inverse affine
    # to confirm it still lands at the palm-shifted frame position.
    rot_src = WristAnchoredROISource(
        target_size=224,
        min_wrist_score=0.3,
        min_elbow_score=0.3,
        size_scale=1.1,
        shift_fraction=0.35,
        max_crops_per_frame=4,
        rotate=True,
    )

    def _decompose(crop_to_frame_affine: np.ndarray) -> tuple:
        """Compute (phi_from_forward_matrix, scale_from_forward_matrix).

        We recover the forward (frame→crop) matrix from its inverse,
        then extract the rotation angle phi and the uniform scale s
        from the leading 2×2 block ``[[s*cos(phi), -s*sin(phi)],
        [s*sin(phi), s*cos(phi)]]``.
        """
        import cv2 as _cv2
        forward = _cv2.invertAffineTransform(crop_to_frame_affine)
        a = float(forward[0, 0])
        c = float(forward[1, 0])
        s = float(np.hypot(a, c))
        phi = float(np.arctan2(c, a))
        return phi, s

    # ---- test 8: vertical forearm (wrist above elbow) → phi ≈ 0 ----
    # Wrist at (400, 200), elbow at (400, 300). In image coords y grows
    # down, so the wrist is geometrically ABOVE the elbow. The forearm
    # vector is (0, -100) which is already "up". Rotation should be
    # the identity.
    person_v = mk_person(
        wrist_l=(400, 200), elbow_l=(400, 300),
        wrist_r=(900, 200), elbow_r=(900, 300),
    )
    rois_v = rot_src.extract_rois(frame, [person_v])
    assert len(rois_v) == 2, f"test 8: expected 2 crops, got {len(rois_v)}"
    for roi in rois_v:
        phi, s_mag = _decompose(roi.affine)
        assert abs(phi) < 1e-4, f"test 8 vertical: phi={phi} expected ~0"
    # forearm_len = 100, crop_size = 100*1.1 = 110, scale = 224/110 ≈ 2.0364
    expected_scale = 224.0 / 110.0
    phi_left, s_left = _decompose(rois_v[0].affine)
    assert abs(s_left - expected_scale) < 1e-3, (
        f"test 8 scale: got {s_left} expected {expected_scale}"
    )

    # ---- test 9: horizontal forearm (wrist right of elbow) → phi ≈ -pi/2 ----
    # Wrist at (400, 300), elbow at (300, 300). Forearm vector (100, 0)
    # points horizontally right. Rotation needed to make it point "up"
    # in the crop is -pi/2 (in image coords).
    person_h = mk_person(
        wrist_l=(400, 300), elbow_l=(300, 300),
        wrist_r=(900, 300), elbow_r=(800, 300),
    )
    rois_h = rot_src.extract_rois(frame, [person_h])
    assert len(rois_h) == 2, f"test 9: expected 2 crops, got {len(rois_h)}"
    for roi in rois_h:
        phi, _ = _decompose(roi.affine)
        assert abs(phi - (-math.pi / 2.0)) < 1e-4, (
            f"test 9 horizontal: phi={phi} expected ~{-math.pi / 2.0}"
        )

    # ---- test 10: diagonal 45° forearm (up-right) → phi ≈ -pi/4 ----
    # Wrist at (400, 200), elbow at (300, 300). Forearm vector (100, -100)
    # points up and to the right at 45° in image coords.
    person_d = mk_person(
        wrist_l=(400, 200), elbow_l=(300, 300),
        wrist_r=(900, 200), elbow_r=(800, 300),
    )
    rois_d = rot_src.extract_rois(frame, [person_d])
    assert len(rois_d) == 2, f"test 10: expected 2 crops, got {len(rois_d)}"
    for roi in rois_d:
        phi, _ = _decompose(roi.affine)
        assert abs(phi - (-math.pi / 4.0)) < 1e-4, (
            f"test 10 diagonal: phi={phi} expected ~{-math.pi / 4.0}"
        )

    # ---- test 11: rotated crop centre still round-trips to frame centre ----
    # The palm-shifted centre is a fixed point of rotation about itself,
    # so back-projecting crop (112, 112) through the inverse affine must
    # still yield the palm-shifted frame position regardless of rotation.
    # With size_scale=1.1, shift_fraction=0.35:
    #
    #   person_v: wrist=(400,200), elbow=(400,300), shift=0.35*(0,-100)=(0,-35)
    #             centre_v = (400, 200 - 35) = (400, 165)
    #   person_h: wrist=(400,300), elbow=(300,300), shift=0.35*(100,0)=(35,0)
    #             centre_h = (400 + 35, 300) = (435, 300)
    #   person_d: wrist=(400,200), elbow=(300,300), shift=0.35*(100,-100)=(35,-35)
    #             centre_d = (400 + 35, 200 - 35) = (435, 165)
    expected_centres = {
        "vertical":   (400.0, 165.0, rois_v),
        "horizontal": (435.0, 300.0, rois_h),
        "diagonal":   (435.0, 165.0, rois_d),
    }
    crop_centre_h = np.array([[112.0], [112.0], [1.0]], dtype=np.float32)
    for label, (exp_cx, exp_cy, roi_list) in expected_centres.items():
        left_roi = next(r for r in roi_list if r.side == "left")
        back = left_roi.affine @ crop_centre_h
        assert abs(float(back[0, 0]) - exp_cx) < 0.5, (
            f"test 11 {label} frame_x: got {float(back[0, 0])} "
            f"expected {exp_cx}"
        )
        assert abs(float(back[1, 0]) - exp_cy) < 0.5, (
            f"test 11 {label} frame_y: got {float(back[1, 0])} "
            f"expected {exp_cy}"
        )

    print("hand_roi self-test PASS")


if __name__ == "__main__":
    _self_test()
