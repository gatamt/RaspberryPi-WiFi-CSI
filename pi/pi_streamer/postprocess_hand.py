"""MediaPipe hand_landmark_lite postprocess — pure numpy.

Decodes the raw Hailo outputs of ``hand_landmark_lite.hef`` into
(x, y) landmark coordinates in the CROP input pixel space (0..target_size).
Splitting the math into its own module makes it unit-testable on the Mac
without a Hailo device. ``HailoHandInference`` is responsible for calling
the decoder once per crop and back-projecting to frame coordinates via
the stored per-crop affine.

The ``hand_landmark_lite`` HEF from Hailo Model Zoo v5.1.0/hailo10h has:

  - input : ``hand_landmark_lite/input_layer1`` shape ``(224, 224, 3)``
            UINT8 NHWC, identity quant (scale=1.0, zp=0), takes raw 0..255
            pixel values from the crop.
  - 4 UINT8 outputs (from ``HEF.get_output_vstream_infos()``):
      * ``hand_landmark_lite/fc1`` shape ``(63,)`` — 21 × (x, y, z) landmarks
        in **crop pixel coordinates**. Quant scale ~1.145, zp ~63,
        dequantized range approximately ``[-72, 220]`` on 224×224 input.
      * ``hand_landmark_lite/fc3`` shape ``(63,)`` — 21 × (x, y, z) world
        landmarks in meters (wrist at origin). Quant scale ~5.4e-4, range
        approximately ``[-0.05, 0.09]``. Unused for 2D overlay.
      * ``hand_landmark_lite/fc2`` shape ``(1,)`` — sigmoid scalar, scale
        ~1/255 zp=0, range [0, 1]. One of {presence, handedness}.
      * ``hand_landmark_lite/fc4`` shape ``(1,)`` — sigmoid scalar, same
        quant as fc2. The other of {presence, handedness}.

Hailo does not document which of fc2/fc4 is presence and which is
handedness, and shape+quant alone can't distinguish them. What the live
diagnostics showed:

  * Both scalars reach near-1.0 together when a well-framed hand is in a
    MediaPipe-style crop (presence histogram median ~0.82 for a reliably
    detected hand).
  * Both scalars collapse toward 0 together when no hand is present
    (a no-hand 5s window showed max(min_per_crop) = 0.106 across 168
    crops).
  * Per-side, one hand often had one scalar high and the other low even
    though the model had clearly registered the hand, so gating on
    ``min()`` kept suppressing that side.

Because of that, ``HandLandmarks.presence`` is derived as
``max(fc2, fc4)``: if either scalar reports a confident hand we'd rather
render it and let the HandTracker EMA + confirmation hysteresis clean up
single-frame noise than blackhole the output because one scalar lagged.
Both scalars go low together on no-hand frames, so max should not
introduce ghost hands in practice. The wrist-anchored ROI cascade (only
crops when pose saw a wrist) and the tracker confirmation hysteresis are
the two other layers of defense.

Dispatch is by shape, not name: the two (63,) tensors are identified as
landmarks vs. world by the quant ``scale`` field (the pixel-space landmarks
have a much larger scale, ~1.1, while world landmarks have scale ~5e-4).
This is robust against Hailo renaming outputs in future compiler versions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, NamedTuple, Tuple

import numpy as np

from .hand_types import NUM_HAND_KEYPOINTS


class QuantInfo(NamedTuple):
    """Quantization parameters for a single Hailo output tensor."""

    scale: float
    zero_point: float


@dataclass(frozen=True)
class RawHandLandmarks:
    """Decoder output in crop pixel coordinates (0..target_size).

    ``HailoHandInference`` back-projects these through the crop's affine
    to produce frame-space ``HandLandmarks`` for the tracker and overlay.

    ``presence`` is the scalar the overlay will broadcast across all 21
    landmarks and gate against ``HAND_MIN_KPT_SCORE``. See
    :class:`HandLandmarkDecoder` for how it is computed from the two HEF
    scalar outputs.
    """

    kpts: np.ndarray   # shape (21, 2) float32, crop pixel (x, y)
    presence: float
    handedness_raw: float


class HandLandmarkDecoder:
    """Pure-numpy decoder for the MediaPipe hand_landmark_lite HEF."""

    def __init__(
        self,
        output_shapes: Dict[str, Tuple[int, ...]],
        quant_info: Dict[str, QuantInfo],
        target_size: int = 224,
    ) -> None:
        self.target_size = target_size
        self._quant_info = dict(quant_info)

        # Dispatch outputs by shape + quant scale, not by name, so the decoder
        # keeps working if the Hailo compiler renames tensors in a future HEF.
        big_names = [
            name for name, shape in output_shapes.items()
            if len(shape) == 1 and shape[0] == 3 * NUM_HAND_KEYPOINTS
        ]
        small_names = [
            name for name, shape in output_shapes.items()
            if len(shape) == 1 and shape[0] == 1
        ]

        if len(big_names) != 2 or len(small_names) != 2:
            raise ValueError(
                "hand_landmark_lite: expected 2 landmark outputs (shape (63,)) "
                f"and 2 scalar outputs (shape (1,)), got big={big_names} "
                f"small={small_names}"
            )

        # Of the two (63,) tensors, the pixel-space landmarks have the much
        # larger quant scale (~1.1 vs ~5e-4). World landmarks are in meters.
        big_names.sort(key=lambda n: self._quant_info[n].scale, reverse=True)
        self._landmarks_name = big_names[0]
        self._world_name = big_names[1]  # unused by 2D overlay, kept for completeness

        # The two scalars cannot be distinguished by shape or quant info —
        # see module docstring. We decode both and expose max/min as
        # presence/handedness_raw; MVP rendering ignores handedness.
        self._scalar_a_name, self._scalar_b_name = small_names

    def decode(self, raw_outputs: Dict[str, np.ndarray]) -> RawHandLandmarks:
        """Decode one set of raw Hailo output tensors into crop-pixel landmarks.

        ``raw_outputs`` is a dict keyed by output tensor name with the native
        dtype buffers filled by ``ConfiguredInferModel.run_async`` — the same
        buffers bound in ``HailoHandInference._run_worker``.
        """
        landmarks_raw = raw_outputs[self._landmarks_name]
        scalar_a_raw = raw_outputs[self._scalar_a_name]
        scalar_b_raw = raw_outputs[self._scalar_b_name]

        # Dequantize: (raw - zp) * scale
        qi_lm = self._quant_info[self._landmarks_name]
        lm_dequant = (
            landmarks_raw.astype(np.float32) - np.float32(qi_lm.zero_point)
        ) * np.float32(qi_lm.scale)
        # Reshape (63,) → (21, 3); we only keep (x, y).
        kpts_xyz = lm_dequant.reshape(NUM_HAND_KEYPOINTS, 3)
        kpts = kpts_xyz[:, :2].astype(np.float32, copy=True)

        qi_a = self._quant_info[self._scalar_a_name]
        scalar_a = float(
            (float(scalar_a_raw.reshape(-1)[0]) - qi_a.zero_point) * qi_a.scale
        )
        qi_b = self._quant_info[self._scalar_b_name]
        scalar_b = float(
            (float(scalar_b_raw.reshape(-1)[0]) - qi_b.zero_point) * qi_b.scale
        )

        # Presence gate: take the MAX of the two scalars. Short version:
        # both scalars go low together when there is no hand (max-of-min
        # < 0.11 across 168 no-hand crops), but when a hand is present one
        # scalar often leads the other, and gating on the LOWER kept the
        # overlay from ever rendering the hand that lagged.
        # ``handedness_raw`` exposes the min for debugging only.
        presence = max(scalar_a, scalar_b)
        handedness_raw = min(scalar_a, scalar_b)

        return RawHandLandmarks(
            kpts=kpts,
            presence=presence,
            handedness_raw=handedness_raw,
        )


def back_project(
    raw: RawHandLandmarks,
    affine_crop_to_frame: np.ndarray,
) -> np.ndarray:
    """Back-project crop-pixel landmarks into frame pixel coordinates.

    ``affine_crop_to_frame`` is the 2x3 affine that maps (crop_x, crop_y) to
    (frame_x, frame_y). Built by ``WristAnchoredROISource`` via
    ``cv2.invertAffineTransform`` of the crop extraction affine. This is the
    inverse of the letterbox back-projection pattern used in
    ``HailoPoseInference._transform_raw_detection``; the math is the same,
    only the affine is per-crop instead of per-frame.

    Returns a (21, 2) float32 array of frame pixel coordinates.
    """
    kpts = raw.kpts  # (21, 2)
    M = affine_crop_to_frame[:, :2].T.astype(np.float32)  # (2, 2)
    t = affine_crop_to_frame[:, 2].astype(np.float32)      # (2,)
    return (kpts @ M + t).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Self-test (run with ``python -m pi_streamer.postprocess_hand``)
# ---------------------------------------------------------------------------


def _self_test() -> None:
    # ----- decoder -----
    # Synthetic HEF metadata mimicking the observed hand_landmark_lite layout
    output_shapes = {
        "fc1": (63,),  # landmarks (pixel space)
        "fc2": (1,),   # scalar (either presence or handedness)
        "fc3": (63,),  # world landmarks (meters)
        "fc4": (1,),   # the other scalar
    }
    quant_info = {
        "fc1": QuantInfo(scale=1.145, zero_point=63.0),
        "fc2": QuantInfo(scale=1.0 / 255.0, zero_point=0.0),
        "fc3": QuantInfo(scale=5.4e-4, zero_point=96.0),
        "fc4": QuantInfo(scale=1.0 / 255.0, zero_point=0.0),
    }
    decoder = HandLandmarkDecoder(output_shapes, quant_info, target_size=224)
    assert decoder._landmarks_name == "fc1", decoder._landmarks_name
    assert decoder._world_name == "fc3", decoder._world_name

    # Build raw uint8 buffers so the dequant path matches what Hailo writes.
    # For fc1 (pixel landmarks), put a known value at raw=63+87=150, which
    # dequantizes to (150-63)*1.145 = 99.615 for every element. So every
    # landmark x/y/z should come out as 99.615.
    fc1_raw = np.full((63,), 150, dtype=np.uint8)
    fc3_raw = np.full((63,), 96, dtype=np.uint8)  # world landmarks = 0
    fc2_raw = np.array([255], dtype=np.uint8)     # → 1.0 (one scalar very high)
    fc4_raw = np.array([0], dtype=np.uint8)       # → 0.0 (the other very low)

    raw = decoder.decode({
        "fc1": fc1_raw,
        "fc2": fc2_raw,
        "fc3": fc3_raw,
        "fc4": fc4_raw,
    })
    assert raw.kpts.shape == (21, 2), raw.kpts.shape
    assert raw.kpts.dtype == np.float32
    # Every kpt coordinate should dequantize to ~99.615.
    expected = (150 - 63) * 1.145
    assert np.allclose(raw.kpts, expected, atol=1e-3), raw.kpts[:3]
    # presence = max(fc2, fc4) so a confident scalar on EITHER output
    # triggers render, avoiding the one-lagging-scalar suppression.
    assert abs(raw.presence - 1.0) < 1e-4, raw.presence
    # handedness_raw (debug-only) = min(fc2, fc4) = 0.0
    assert abs(raw.handedness_raw - 0.0) < 1e-4, raw.handedness_raw

    # ----- back_project -----
    # Affine that maps crop (0,0) → frame (100, 200) with scale 2 in x and 2 in y.
    # Row-major 2x3: [[sx, 0, tx], [0, sy, ty]]
    affine = np.array([[2.0, 0.0, 100.0],
                       [0.0, 2.0, 200.0]], dtype=np.float32)
    # Force crop-space landmarks to known values for this test
    raw_test = RawHandLandmarks(
        kpts=np.array([[0.0, 0.0], [10.0, 20.0], [112.0, 112.0]] + [[0.0, 0.0]] * 18,
                      dtype=np.float32),
        presence=1.0,
        handedness_raw=0.0,
    )
    projected = back_project(raw_test, affine)
    # (0,0) crop → (100,200) frame
    assert np.allclose(projected[0], [100.0, 200.0]), projected[0]
    # (10,20) crop → (100 + 2*10, 200 + 2*20) = (120, 240)
    assert np.allclose(projected[1], [120.0, 240.0]), projected[1]
    # (112, 112) crop → (100 + 2*112, 200 + 2*112) = (324, 424)
    assert np.allclose(projected[2], [324.0, 424.0]), projected[2]

    # ----- shape-validation errors -----
    try:
        HandLandmarkDecoder(
            output_shapes={"bad": (100,)},
            quant_info={"bad": QuantInfo(scale=1.0, zero_point=0.0)},
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for malformed output shapes")

    print("postprocess_hand self-test PASS")


if __name__ == "__main__":
    _self_test()
