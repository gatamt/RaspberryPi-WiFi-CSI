"""YOLOv8 pose postprocess — pure numpy.

Decodes raw Hailo outputs (9 feature map tensors at 3 scales) into
``RawDetection`` objects in letterbox coordinate space. Split into its own
module so the math can be unit-tested without a Hailo device.

The HEF ``yolov8m_pose_h10.hef`` from the Raspberry Pi ``hailo-models``
package does not include on-chip NMS or postprocess, so the CPU side has
to do the full decode:

  1. Dequantize each output tensor: ``(raw - zero_point) * scale``.
  2. DFL decode for bbox regression:
       - reshape ``(H, W, 64)`` → ``(H*W, 4, 16)``
       - softmax over the 16 DFL bins
       - weighted sum with ``[0, 1, ..., 15]`` → 4 distances ``(lt, tt, rt, bt)``
         in stride units.
  3. Anchor-relative bbox: ``anchor = (col + 0.5, row + 0.5)`` in grid units,
     ``x1 = (anchor_x - lt) * stride``, etc.
  4. Keypoint decode (17 keypoints × (x, y, visibility)):
       - ``kx = (raw_x * 2.0 + col) * stride``
       - ``ky = (raw_y * 2.0 + row) * stride``
       - ``vis = sigmoid(raw_vis)``
       (Objectness already has sigmoid baked into the last conv by the Hailo
       compiler — it comes out as UINT8 with scale=1/255 and zp=0, so the
       dequantized value IS the probability.)
  5. Concatenate across scales, threshold by objectness, NMS (single class).

References:
  - Ultralytics YOLOv8 head: ``ultralytics/nn/modules/head.py`` (``Pose``,
    ``dist2bbox``, ``kpts_decode``).
  - DFL: "Generalized Focal Loss" (Li et al., 2020).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Tuple

import numpy as np


class QuantInfo(NamedTuple):
    """Quantization parameters for a single Hailo output tensor."""

    scale: float
    zero_point: float


@dataclass(frozen=True)
class Keypoint:
    """One keypoint in letterbox pixel coordinates (0 .. input_size)."""

    x: float
    y: float
    score: float  # 0..1 visibility


@dataclass(frozen=True)
class RawDetection:
    """Single person detection in letterbox coordinate space.

    Call sites (``inference.py``) are responsible for reversing the
    letterbox transform to get frame coordinates.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    keypoints: Tuple[Keypoint, ...]


@dataclass(frozen=True)
class _ScaleSpec:
    """Grouping of the 3 output tensors that belong to one feature scale."""

    grid_h: int
    grid_w: int
    stride: int
    bbox_name: str
    obj_name: str
    kpts_name: str


class YOLOv8PoseDecoder:
    """Stateful decoder that caches anchors, DFL weights, and quant info.

    Create once per inference worker, then call :meth:`decode` for every
    frame. The class is not thread-safe — use one decoder per thread.
    """

    def __init__(
        self,
        output_shapes: Dict[str, Tuple[int, int, int]],
        quant_info: Dict[str, QuantInfo],
        input_size: int = 640,
        reg_max: int = 16,
        num_keypoints: int = 17,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.7,
    ) -> None:
        self.input_size = input_size
        self.reg_max = reg_max
        self.num_keypoints = num_keypoints
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.quant_info = quant_info

        self._dfl_weights = np.arange(reg_max, dtype=np.float32)
        self._scales = self._group_outputs(output_shapes)

        # Pre-compute flat anchor index arrays per scale
        self._anchor_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for scale in self._scales:
            cols = np.tile(
                np.arange(scale.grid_w, dtype=np.float32), scale.grid_h
            )
            rows = np.repeat(
                np.arange(scale.grid_h, dtype=np.float32), scale.grid_w
            )
            self._anchor_cache[scale.stride] = (cols, rows)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _group_outputs(
        self,
        output_shapes: Dict[str, Tuple[int, int, int]],
    ) -> List[_ScaleSpec]:
        """Group Hailo output tensors into (bbox, obj, kpts) triples per scale.

        Grouping is done by spatial grid size so the module does not depend
        on Hailo's specific layer names (``conv90``, ``conv75``, …).
        """
        triples: Dict[Tuple[int, int], Dict[int, str]] = {}
        for name, shape in output_shapes.items():
            gh, gw, ch = shape
            triples.setdefault((gh, gw), {})[ch] = name

        bbox_ch = self.reg_max * 4
        kpts_ch = self.num_keypoints * 3
        obj_ch = 1

        scales: List[_ScaleSpec] = []
        for (gh, gw), trio in triples.items():
            missing = [c for c in (bbox_ch, obj_ch, kpts_ch) if c not in trio]
            if missing:
                raise ValueError(
                    f"scale {gh}x{gw} missing channel counts {missing}; "
                    f"got channels {sorted(trio.keys())}"
                )
            if self.input_size % gw != 0 or self.input_size % gh != 0:
                raise ValueError(
                    f"input size {self.input_size} not divisible by "
                    f"grid {gh}x{gw}"
                )
            stride = self.input_size // gw
            if self.input_size // gh != stride:
                raise ValueError(
                    f"non-square strides not supported: grid {gh}x{gw}"
                )
            scales.append(
                _ScaleSpec(
                    grid_h=gh,
                    grid_w=gw,
                    stride=stride,
                    bbox_name=trio[bbox_ch],
                    obj_name=trio[obj_ch],
                    kpts_name=trio[kpts_ch],
                )
            )

        scales.sort(key=lambda s: s.stride)
        return scales

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decode(self, raw_outputs: Dict[str, np.ndarray]) -> List[RawDetection]:
        """Run full decode: dequant → DFL → keypoints → threshold → NMS.

        ``raw_outputs`` must be a dict mapping output tensor name to the
        native-dtype numpy array returned by Hailo (``uint8`` for bbox
        regression and objectness, ``uint16`` for keypoints).
        """
        all_boxes: List[np.ndarray] = []
        all_scores: List[np.ndarray] = []
        all_kpts: List[np.ndarray] = []

        for scale in self._scales:
            boxes, scores, kpts = self._decode_scale(scale, raw_outputs)
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_kpts.append(kpts)

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        kpts = np.concatenate(all_kpts, axis=0)

        # Threshold on objectness (single class: person)
        keep = scores >= self.score_threshold
        if not np.any(keep):
            return []
        boxes = boxes[keep]
        scores = scores[keep]
        kpts = kpts[keep]

        # NMS
        keep_idx = self._nms(boxes, scores, self.iou_threshold)

        detections: List[RawDetection] = []
        for idx in keep_idx:
            kp_tuple = tuple(
                Keypoint(
                    x=float(kpts[idx, k, 0]),
                    y=float(kpts[idx, k, 1]),
                    score=float(kpts[idx, k, 2]),
                )
                for k in range(self.num_keypoints)
            )
            detections.append(
                RawDetection(
                    x1=float(boxes[idx, 0]),
                    y1=float(boxes[idx, 1]),
                    x2=float(boxes[idx, 2]),
                    y2=float(boxes[idx, 3]),
                    score=float(scores[idx]),
                    keypoints=kp_tuple,
                )
            )
        return detections

    # ------------------------------------------------------------------
    # Per-scale decode
    # ------------------------------------------------------------------

    def _decode_scale(
        self,
        scale: _ScaleSpec,
        raw_outputs: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        bbox_raw = raw_outputs[scale.bbox_name]
        obj_raw = raw_outputs[scale.obj_name]
        kpts_raw = raw_outputs[scale.kpts_name]

        bbox_f = self._dequant(bbox_raw, scale.bbox_name)
        obj_f = self._dequant(obj_raw, scale.obj_name)
        kpts_f = self._dequant(kpts_raw, scale.kpts_name)

        n_cells = scale.grid_h * scale.grid_w
        stride_f = float(scale.stride)
        cols, rows = self._anchor_cache[scale.stride]

        # DFL decode for bbox
        bbox_flat = bbox_f.reshape(n_cells, 4, self.reg_max)
        # Numerically stable softmax
        bbox_max = bbox_flat.max(axis=2, keepdims=True)
        exp = np.exp(bbox_flat - bbox_max)
        softmax = exp / exp.sum(axis=2, keepdims=True)
        distances = np.einsum("ncb,b->nc", softmax, self._dfl_weights)
        # distances layout: [left, top, right, bottom] (Ultralytics dist2bbox order)

        anchor_x = cols + 0.5
        anchor_y = rows + 0.5
        x1 = (anchor_x - distances[:, 0]) * stride_f
        y1 = (anchor_y - distances[:, 1]) * stride_f
        x2 = (anchor_x + distances[:, 2]) * stride_f
        y2 = (anchor_y + distances[:, 3]) * stride_f
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # Objectness — already post-sigmoid so this IS the probability
        scores = obj_f.reshape(n_cells)

        # Keypoint decode (interleaved x, y, v per keypoint)
        kpts_flat = kpts_f.reshape(n_cells, self.num_keypoints, 3)
        raw_kx = kpts_flat[:, :, 0]
        raw_ky = kpts_flat[:, :, 1]
        raw_kv = kpts_flat[:, :, 2]

        kx = (raw_kx * 2.0 + cols.reshape(-1, 1)) * stride_f
        ky = (raw_ky * 2.0 + rows.reshape(-1, 1)) * stride_f
        kv = _sigmoid(raw_kv)
        decoded_kpts = np.stack([kx, ky, kv], axis=2)

        return boxes, scores, decoded_kpts

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _dequant(self, raw: np.ndarray, name: str) -> np.ndarray:
        qi = self.quant_info[name]
        return (raw.astype(np.float32) - qi.zero_point) * qi.scale

    @staticmethod
    def _nms(
        boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
    ) -> List[int]:
        if boxes.size == 0:
            return []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]

        keep: List[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter = inter_w * inter_h
            iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
            order = rest[iou <= iou_threshold]
        return keep


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


def _self_test() -> None:
    """Lightweight self-tests that don't require a Hailo device.

    Run manually with::

        python -m pi_streamer.postprocess
    """

    # 1. Construct a decoder for yolov8m_pose output shapes at input 640
    output_shapes = {
        "p3_bbox": (80, 80, 64),
        "p3_obj": (80, 80, 1),
        "p3_kpts": (80, 80, 51),
        "p4_bbox": (40, 40, 64),
        "p4_obj": (40, 40, 1),
        "p4_kpts": (40, 40, 51),
        "p5_bbox": (20, 20, 64),
        "p5_obj": (20, 20, 1),
        "p5_kpts": (20, 20, 51),
    }
    quant = {name: QuantInfo(scale=1.0, zero_point=0.0) for name in output_shapes}
    # objectness is stored with scale 1/255 so a raw 255 maps to probability 1.0
    for k in ("p3_obj", "p4_obj", "p5_obj"):
        quant[k] = QuantInfo(scale=1.0 / 255.0, zero_point=0.0)

    decoder = YOLOv8PoseDecoder(
        output_shapes=output_shapes,
        quant_info=quant,
    )

    # 2. All-zero inputs → no detections
    zero_outputs: Dict[str, np.ndarray] = {}
    for name, shape in output_shapes.items():
        if "kpts" in name:
            zero_outputs[name] = np.zeros(shape, dtype=np.uint16)
        else:
            zero_outputs[name] = np.zeros(shape, dtype=np.uint8)
    dets = decoder.decode(zero_outputs)
    assert dets == [], f"expected empty, got {len(dets)} detections"

    # 3. Plant one high-confidence cell in P3 at (row=40, col=40) which is the
    #    center of the 640x640 input. Set objectness to ~1.0 and keep bbox and
    #    keypoint outputs at zero → the decoder should produce exactly one
    #    detection near the image center with a small box.
    outputs = {
        name: (
            np.zeros(shape, dtype=np.uint16)
            if "kpts" in name
            else np.zeros(shape, dtype=np.uint8)
        )
        for name, shape in output_shapes.items()
    }
    outputs["p3_obj"][40, 40, 0] = 255  # raw 255 * (1/255) = 1.0 probability
    dets = decoder.decode(outputs)
    assert len(dets) == 1, f"expected 1 detection, got {len(dets)}"
    det = dets[0]
    # Cell (row=40, col=40) on stride-8 grid has anchor center (40.5, 40.5).
    # With all DFL bbox logits zero the softmax is uniform (1/16 per bin), so
    # the expected distance along every side is sum(k/16 for k in 0..15) = 7.5
    # in stride units. The decoded bbox is therefore 15*stride = 120 px wide
    # and tall, centered at (40.5, 40.5) * 8 = (324, 324).
    expected_center = 40.5 * 8.0
    expected_half = 7.5 * 8.0
    cx = (det.x1 + det.x2) / 2.0
    cy = (det.y1 + det.y2) / 2.0
    assert abs(cx - expected_center) < 0.5, f"cx={cx}"
    assert abs(cy - expected_center) < 0.5, f"cy={cy}"
    assert abs((det.x2 - det.x1) - 2 * expected_half) < 1.0, (
        f"width={det.x2 - det.x1}"
    )
    assert det.score > 0.99, f"score={det.score}"
    assert len(det.keypoints) == 17
    # With all zero keypoint logits, kx = (0*2 + col) * stride = col * stride.
    # For the cell at col=40 that is 320.0, and visibility = sigmoid(0) = 0.5.
    assert abs(det.keypoints[0].x - 320.0) < 0.5, (
        f"kpt0.x={det.keypoints[0].x}"
    )
    assert abs(det.keypoints[0].score - 0.5) < 0.01, (
        f"kpt0.score={det.keypoints[0].score}"
    )

    # 4. Grouping test: inverted dict order must still pair channels correctly.
    reversed_shapes = {
        "a": (20, 20, 1),
        "b": (20, 20, 51),
        "c": (20, 20, 64),
        "d": (40, 40, 64),
        "e": (40, 40, 51),
        "f": (40, 40, 1),
        "g": (80, 80, 51),
        "h": (80, 80, 1),
        "i": (80, 80, 64),
    }
    q2 = {name: QuantInfo(1.0, 0.0) for name in reversed_shapes}
    dec2 = YOLOv8PoseDecoder(
        output_shapes=reversed_shapes,
        quant_info=q2,
    )
    strides = [s.stride for s in dec2._scales]
    assert strides == [8, 16, 32], f"wrong stride order: {strides}"

    print("postprocess self-test PASS")


if __name__ == "__main__":
    _self_test()
