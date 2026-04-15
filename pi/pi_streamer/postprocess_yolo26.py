"""YOLO26 object-detection postprocess — pure numpy.

Decodes the 6 raw Hailo output tensors (3 scales × {bbox, class_scores})
from ``yolo26m.hef`` (the ``hailo-models`` deb package variant compiled for
HAILO10H) into ``ObjectDetection`` objects in letterbox coordinate space.

Compared to ``yolov8m_pose``, two things differ in the Hailo compilation
and therefore in the decode path:

  1. DFL is already fused into the HEF, so bbox outputs are 4 channels
     (ltrb distances in cell units) instead of 64 channels (16 DFL bins
     times 4 sides). No softmax / weighted-sum is needed on the CPU.
  2. Class scores come out as raw logits, *not* post-sigmoid, so we have
     to apply sigmoid before thresholding. This is the opposite of
     yolov8_pose's objectness which was baked to UINT8 with scale=1/255.

Quantization on ``yolo26m.hef``:

  - bbox outputs (4 ch, UINT16): scale ≈ 4e-4 .. 6e-4, non-zero zp
    → dequantized value is ltrb distance in cell units
  - class outputs (80 ch, UINT16): scale ≈ 1.4e-3 .. 2.6e-3, zp ≈ 30-31k
    → dequantized value is a raw logit in roughly (-80, +90)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Tuple

import numpy as np


# COCO 80-class names, index matches the 80 class channels in the HEF
# (same order as in the official YOLO Model Zoo / Ultralytics coco.yaml).
COCO_CLASSES: Tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)


class QuantInfo(NamedTuple):
    """Quantization parameters for a single Hailo output tensor."""

    scale: float
    zero_point: float


@dataclass(frozen=True)
class RawObjectDetection:
    """Single object detection in letterbox coordinate space.

    ``inference.py`` is responsible for reversing the letterbox transform
    to get frame coordinates, same as for pose detections.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int


@dataclass(frozen=True)
class _ScaleSpec:
    """Grouping of the 2 output tensors that belong to one feature scale."""

    grid_h: int
    grid_w: int
    stride: int
    bbox_name: str
    cls_name: str


class YOLO26Decoder:
    """Stateful YOLO26 decoder — fused-DFL variant.

    Create once per inference worker, then call :meth:`decode` for every
    frame. The class is not thread-safe; use one decoder per thread.
    """

    def __init__(
        self,
        output_shapes: Dict[str, Tuple[int, int, int]],
        quant_info: Dict[str, QuantInfo],
        input_size: int = 640,
        num_classes: int = 80,
        score_threshold: float = 0.30,
        iou_threshold: float = 0.5,
    ) -> None:
        self.input_size = input_size
        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.quant_info = quant_info

        self._scales = self._group_outputs(output_shapes)

        # Pre-compute flat (col, row) index arrays per scale.
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
        """Group Hailo output tensors into (bbox, cls) pairs per scale.

        Grouping is done by spatial grid size so the module does not depend
        on Hailo's specific layer names (``conv71``, ``conv87``, …).
        """
        pairs: Dict[Tuple[int, int], Dict[int, str]] = {}
        for name, shape in output_shapes.items():
            gh, gw, ch = shape
            pairs.setdefault((gh, gw), {})[ch] = name

        bbox_ch = 4
        cls_ch = self.num_classes

        scales: List[_ScaleSpec] = []
        for (gh, gw), pair in pairs.items():
            missing = [c for c in (bbox_ch, cls_ch) if c not in pair]
            if missing:
                raise ValueError(
                    f"scale {gh}x{gw} missing channel counts {missing}; "
                    f"got channels {sorted(pair.keys())}"
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
                    bbox_name=pair[bbox_ch],
                    cls_name=pair[cls_ch],
                )
            )

        scales.sort(key=lambda s: s.stride)
        return scales

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decode(
        self, raw_outputs: Dict[str, np.ndarray]
    ) -> List[RawObjectDetection]:
        """Run full decode: dequant → sigmoid → threshold → NMS."""
        all_boxes: List[np.ndarray] = []
        all_scores: List[np.ndarray] = []
        all_classes: List[np.ndarray] = []

        for scale in self._scales:
            boxes, scores, classes = self._decode_scale(scale, raw_outputs)
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(classes)

        if not all_boxes:
            return []

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        classes = np.concatenate(all_classes, axis=0)

        # Early filter before NMS
        keep = scores >= self.score_threshold
        if not np.any(keep):
            return []
        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep]

        # Class-agnostic NMS
        keep_idx = self._nms(boxes, scores, self.iou_threshold)

        return [
            RawObjectDetection(
                x1=float(boxes[i, 0]),
                y1=float(boxes[i, 1]),
                x2=float(boxes[i, 2]),
                y2=float(boxes[i, 3]),
                score=float(scores[i]),
                class_id=int(classes[i]),
            )
            for i in keep_idx
        ]

    # ------------------------------------------------------------------
    # Per-scale decode
    # ------------------------------------------------------------------

    def _decode_scale(
        self,
        scale: _ScaleSpec,
        raw_outputs: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        bbox_raw = raw_outputs[scale.bbox_name]
        cls_raw = raw_outputs[scale.cls_name]

        bbox_f = self._dequant(bbox_raw, scale.bbox_name)  # (H, W, 4) cell-units
        cls_logits = self._dequant(cls_raw, scale.cls_name)  # (H, W, 80)

        n_cells = scale.grid_h * scale.grid_w
        stride_f = float(scale.stride)
        cols, rows = self._anchor_cache[scale.stride]

        # ltrb distances in cell units — layout in channels is (l, t, r, b).
        dist = bbox_f.reshape(n_cells, 4)

        anchor_x = cols + 0.5
        anchor_y = rows + 0.5
        x1 = (anchor_x - dist[:, 0]) * stride_f
        y1 = (anchor_y - dist[:, 1]) * stride_f
        x2 = (anchor_x + dist[:, 2]) * stride_f
        y2 = (anchor_y + dist[:, 3]) * stride_f
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # Sigmoid once, then pick argmax class per cell. We use the
        # numerically-stable logistic form: 1 / (1 + exp(-x)).
        cls_flat = cls_logits.reshape(n_cells, self.num_classes)
        # Fast path: rank by argmax on logits (monotonic), sigmoid only
        # the winning logit to get the probability. Saves 79/80 exp calls.
        best_cls = np.argmax(cls_flat, axis=1).astype(np.int32)
        best_logit = cls_flat[np.arange(n_cells), best_cls]
        best_prob = _sigmoid(best_logit)

        return boxes, best_prob, best_cls

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

    Run with::

        python -m pi_streamer.postprocess_yolo26
    """

    output_shapes = {
        "p3_bbox": (80, 80, 4),
        "p3_cls": (80, 80, 80),
        "p4_bbox": (40, 40, 4),
        "p4_cls": (40, 40, 80),
        "p5_bbox": (20, 20, 4),
        "p5_cls": (20, 20, 80),
    }
    # Use scale=1, zp=0 so raw uint values map directly to floats. With a
    # zero zero-point, raw=0 gives logit=0 → sigmoid(0)=0.5, which would
    # accept every empty cell at the default threshold 0.3. Use 0.9 for
    # the self-test so empty cells are rejected while a planted logit=10
    # cell (sigmoid ≈ 1.0) still passes.
    quant = {name: QuantInfo(1.0, 0.0) for name in output_shapes}
    decoder = YOLO26Decoder(
        output_shapes=output_shapes,
        quant_info=quant,
        score_threshold=0.9,
    )

    # 1. All-zero inputs → no detections.
    zero_outputs: Dict[str, np.ndarray] = {
        name: np.zeros(shape, dtype=np.uint16)
        for name, shape in output_shapes.items()
    }
    dets = decoder.decode(zero_outputs)
    assert dets == [], f"expected empty, got {len(dets)} detections"

    # 2. Plant one high-confidence cell in P3 at (row=40, col=40) with class
    #    15 ("cat"). Bbox distances uniformly 4 cells in every direction.
    outputs = {
        name: np.zeros(shape, dtype=np.uint16)
        for name, shape in output_shapes.items()
    }
    # Bbox (l, t, r, b) = (4, 4, 4, 4) in cell units → 64x64 px box centered
    # at cell center (40.5, 40.5)*8 = (324, 324).
    outputs["p3_bbox"][40, 40] = [4, 4, 4, 4]
    # Class 15 logit = 10 → sigmoid(10) ≈ 1.0. All others stay at 0
    # (logit 0, prob 0.5), so argmax picks 15 (highest logit).
    outputs["p3_cls"][40, 40, 15] = 10

    dets = decoder.decode(outputs)
    assert len(dets) == 1, f"expected 1 detection, got {len(dets)}"
    det = dets[0]
    assert det.class_id == 15, f"class_id={det.class_id}"
    assert COCO_CLASSES[det.class_id] == "cat"

    # Expected bbox: (anchor - lt) * stride .. (anchor + rb) * stride
    # anchor = (40.5, 40.5), lt=rb=4, stride=8
    # x1 = (40.5 - 4) * 8 = 292
    # x2 = (40.5 + 4) * 8 = 356
    assert abs(det.x1 - 292.0) < 0.5, f"x1={det.x1}"
    assert abs(det.x2 - 356.0) < 0.5, f"x2={det.x2}"
    assert abs(det.y1 - 292.0) < 0.5, f"y1={det.y1}"
    assert abs(det.y2 - 356.0) < 0.5, f"y2={det.y2}"
    assert det.score > 0.99, f"score={det.score}"

    # 3. Grouping test: inverted dict order must still pair channels correctly.
    reversed_shapes = {
        "a": (20, 20, 4),
        "b": (20, 20, 80),
        "c": (40, 40, 4),
        "d": (40, 40, 80),
        "e": (80, 80, 4),
        "f": (80, 80, 80),
    }
    q2 = {name: QuantInfo(1.0, 0.0) for name in reversed_shapes}
    dec2 = YOLO26Decoder(
        output_shapes=reversed_shapes,
        quant_info=q2,
    )
    strides = [s.stride for s in dec2._scales]
    assert strides == [8, 16, 32], f"wrong stride order: {strides}"

    print("postprocess_yolo26 self-test PASS")


if __name__ == "__main__":
    _self_test()
