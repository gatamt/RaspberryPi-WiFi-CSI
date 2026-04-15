"""Hailo async inference workers for ``yolov8m_pose`` and ``yolo26m``.

Both workers use the HailoRT 5.x ``VDevice.create_infer_model`` async API
with a persistent ``ConfiguredInferModel``. They share the same physical
``/dev/hailo0`` via the HailoRT scheduler, which time-slices between the
two VDevice instances. Raw output feature maps are dequantized and decoded
by dedicated CPU-side decoders
(:class:`pi_streamer.postprocess.YOLOv8PoseDecoder` and
:class:`pi_streamer.postprocess_yolo26.YOLO26Decoder`), so this module only
handles device lifetime, worker threads, letterbox preprocessing, and
letterbox→frame coordinate transformation.

Quantization details for ``yolov8m_pose_h10.hef`` (from the Raspberry Pi
``hailo-models`` deb package), read off via
``HEF.get_output_vstream_infos()``:

  - bbox regression outputs (64 ch, UINT8): scale ≈ 0.07–0.08, non-zero zp
  - objectness outputs (1 ch, UINT8): scale = 1/255, zp = 0
    → the dequantized value IS the post-sigmoid probability
  - keypoint outputs (51 ch, UINT16): scale ≈ 0.0005, zp ≈ 17k–20k
    → raw logits, sigmoid only applied to visibility in postprocess

Quantization details for ``yolo26m.hef`` (same deb package):

  - bbox outputs (4 ch, UINT16): scale ≈ 4e-4..6e-4, non-zero zp
    → DFL already fused, dequantized value is (l, t, r, b) in cell units
  - class outputs (80 ch, UINT16): scale ≈ 1.4e-3..2.6e-3, zp ≈ 30-31k
    → raw logits, sigmoid applied in postprocess before thresholding

Output buffers are allocated at their **native** dtype (uint8/uint16) and
bound once. Using ``float32`` buffers triggers
``HAILO_INVALID_OPERATION (6): Output buffer size ... different than
expected`` because the async API does not do automatic dequantization.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .postprocess import (
    Keypoint as _RawKeypoint,
    QuantInfo,
    RawDetection,
    YOLOv8PoseDecoder,
)
from .postprocess_yolo26 import (
    COCO_CLASSES,
    QuantInfo as _YOLO26QuantInfo,
    RawObjectDetection,
    YOLO26Decoder,
)
from .tracker import ObjectTracker, PersonTracker

LOG = logging.getLogger("pi_streamer.inference")

HEF_INPUT_WIDTH = 640
HEF_INPUT_HEIGHT = 640
NUM_KEYPOINTS = 17

# HailoRT scheduler priority levels. pyhailort 5.1.1 exposes
# set_scheduler_priority(int) on ConfiguredInferModel but does not expose
# the HAILO_SCHEDULER_PRIORITY_{MIN,NORMAL,MAX} C constants as a Python
# enum, so we use integer literals matching the HailoRT C header
# (MIN=0, NORMAL=16, MAX=31). Larger number = higher priority.
#
# Rationale for the assignment:
#   pose  = NORMAL + 2  (highest; must run every frame to feed hand ROI source)
#   object = NORMAL + 1 (runs every frame, yields to pose on contention)
#   hand  = NORMAL - 1  (yields to pose and object; fills scheduler gaps)
HAILO_SCHEDULER_PRIORITY_NORMAL = 16
HAILO_SCHEDULER_PRIORITY_POSE = 18
HAILO_SCHEDULER_PRIORITY_OBJECT = 17
HAILO_SCHEDULER_PRIORITY_HAND = 15

HAND_HEF_INPUT_SIZE = 224

# Confidence threshold for drawing a keypoint in overlay.py. Kept here as the
# source of truth because overlay.py imports it from this module.
KEYPOINT_SCORE_THRESHOLD = 0.30

# COCO pose skeleton — pairs of keypoint indices that get connected with a line
# in overlay.py. Order: left arm, right arm, shoulders, torso, left leg,
# right leg, face.
COCO_SKELETON: Tuple[Tuple[int, int], ...] = (
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
)


# ---------------------------------------------------------------------------
# Public result dataclasses (consumed by overlay.py and main.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Keypoint:
    x: float
    y: float
    score: float


@dataclass(frozen=True)
class PersonDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    keypoints: Tuple[Keypoint, ...]


@dataclass(frozen=True)
class InferenceResult:
    frame_id: int
    timestamp_s: float
    persons: Tuple[PersonDetection, ...]
    latency_ms: float = 0.0


@dataclass(frozen=True)
class ObjectDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    class_name: str


@dataclass(frozen=True)
class ObjectInferenceResult:
    frame_id: int
    timestamp_s: float
    objects: Tuple[ObjectDetection, ...]
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


@dataclass
class _Submission:
    frame_id: int
    frame: np.ndarray  # HxWx3 BGR (from picamera2 "RGB888")


class HailoPoseInference:
    """Async Hailo-10H inference worker with persistent configured model.

    Lifecycle:
        worker = HailoPoseInference(hef_path, frame_w, frame_h)
        worker.start()
        ...
        for each frame:
            worker.submit(frame_id, frame)
            result = worker.latest()  # may be None or a previous frame's result
        ...
        worker.stop()
    """

    def __init__(
        self,
        hef_path: str,
        frame_width: int = 1280,
        frame_height: int = 720,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        vdevice: Optional[Any] = None,
        enable_tracking: bool = True,
        scheduler_priority: int = HAILO_SCHEDULER_PRIORITY_POSE,
    ) -> None:
        self.hef_path = hef_path
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self._scheduler_priority = scheduler_priority
        # Shared Hailo VDevice owned by main.py. When provided, the worker
        # creates its ConfiguredInferModel from this instance instead of
        # opening a new one — required for dual-model operation because the
        # physical device can only be held exclusively by one VDevice at a
        # time. Multiple ConfiguredInferModels on the *same* VDevice are
        # time-sliced by the HailoRT scheduler.
        self._shared_vdevice = vdevice
        # Optional temporal smoother. When enabled, raw decoder output is
        # passed through PersonTracker before letterbox→frame conversion,
        # which removes one-frame flicker on keypoints. Disable via the
        # ``--no-smoothing`` CLI flag in main.py for A/B comparison.
        self._tracker: Optional[PersonTracker] = (
            PersonTracker() if enable_tracking else None
        )

        self._pending: Optional[_Submission] = None
        self._pending_lock = threading.Lock()
        self._pending_event = threading.Event()

        self._latest_result: Optional[InferenceResult] = None
        self._latest_lock = threading.Lock()

        self._worker: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._ready = threading.Event()

        self._inference_count = 0
        self._last_log_time = 0.0

        # Letterbox state, overwritten on every submission before decode
        self._lb_scale = 1.0
        self._lb_pad_x = 0
        self._lb_pad_y = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._worker is not None:
            return
        self._running.set()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="hailo-pose-worker",
            daemon=True,
        )
        self._worker.start()

    def stop(self) -> None:
        self._running.clear()
        self._pending_event.set()
        if self._worker is not None:
            self._worker.join(timeout=3.0)
            self._worker = None

    def wait_ready(self, timeout: float = 10.0) -> bool:
        return self._ready.wait(timeout)

    # ------------------------------------------------------------------
    # Submission / result fetching
    # ------------------------------------------------------------------

    def submit(self, frame_id: int, frame: np.ndarray) -> None:
        """Submit the latest frame for inference, overwriting any pending one."""
        with self._pending_lock:
            self._pending = _Submission(frame_id=frame_id, frame=frame)
        self._pending_event.set()

    def latest(self) -> Optional[InferenceResult]:
        with self._latest_lock:
            return self._latest_result

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        try:
            self._run_worker()
        except Exception as exc:  # noqa: BLE001
            LOG.exception("Hailo worker crashed: %s", exc)
        finally:
            self._ready.clear()

    def _run_worker(self) -> None:
        import cv2  # lazy import

        from hailo_platform import (  # type: ignore
            HEF,
            VDevice,
            FormatType,
        )

        # ------- inspect HEF -------
        hef = HEF(self.hef_path)

        output_shapes: Dict[str, Tuple[int, int, int]] = {}
        quant_info: Dict[str, QuantInfo] = {}
        output_dtypes: Dict[str, np.dtype] = {}

        for info in hef.get_output_vstream_infos():
            shape = tuple(info.shape)
            output_shapes[info.name] = shape
            qi = info.quant_info
            quant_info[info.name] = QuantInfo(
                scale=float(qi.qp_scale),
                zero_point=float(qi.qp_zp),
            )
            output_dtypes[info.name] = _format_to_np_dtype(info.format)

        input_infos = hef.get_input_vstream_infos()
        if not input_infos:
            raise RuntimeError("HEF has no input vstreams")
        input_info = input_infos[0]
        input_name = input_info.name

        LOG.info(
            "Hailo HEF loaded: %s, %d outputs at input %dx%d",
            self.hef_path,
            len(output_shapes),
            HEF_INPUT_WIDTH,
            HEF_INPUT_HEIGHT,
        )
        for name in sorted(output_shapes):
            qi = quant_info[name]
            LOG.debug(
                "  %-28s shape=%s dtype=%s scale=%.6f zp=%.1f",
                name,
                output_shapes[name],
                np.dtype(output_dtypes[name]).name,
                qi.scale,
                qi.zero_point,
            )

        decoder = YOLOv8PoseDecoder(
            output_shapes=output_shapes,
            quant_info=quant_info,
            input_size=HEF_INPUT_WIDTH,
            num_keypoints=NUM_KEYPOINTS,
            score_threshold=self.score_threshold,
            iou_threshold=self.iou_threshold,
        )

        # Clear any stale tracker state from a previous worker lifecycle.
        if self._tracker is not None:
            self._tracker.reset()

        # ------- open device and configure model -------
        # Use the shared VDevice when main.py provides one (required for
        # dual-model operation). Otherwise fall back to a locally-owned
        # VDevice for single-model runs.
        if self._shared_vdevice is not None:
            vdevice_cm = nullcontext(self._shared_vdevice)
        else:
            vdevice_cm = VDevice()
        with vdevice_cm as vdevice:
            infer_model = vdevice.create_infer_model(self.hef_path)
            infer_model.set_batch_size(1)

            with infer_model.configure() as configured:
                # Give pose the highest scheduler priority — it must run every
                # frame to keep baseline quality and to feed the hand ROI
                # source. A single-model run is unaffected; priorities only
                # matter once the scheduler is contending between 2+ models.
                _apply_scheduler_priority(configured, self._scheduler_priority)

                bindings = configured.create_bindings()

                # Input buffer: HWC uint8, native Hailo format for this HEF
                input_buf = np.empty(
                    (HEF_INPUT_HEIGHT, HEF_INPUT_WIDTH, 3),
                    dtype=np.uint8,
                )
                bindings.input().set_buffer(input_buf)

                # Output buffers: native dtypes. Hailo writes into these in place.
                output_buffers: Dict[str, np.ndarray] = {}
                for name, shape in output_shapes.items():
                    buf = np.empty(shape, dtype=output_dtypes[name])
                    bindings.output(name).set_buffer(buf)
                    output_buffers[name] = buf

                LOG.info(
                    "Hailo pose configured, input=%s ready (%dx%d) prio=%d",
                    input_name,
                    HEF_INPUT_WIDTH,
                    HEF_INPUT_HEIGHT,
                    self._scheduler_priority,
                )
                self._ready.set()

                self._inference_loop(
                    cv2_mod=cv2,
                    configured=configured,
                    bindings=bindings,
                    input_buf=input_buf,
                    output_buffers=output_buffers,
                    decoder=decoder,
                )

    def _inference_loop(
        self,
        cv2_mod,
        configured,
        bindings,
        input_buf: np.ndarray,
        output_buffers: Dict[str, np.ndarray],
        decoder: YOLOv8PoseDecoder,
    ) -> None:
        while self._running.is_set():
            if not self._pending_event.wait(timeout=0.5):
                continue
            self._pending_event.clear()

            with self._pending_lock:
                submission = self._pending
                self._pending = None

            if submission is None:
                continue

            try:
                self._run_one_inference(
                    submission=submission,
                    cv2_mod=cv2_mod,
                    configured=configured,
                    bindings=bindings,
                    input_buf=input_buf,
                    output_buffers=output_buffers,
                    decoder=decoder,
                )
            except Exception as exc:  # noqa: BLE001
                LOG.error(
                    "inference error on frame %d: %s",
                    submission.frame_id,
                    exc,
                )

    def _run_one_inference(
        self,
        submission: _Submission,
        cv2_mod,
        configured,
        bindings,
        input_buf: np.ndarray,
        output_buffers: Dict[str, np.ndarray],
        decoder: YOLOv8PoseDecoder,
    ) -> None:
        t0 = time.monotonic()

        self._letterbox_into(cv2_mod, submission.frame, input_buf)

        configured.wait_for_async_ready(timeout_ms=5000)
        job = configured.run_async([bindings])
        job.wait(5000)

        # get_buffer returns the same ndarray we bound earlier
        raw_outputs: Dict[str, np.ndarray] = {
            name: bindings.output(name).get_buffer()
            for name in output_buffers
        }

        raw_detections = decoder.decode(raw_outputs)
        # Smooth in letterbox space (avoids clip-bias at frame edges).
        # Tracker returns confirmed tracks only — tentative tracks in the
        # confirmation hysteresis are filtered out so the overlay never
        # sees a one-frame false positive.
        if self._tracker is not None:
            raw_detections = self._tracker.update(
                raw_detections, time.monotonic()
            )
        persons = tuple(
            self._transform_raw_detection(d) for d in raw_detections
        )

        elapsed_ms = (time.monotonic() - t0) * 1000.0

        result = InferenceResult(
            frame_id=submission.frame_id,
            timestamp_s=time.monotonic(),
            persons=persons,
            latency_ms=elapsed_ms,
        )
        with self._latest_lock:
            self._latest_result = result

        self._inference_count += 1
        now = time.monotonic()
        if now - self._last_log_time > 5.0:
            LOG.info(
                "inference: %d frames, latest %.1f ms, %d persons",
                self._inference_count,
                elapsed_ms,
                len(persons),
            )
            self._last_log_time = now

    # ------------------------------------------------------------------
    # Letterbox helpers
    # ------------------------------------------------------------------

    def _letterbox_into(
        self, cv2_mod, frame: np.ndarray, input_buf: np.ndarray
    ) -> None:
        """Resize ``frame`` preserving aspect into ``input_buf`` with padding.

        Updates ``self._lb_scale``, ``self._lb_pad_x``, ``self._lb_pad_y`` so
        :meth:`_transform_raw_detection` can reverse the transform later.
        When the input is a pre-downscaled lores stream, ``_lb_scale`` is
        adjusted so the inverse maps back to ``frame_width × frame_height``.
        """
        h, w = frame.shape[:2]
        scale = min(HEF_INPUT_WIDTH / w, HEF_INPUT_HEIGHT / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        pad_x = (HEF_INPUT_WIDTH - new_w) // 2
        pad_y = (HEF_INPUT_HEIGHT - new_h) // 2

        self._lb_scale = scale * (w / self.frame_width)
        self._lb_pad_x = pad_x
        self._lb_pad_y = pad_y

        input_buf.fill(114)
        if new_w == w and new_h == h:
            input_buf[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = frame
        else:
            resized = cv2_mod.resize(
                frame, (new_w, new_h), interpolation=cv2_mod.INTER_LINEAR
            )
            input_buf[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    def _transform_raw_detection(self, det: RawDetection) -> PersonDetection:
        # Inverse letterbox without clipping. Frame-edge clipping used to
        # happen here, but it biased the EMA in the temporal tracker
        # inward when an object was partially off-screen. OpenCV's
        # cv2.rectangle/circle/line in overlay.py handle out-of-bounds
        # coordinates correctly on their own.
        scale = self._lb_scale
        pad_x = self._lb_pad_x
        pad_y = self._lb_pad_y

        def to_frame_x(lb_x: float) -> float:
            return (lb_x - pad_x) / scale

        def to_frame_y(lb_y: float) -> float:
            return (lb_y - pad_y) / scale

        kps = tuple(
            Keypoint(
                x=to_frame_x(k.x),
                y=to_frame_y(k.y),
                score=k.score,
            )
            for k in det.keypoints
        )
        return PersonDetection(
            x1=to_frame_x(det.x1),
            y1=to_frame_y(det.y1),
            x2=to_frame_x(det.x2),
            y2=to_frame_y(det.y2),
            score=det.score,
            keypoints=kps,
        )


# ---------------------------------------------------------------------------
# Object detection worker (yolo26m)
# ---------------------------------------------------------------------------


@dataclass
class _ObjectSubmission:
    frame_id: int
    frame: np.ndarray


class HailoObjectInference:
    """Async Hailo-10H YOLO26 object-detection worker.

    Architecturally parallel to :class:`HailoPoseInference`: separate thread,
    separate ``VDevice``, same submit/latest pattern. Both workers run on
    the same physical ``/dev/hailo0`` — the HailoRT scheduler time-slices
    between VDevice instances automatically.

    The HEF outputs 6 raw feature maps (3 scales × {bbox, cls}) with DFL
    already fused, so :class:`YOLO26Decoder` only needs to dequantize,
    sigmoid the class logits, pick argmax per cell, and run NMS.
    """

    def __init__(
        self,
        hef_path: str,
        frame_width: int = 1280,
        frame_height: int = 720,
        score_threshold: float = 0.30,
        iou_threshold: float = 0.5,
        vdevice: Optional[Any] = None,
        enable_tracking: bool = True,
        scheduler_priority: int = HAILO_SCHEDULER_PRIORITY_OBJECT,
    ) -> None:
        self.hef_path = hef_path
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self._scheduler_priority = scheduler_priority
        # Shared Hailo VDevice — see HailoPoseInference.__init__ for why.
        self._shared_vdevice = vdevice
        # Optional temporal smoother — see HailoPoseInference.__init__.
        self._tracker: Optional[ObjectTracker] = (
            ObjectTracker() if enable_tracking else None
        )

        self._pending: Optional[_ObjectSubmission] = None
        self._pending_lock = threading.Lock()
        self._pending_event = threading.Event()

        self._latest_result: Optional[ObjectInferenceResult] = None
        self._latest_lock = threading.Lock()

        self._worker: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._ready = threading.Event()

        self._inference_count = 0
        self._last_log_time = 0.0

        self._lb_scale = 1.0
        self._lb_pad_x = 0
        self._lb_pad_y = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._worker is not None:
            return
        self._running.set()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="hailo-object-worker",
            daemon=True,
        )
        self._worker.start()

    def stop(self) -> None:
        self._running.clear()
        self._pending_event.set()
        if self._worker is not None:
            self._worker.join(timeout=3.0)
            self._worker = None

    def wait_ready(self, timeout: float = 10.0) -> bool:
        return self._ready.wait(timeout)

    # ------------------------------------------------------------------
    # Submission / result fetching
    # ------------------------------------------------------------------

    def submit(self, frame_id: int, frame: np.ndarray) -> None:
        """Submit the latest frame for inference, overwriting any pending one."""
        with self._pending_lock:
            self._pending = _ObjectSubmission(frame_id=frame_id, frame=frame)
        self._pending_event.set()

    def latest(self) -> Optional[ObjectInferenceResult]:
        with self._latest_lock:
            return self._latest_result

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        try:
            self._run_worker()
        except Exception as exc:  # noqa: BLE001
            LOG.exception("Hailo object worker crashed: %s", exc)
        finally:
            self._ready.clear()

    def _run_worker(self) -> None:
        import cv2  # lazy import

        from hailo_platform import (  # type: ignore
            HEF,
            VDevice,
        )

        hef = HEF(self.hef_path)

        output_shapes: Dict[str, Tuple[int, int, int]] = {}
        quant_info: Dict[str, _YOLO26QuantInfo] = {}
        output_dtypes: Dict[str, np.dtype] = {}

        for info in hef.get_output_vstream_infos():
            shape = tuple(info.shape)
            output_shapes[info.name] = shape
            qi = info.quant_info
            quant_info[info.name] = _YOLO26QuantInfo(
                scale=float(qi.qp_scale),
                zero_point=float(qi.qp_zp),
            )
            output_dtypes[info.name] = _format_to_np_dtype(info.format)

        input_infos = hef.get_input_vstream_infos()
        if not input_infos:
            raise RuntimeError("yolo26 HEF has no input vstreams")
        input_info = input_infos[0]
        input_name = input_info.name

        LOG.info(
            "Hailo object HEF loaded: %s, %d outputs at input %dx%d",
            self.hef_path,
            len(output_shapes),
            HEF_INPUT_WIDTH,
            HEF_INPUT_HEIGHT,
        )
        for name in sorted(output_shapes):
            qi = quant_info[name]
            LOG.debug(
                "  %-28s shape=%s dtype=%s scale=%.6f zp=%.1f",
                name,
                output_shapes[name],
                np.dtype(output_dtypes[name]).name,
                qi.scale,
                qi.zero_point,
            )

        decoder = YOLO26Decoder(
            output_shapes=output_shapes,
            quant_info=quant_info,
            input_size=HEF_INPUT_WIDTH,
            num_classes=80,
            score_threshold=self.score_threshold,
            iou_threshold=self.iou_threshold,
        )

        if self._tracker is not None:
            self._tracker.reset()

        if self._shared_vdevice is not None:
            vdevice_cm = nullcontext(self._shared_vdevice)
        else:
            vdevice_cm = VDevice()
        with vdevice_cm as vdevice:
            infer_model = vdevice.create_infer_model(self.hef_path)
            infer_model.set_batch_size(1)

            with infer_model.configure() as configured:
                _apply_scheduler_priority(configured, self._scheduler_priority)

                bindings = configured.create_bindings()

                input_buf = np.empty(
                    (HEF_INPUT_HEIGHT, HEF_INPUT_WIDTH, 3),
                    dtype=np.uint8,
                )
                bindings.input().set_buffer(input_buf)

                output_buffers: Dict[str, np.ndarray] = {}
                for name, shape in output_shapes.items():
                    buf = np.empty(shape, dtype=output_dtypes[name])
                    bindings.output(name).set_buffer(buf)
                    output_buffers[name] = buf

                LOG.info(
                    "Hailo object inference configured, input=%s ready (%dx%d) prio=%d",
                    input_name,
                    HEF_INPUT_WIDTH,
                    HEF_INPUT_HEIGHT,
                    self._scheduler_priority,
                )
                self._ready.set()

                self._inference_loop(
                    cv2_mod=cv2,
                    configured=configured,
                    bindings=bindings,
                    input_buf=input_buf,
                    output_buffers=output_buffers,
                    decoder=decoder,
                )

    def _inference_loop(
        self,
        cv2_mod,
        configured,
        bindings,
        input_buf: np.ndarray,
        output_buffers: Dict[str, np.ndarray],
        decoder: YOLO26Decoder,
    ) -> None:
        while self._running.is_set():
            if not self._pending_event.wait(timeout=0.5):
                continue
            self._pending_event.clear()

            with self._pending_lock:
                submission = self._pending
                self._pending = None

            if submission is None:
                continue

            try:
                self._run_one_inference(
                    submission=submission,
                    cv2_mod=cv2_mod,
                    configured=configured,
                    bindings=bindings,
                    input_buf=input_buf,
                    output_buffers=output_buffers,
                    decoder=decoder,
                )
            except Exception as exc:  # noqa: BLE001
                LOG.error(
                    "object inference error on frame %d: %s",
                    submission.frame_id,
                    exc,
                )

    def _run_one_inference(
        self,
        submission: _ObjectSubmission,
        cv2_mod,
        configured,
        bindings,
        input_buf: np.ndarray,
        output_buffers: Dict[str, np.ndarray],
        decoder: YOLO26Decoder,
    ) -> None:
        t0 = time.monotonic()

        self._letterbox_into(cv2_mod, submission.frame, input_buf)

        configured.wait_for_async_ready(timeout_ms=5000)
        job = configured.run_async([bindings])
        job.wait(5000)

        raw_outputs: Dict[str, np.ndarray] = {
            name: bindings.output(name).get_buffer()
            for name in output_buffers
        }

        raw_detections = decoder.decode(raw_outputs)
        # Smooth in letterbox space — see HailoPoseInference._run_one_inference.
        if self._tracker is not None:
            raw_detections = self._tracker.update(
                raw_detections, time.monotonic()
            )
        objects = tuple(
            self._transform_raw_detection(d) for d in raw_detections
        )

        elapsed_ms = (time.monotonic() - t0) * 1000.0

        result = ObjectInferenceResult(
            frame_id=submission.frame_id,
            timestamp_s=time.monotonic(),
            objects=objects,
            latency_ms=elapsed_ms,
        )
        with self._latest_lock:
            self._latest_result = result

        self._inference_count += 1
        now = time.monotonic()
        if now - self._last_log_time > 5.0:
            LOG.info(
                "object inference: %d frames, latest %.1f ms, %d objects",
                self._inference_count,
                elapsed_ms,
                len(objects),
            )
            self._last_log_time = now

    # ------------------------------------------------------------------
    # Letterbox helpers
    # ------------------------------------------------------------------

    def _letterbox_into(
        self, cv2_mod, frame: np.ndarray, input_buf: np.ndarray
    ) -> None:
        h, w = frame.shape[:2]
        scale = min(HEF_INPUT_WIDTH / w, HEF_INPUT_HEIGHT / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        pad_x = (HEF_INPUT_WIDTH - new_w) // 2
        pad_y = (HEF_INPUT_HEIGHT - new_h) // 2

        self._lb_scale = scale * (w / self.frame_width)
        self._lb_pad_x = pad_x
        self._lb_pad_y = pad_y

        input_buf.fill(114)
        if new_w == w and new_h == h:
            input_buf[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = frame
        else:
            resized = cv2_mod.resize(
                frame, (new_w, new_h), interpolation=cv2_mod.INTER_LINEAR
            )
            input_buf[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    def _transform_raw_detection(
        self, det: RawObjectDetection
    ) -> ObjectDetection:
        # Inverse letterbox without clipping — see HailoPoseInference twin.
        scale = self._lb_scale
        pad_x = self._lb_pad_x
        pad_y = self._lb_pad_y

        def to_frame_x(lb_x: float) -> float:
            return (lb_x - pad_x) / scale

        def to_frame_y(lb_y: float) -> float:
            return (lb_y - pad_y) / scale

        class_name = (
            COCO_CLASSES[det.class_id]
            if 0 <= det.class_id < len(COCO_CLASSES)
            else f"id{det.class_id}"
        )
        return ObjectDetection(
            x1=to_frame_x(det.x1),
            y1=to_frame_y(det.y1),
            x2=to_frame_x(det.x2),
            y2=to_frame_y(det.y2),
            score=det.score,
            class_id=det.class_id,
            class_name=class_name,
        )


# ---------------------------------------------------------------------------
# Hand landmark inference worker (MediaPipe hand_landmark_lite)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HandInferenceResult:
    """Latest hand-landmark inference result — read by main loop and overlay."""

    frame_id: int
    timestamp_s: float
    hands: Tuple["HandLandmarks", ...]  # forward ref resolved at runtime
    latency_ms: float = 0.0
    num_crops: int = 0


@dataclass
class _HandSubmission:
    frame_id: int
    frame: np.ndarray
    persons: tuple


class HailoHandInference:
    """Async Hailo worker for the MediaPipe ``hand_landmark_lite`` HEF.

    Operates on pre-cropped 224×224 hand regions produced by a
    :class:`pi_streamer.hand_roi.HandROISource`, runs one async inference
    per crop sequentially (each ~0.75 ms on Hailo-10H), dequantizes and
    back-projects to frame pixel coordinates, optionally smooths through
    :class:`pi_streamer.tracker.HandTracker`, and publishes the latest
    per-hand landmarks via :meth:`latest`.

    The submit/latest pattern mirrors :class:`HailoPoseInference` and
    :class:`HailoObjectInference` so the main loop wires it up with the
    same cadence. The fundamental differences from the pose/object workers:

      - input is a list of crops, not a full frame
      - preprocessing is a simple copy (not a letterbox resize) because
        the crops are already 224×224 in the correct size
      - back-projection uses a per-crop affine (from :class:`CropInfo`),
        not a frame-wide letterbox scale+pad
      - a top-level try/except in the worker thread sets ``_worker_dead``
        on uncaught exceptions so the main loop can crash-exit instead of
        silently wedging if the worker dies during operation
    """

    def __init__(
        self,
        hef_path: str,
        frame_width: int = 1280,
        frame_height: int = 720,
        vdevice: Optional[Any] = None,
        tracker: Optional[Any] = None,
        roi_source: Optional[Any] = None,
        scheduler_priority: int = HAILO_SCHEDULER_PRIORITY_HAND,
        scheduler_threshold: int = 1,
        scheduler_timeout_ms: int = 30,
    ) -> None:
        self.hef_path = hef_path
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._shared_vdevice = vdevice
        self._tracker = tracker
        self._roi_source = roi_source
        self._scheduler_priority = scheduler_priority
        self._scheduler_threshold = scheduler_threshold
        self._scheduler_timeout_ms = scheduler_timeout_ms

        self._pending: Optional[_HandSubmission] = None
        self._pending_lock = threading.Lock()
        self._pending_event = threading.Event()

        self._latest_result: Optional[HandInferenceResult] = None
        self._latest_lock = threading.Lock()

        self._worker: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._ready = threading.Event()

        # Worker-death flag. Set to True if ``_run_worker`` raises an
        # uncaught exception. Main loop polls this each iteration and
        # exits with non-zero code if set — this turns a silent
        # daemon-thread death into a loud process crash, which is
        # recoverable by systemd/supervisor or a manual restart.
        self._worker_dead = False

        self._inference_count = 0
        self._last_log_time = 0.0

        # Diagnostic state for disambiguating the fc2/fc4 scalar outputs
        # of hand_landmark_lite: one is presence, one is handedness, and
        # the HEF metadata does not say which. After a handful of frames
        # with a real hand in view, the scalar that flips between ~0
        # (no hand) and ~1 (hand) is presence. These counters also
        # survived an earlier round where the two hands behaved
        # asymmetrically, so we keep enough frames to distinguish hand
        # identity from hand presence.
        self._diag_frames_remaining = 150
        self._diag_raw_crops = 0
        self._diag_rejected_shape = 0
        self._diag_presence_samples: list = []
        self._diag_presence_by_side: dict = {"left": [], "right": []}
        self._diag_window_start = 0.0
        self._wrist_sanity_done = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._worker is not None:
            return
        self._running.set()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="hailo-hand-worker",
            daemon=True,
        )
        self._worker.start()

    def stop(self) -> None:
        self._running.clear()
        self._pending_event.set()
        if self._worker is not None:
            self._worker.join(timeout=3.0)
            self._worker = None

    def wait_ready(self, timeout: float = 10.0) -> bool:
        return self._ready.wait(timeout)

    @property
    def worker_dead(self) -> bool:
        return self._worker_dead

    # ------------------------------------------------------------------
    # Submission / result fetching
    # ------------------------------------------------------------------

    def submit(self, frame_id: int, frame: np.ndarray, persons: tuple = ()) -> None:
        """Submit a frame and pose results for hand ROI extraction + inference."""
        with self._pending_lock:
            self._pending = _HandSubmission(
                frame_id=frame_id, frame=frame, persons=persons,
            )
        self._pending_event.set()

    def latest(self) -> Optional[HandInferenceResult]:
        with self._latest_lock:
            return self._latest_result

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        try:
            self._run_worker()
        except Exception as exc:  # noqa: BLE001
            LOG.exception("Hailo hand worker crashed: %s", exc)
            # Signal main loop that we have died. Main loop polls this
            # each iteration and exits with non-zero code.
            self._worker_dead = True
        finally:
            self._ready.clear()

    def _run_worker(self) -> None:
        from hailo_platform import (  # type: ignore
            HEF,
            VDevice,
        )

        from .hand_types import HandLandmarks
        from .postprocess_hand import (
            HandLandmarkDecoder,
            QuantInfo as HandQuantInfo,
            back_project,
        )

        # ------- inspect HEF -------
        hef = HEF(self.hef_path)

        output_shapes: Dict[str, Tuple[int, ...]] = {}
        quant_info: Dict[str, HandQuantInfo] = {}
        output_dtypes: Dict[str, np.dtype] = {}
        for info in hef.get_output_vstream_infos():
            shape = tuple(info.shape)
            output_shapes[info.name] = shape
            qi = info.quant_info
            quant_info[info.name] = HandQuantInfo(
                scale=float(qi.qp_scale),
                zero_point=float(qi.qp_zp),
            )
            output_dtypes[info.name] = _format_to_np_dtype(info.format)

        input_infos = hef.get_input_vstream_infos()
        if not input_infos:
            raise RuntimeError("hand HEF has no input vstreams")
        input_name = input_infos[0].name

        LOG.info(
            "Hailo hand HEF loaded: %s, %d outputs at input %dx%d",
            self.hef_path,
            len(output_shapes),
            HAND_HEF_INPUT_SIZE,
            HAND_HEF_INPUT_SIZE,
        )
        for name in sorted(output_shapes):
            qi = quant_info[name]
            LOG.debug(
                "  %-32s shape=%s dtype=%s scale=%.6f zp=%.1f",
                name,
                output_shapes[name],
                np.dtype(output_dtypes[name]).name,
                qi.scale,
                qi.zero_point,
            )

        decoder = HandLandmarkDecoder(
            output_shapes=output_shapes,
            quant_info=quant_info,
            target_size=HAND_HEF_INPUT_SIZE,
        )

        # Log the decoder dispatch so fc2/fc4 is no mystery. The two
        # scalar names below are the concrete HEF output names (e.g.
        # "fc2", "fc4") picked by the decoder in shape order; one is
        # presence, one is handedness, and the per-frame scalar log emits
        # both values so we can see which one flips with a real hand in
        # view.
        def _describe(name: str) -> str:
            qi = quant_info[name]
            return (
                f"{name} shape={output_shapes[name]} "
                f"dtype={np.dtype(output_dtypes[name]).name} "
                f"scale={qi.scale:.6g} zp={qi.zero_point:.1f}"
            )

        LOG.info(
            "hand decoder dispatch: landmarks=[%s] world=[%s] "
            "scalar_a=[%s] scalar_b=[%s]",
            _describe(decoder._landmarks_name),
            _describe(decoder._world_name),
            _describe(decoder._scalar_a_name),
            _describe(decoder._scalar_b_name),
        )

        # ------- open device and configure model -------
        if self._shared_vdevice is not None:
            vdevice_cm = nullcontext(self._shared_vdevice)
        else:
            vdevice_cm = VDevice()
        with vdevice_cm as vdevice:
            infer_model = vdevice.create_infer_model(self.hef_path)
            infer_model.set_batch_size(1)

            with infer_model.configure() as configured:
                # Low scheduler priority — hand yields to pose and object so
                # the baseline detectors keep running at full rate. With only
                # one hand model and 2 crops/frame ≈ 1.5 ms total on-chip,
                # this fits comfortably in scheduler gaps.
                _apply_scheduler_priority(configured, self._scheduler_priority)
                _apply_scheduler_threshold(configured, self._scheduler_threshold)
                _apply_scheduler_timeout(configured, self._scheduler_timeout_ms)

                bindings = configured.create_bindings()

                # Input buffer: 224x224x3 uint8 BGR, native format.
                input_buf = np.empty(
                    (HAND_HEF_INPUT_SIZE, HAND_HEF_INPUT_SIZE, 3),
                    dtype=np.uint8,
                )
                bindings.input().set_buffer(input_buf)

                # Output buffers: native dtypes. Hailo writes in place.
                output_buffers: Dict[str, np.ndarray] = {}
                for name, shape in output_shapes.items():
                    buf = np.empty(shape, dtype=output_dtypes[name])
                    bindings.output(name).set_buffer(buf)
                    output_buffers[name] = buf

                LOG.info(
                    "Hailo hand inference configured, input=%s ready "
                    "(%dx%d) prio=%d threshold=%d timeout=%dms",
                    input_name,
                    HAND_HEF_INPUT_SIZE,
                    HAND_HEF_INPUT_SIZE,
                    self._scheduler_priority,
                    self._scheduler_threshold,
                    self._scheduler_timeout_ms,
                )
                self._ready.set()

                self._inference_loop(
                    configured=configured,
                    bindings=bindings,
                    input_buf=input_buf,
                    output_buffers=output_buffers,
                    decoder=decoder,
                    HandLandmarks=HandLandmarks,
                    back_project=back_project,
                )

    def _inference_loop(
        self,
        configured,
        bindings,
        input_buf: np.ndarray,
        output_buffers: Dict[str, np.ndarray],
        decoder,
        HandLandmarks,
        back_project,
    ) -> None:
        while self._running.is_set():
            if not self._pending_event.wait(timeout=0.5):
                continue
            self._pending_event.clear()

            with self._pending_lock:
                submission = self._pending
                self._pending = None

            if submission is None:
                continue

            try:
                self._run_one_frame(
                    submission=submission,
                    configured=configured,
                    bindings=bindings,
                    input_buf=input_buf,
                    output_buffers=output_buffers,
                    decoder=decoder,
                    HandLandmarks=HandLandmarks,
                    back_project=back_project,
                )
            except Exception as exc:  # noqa: BLE001
                LOG.error(
                    "hand inference error on frame %d: %s",
                    submission.frame_id,
                    exc,
                )

    def _run_one_frame(
        self,
        submission: _HandSubmission,
        configured,
        bindings,
        input_buf: np.ndarray,
        output_buffers: Dict[str, np.ndarray],
        decoder,
        HandLandmarks,
        back_project,
    ) -> None:
        t0 = time.monotonic()
        raw_hands = []

        if self._roi_source is not None:
            crops = self._roi_source.extract_rois(
                submission.frame, submission.persons,
            )
        else:
            crops = []

        for crop in crops:
            self._diag_raw_crops += 1
            # Copy crop pixels into the pre-bound input buffer. The crop
            # arrives as a contiguous uint8 (H, W, 3) array from
            # WristAnchoredROISource so the assignment is zero-alloc.
            if crop.pixels.shape != input_buf.shape:
                self._diag_rejected_shape += 1
                LOG.warning(
                    "hand crop shape mismatch: got %s, expected %s",
                    crop.pixels.shape,
                    input_buf.shape,
                )
                continue
            input_buf[...] = crop.pixels

            configured.wait_for_async_ready(timeout_ms=5000)
            job = configured.run_async([bindings])
            job.wait(5000)

            raw_outputs: Dict[str, np.ndarray] = {
                name: bindings.output(name).get_buffer()
                for name in output_buffers
            }

            raw = decoder.decode(raw_outputs)
            frame_kpts = back_project(raw, crop.affine)

            # Diagnostic: log raw + dequantized scalars by their concrete
            # HEF output names (e.g. "fc2", "fc4") for the first few
            # frames so we can see which one flips when a real hand
            # enters or leaves the view — that's what disambiguates
            # presence vs. handedness.
            if self._diag_frames_remaining > 0:
                qi_a = decoder._quant_info[decoder._scalar_a_name]
                qi_b = decoder._quant_info[decoder._scalar_b_name]
                buf_a = raw_outputs[decoder._scalar_a_name].reshape(-1)
                buf_b = raw_outputs[decoder._scalar_b_name].reshape(-1)
                scalar_a_raw = float(buf_a[0])
                scalar_b_raw = float(buf_b[0])
                scalar_a_dq = (scalar_a_raw - qi_a.zero_point) * qi_a.scale
                scalar_b_dq = (scalar_b_raw - qi_b.zero_point) * qi_b.scale
                log_fn = LOG.info if self._diag_frames_remaining > 120 else LOG.debug
                log_fn(
                    "hand diag f%d crop[%s]: %s raw=%.0f dq=%.4f  "
                    "%s raw=%.0f dq=%.4f  kpts_range=[%.1f..%.1f]  "
                    "(presence=%.4f handedness_raw=%.4f)",
                    submission.frame_id,
                    crop.side,
                    decoder._scalar_a_name, scalar_a_raw, scalar_a_dq,
                    decoder._scalar_b_name, scalar_b_raw, scalar_b_dq,
                    float(raw.kpts.min()),
                    float(raw.kpts.max()),
                    raw.presence,
                    raw.handedness_raw,
                )
                self._diag_frames_remaining -= 1

            # One-shot sanity check: the MediaPipe hand wrist landmark is
            # index 0. After back-projection it should land near the pose
            # wrist that anchored this crop. If it is far off, the affine
            # math is wrong and no presence fix will help. Logs a WARNING
            # and never raises so a noisy first frame can't kill the
            # worker.
            if not self._wrist_sanity_done and frame_kpts.shape[0] > 0:
                mp_wrist = frame_kpts[0]
                anchor_wrist = np.asarray(crop.wrist_xy, dtype=np.float32)
                delta = float(np.hypot(
                    mp_wrist[0] - anchor_wrist[0],
                    mp_wrist[1] - anchor_wrist[1],
                ))
                LOG.info(
                    "hand sanity: MediaPipe wrist landmark at (%.1f, %.1f), "
                    "pose anchor at (%.1f, %.1f), delta=%.1f px",
                    float(mp_wrist[0]), float(mp_wrist[1]),
                    float(anchor_wrist[0]), float(anchor_wrist[1]),
                    delta,
                )
                if delta > 80.0:
                    LOG.warning(
                        "hand sanity: back-projected wrist is %.0f px from "
                        "pose anchor — affine may be wrong",
                        delta,
                    )
                self._wrist_sanity_done = True

            self._diag_presence_samples.append(float(raw.presence))
            # Track presence per side so the 5s diag log can show
            # left vs right distributions separately. That is what
            # disambiguates hand-identity issues (one side confident,
            # the other not) from pure presence misclassification.
            side_bucket = self._diag_presence_by_side.get(crop.side)
            if side_bucket is None:
                self._diag_presence_by_side[crop.side] = [float(raw.presence)]
            else:
                side_bucket.append(float(raw.presence))

            # Per-keypoint scores: hand_landmark_lite does not emit
            # per-point confidence, so broadcast the presence score across
            # all 21 landmarks. Overlay uses the same >= threshold check
            # as body pose, which will render/skip all-or-none.
            kpt_scores = np.full(
                (frame_kpts.shape[0],), raw.presence, dtype=np.float32
            )
            raw_hands.append(
                HandLandmarks(
                    kpts=frame_kpts,
                    kpt_scores=kpt_scores,
                    presence=raw.presence,
                    person_id=crop.person_id,
                    side=crop.side,
                )
            )

        # Optional temporal smoothing. HandTracker keys by (person_id, side)
        # so left and right hands of the same person are smoothed
        # independently. Confirmation hysteresis filters out one-frame
        # false positives before the overlay sees them.
        if self._tracker is not None:
            smoothed = self._tracker.update(raw_hands, time.monotonic())
        else:
            smoothed = raw_hands

        elapsed_ms = (time.monotonic() - t0) * 1000.0

        result = HandInferenceResult(
            frame_id=submission.frame_id,
            timestamp_s=time.monotonic(),
            hands=tuple(smoothed),
            latency_ms=elapsed_ms,
            num_crops=len(crops),
        )
        with self._latest_lock:
            self._latest_result = result

        self._inference_count += 1
        now = time.monotonic()
        if now - self._last_log_time > 5.0:
            LOG.info(
                "hand inference: %d frames, latest %.1f ms, %d crops, %d hands",
                self._inference_count,
                elapsed_ms,
                len(crops),
                len(smoothed),
            )
            # 5s diag summary: presence histogram and silent-failure
            # counters, split by side. Reset after each summary so every
            # tick reflects the last 5s window only.
            def _hist(samples: list):
                n = len(samples)
                if n == 0:
                    return 0.0, 0.0, 0.0, 0
                s = sorted(samples)
                return s[0], s[n // 2], s[-1], n

            lo_all, med_all, hi_all, n_all = _hist(self._diag_presence_samples)
            lo_l, med_l, hi_l, n_l = _hist(
                self._diag_presence_by_side.get("left", [])
            )
            lo_r, med_r, hi_r, n_r = _hist(
                self._diag_presence_by_side.get("right", [])
            )
            LOG.info(
                "hand diag (5s): raw_crops=%d rejected_shape=%d "
                "all=[%.3f/%.3f/%.3f n=%d] "
                "left=[%.3f/%.3f/%.3f n=%d] "
                "right=[%.3f/%.3f/%.3f n=%d]",
                self._diag_raw_crops,
                self._diag_rejected_shape,
                lo_all, med_all, hi_all, n_all,
                lo_l, med_l, hi_l, n_l,
                lo_r, med_r, hi_r, n_r,
            )
            self._diag_raw_crops = 0
            self._diag_rejected_shape = 0
            self._diag_presence_samples = []
            self._diag_presence_by_side = {"left": [], "right": []}
            self._last_log_time = now


# ---------------------------------------------------------------------------
# Scheduler helpers — isolated so failure at startup is easy to diagnose
# ---------------------------------------------------------------------------


def _apply_scheduler_priority(configured, priority: int) -> None:
    """Call ``set_scheduler_priority`` on a ConfiguredInferModel.

    Wrapped in a helper so the call site is easy to grep, and so that
    non-fatal errors (if a future pyhailort rename breaks the API) are
    logged with a clear source before raising.
    """
    try:
        configured.set_scheduler_priority(priority)
    except Exception as exc:  # noqa: BLE001
        LOG.exception("set_scheduler_priority(%d) failed: %s", priority, exc)
        raise


def _apply_scheduler_threshold(configured, threshold: int) -> None:
    try:
        configured.set_scheduler_threshold(threshold)
    except Exception as exc:  # noqa: BLE001
        LOG.exception("set_scheduler_threshold(%d) failed: %s", threshold, exc)
        raise


def _apply_scheduler_timeout(configured, timeout_ms: int) -> None:
    try:
        configured.set_scheduler_timeout(timeout_ms)
    except Exception as exc:  # noqa: BLE001
        LOG.exception("set_scheduler_timeout(%d) failed: %s", timeout_ms, exc)
        raise


# ---------------------------------------------------------------------------
# Shared VDevice factory
# ---------------------------------------------------------------------------


def create_shared_vdevice() -> Any:
    """Create a HailoRT ``VDevice`` suitable for sharing between workers.

    The HailoRT scheduler inside a single ``VDevice`` automatically time-
    slices between any ``ConfiguredInferModel`` attached to it, so this is
    the correct way to run multiple models (e.g., pose + object detection)
    on the same physical Hailo chip — attempting to open two independent
    ``VDevice()`` instances raises ``HAILO_OUT_OF_PHYSICAL_DEVICES``.
    """
    from hailo_platform import VDevice  # type: ignore
    return VDevice()


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------


def _format_to_np_dtype(fmt) -> np.dtype:
    """Map a Hailo ``FormatType`` enum to a numpy dtype.

    The Hailo Python API exposes format as ``info.format.type`` which has an
    attribute name like ``FormatType.UINT8`` / ``UINT16`` / ``FLOAT32``. We
    only need the first two for yolov8m_pose.
    """
    type_enum = fmt.type if hasattr(fmt, "type") else fmt
    name = getattr(type_enum, "name", str(type_enum)).upper()
    if "UINT8" in name:
        return np.dtype(np.uint8)
    if "UINT16" in name:
        return np.dtype(np.uint16)
    if "FLOAT32" in name:
        return np.dtype(np.float32)
    raise ValueError(f"unsupported Hailo format {name!r}")
