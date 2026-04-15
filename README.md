# RaspberryPi-WiFi-CSI

> Real-time pose estimation video streamer on Raspberry Pi 5 + Hailo-10H NPU, with a SwiftUI iOS viewer. Part of a larger through-wall WiFi-CSI sensing project — this repository holds only the vision pipeline half.

## Overview

The Pi 5 captures 1280x720 at 30 fps from a Camera Module 3 Wide (IMX708), runs three Hailo-10H neural networks in parallel (YOLOv8-m pose, YOLO26-m object, MediaPipe hand landmarks) through a single shared `VDevice`, bakes the detections straight into the video with OpenCV, encodes H.264 via libx264 `ultrafast`/`zerolatency`, and streams the result over UDP to an iPhone on Personal Hotspot. Overlay compositing happens on the Pi so the iOS app is pure decode — there is no second bitmap channel to synchronise.

The vision pipeline produces pose detections intended as ground truth for a separate WiFi Channel State Information (CSI) sensing stack. That CSI array hardware and the signal-processing code are tracked in a different repository and are not part of this one.

Two implementations sit side by side. The Python MVP (`pi/pi_streamer/`) is stable and holds ~30 fps on real hardware. A C11 rewrite (`pi/pi_streamer_c/`) targets the Pi 5's Cortex-A76 cores: lockless single-producer/single-consumer ring buffers, a flat Samek-style state machine, io_uring UDP submission, and a byte-identical wire protocol verified against the Python reference by unit test. Interfaces and mock backends exist for every module so the C11 code builds and tests on macOS or x86 Linux hosts with no Pi hardware present.

## Hardware

| Component | Part | Purpose |
|---|---|---|
| SBC | Raspberry Pi 5 (BCM2712, Cortex-A76 @ 2.4 GHz) | Host |
| NPU | Hailo-10H AI HAT+ (HailoRT 5.1.1+) | Inference — pose, object, hand-landmark |
| Camera | Raspberry Pi Camera Module 3 Wide (Sony IMX708) | MIPI-CSI, 12 MP, 120 degree FOV |
| Client | iPhone on Personal Hotspot (`172.20.10.0/28`) | Viewer + BLE provisioning |

## Architecture

Data path:

```
Pi Camera 3 Wide (IMX708)
        │ libcamera / picamera2
        ▼
  RGB frame ring buffer
        │
        ├───▶ Hailo shared VDevice ───▶ pose     (priority 18)
        │                         ───▶ object   (priority 17)
        │                         ───▶ hand     (priority 15)
        │                                │
        │                                ▼
        │                        async result fan-in
        │                                │
        ▼                                │
   OpenCV overlay ◀──────────────────────┘
   (bbox + COCO-17 skeleton + hand landmarks + HUD)
        │
        ▼
   libx264 encode  (ultrafast / zerolatency, GOP 30, 2.5 Mbit/s)
        │
        ▼
   28-byte chunker  (1400 B payload max, 200 us inter-chunk pacing)
        │
        ▼
   UDP :3334 ───▶ iPhone viewer (AVSampleBufferDisplayLayer)
```

A single Hailo `VDevice` is opened and three `ConfiguredInferModel` instances bind on top of it. The HailoRT scheduler time-slices the device between models via integer priorities (pose 18, object 17, hand 15, normal 16). Opening a second `VDevice` raises `HAILO_OUT_OF_PHYSICAL_DEVICES`, so the shared handle must stay alive for the lifetime of the pipeline.

Control path:

```
iPhone (BLE central) ─── GATT ───▶ pi_ble daemon
                                       │
                                       ├── PIN authentication
                                       ├── WiFi provisioning via nmcli
                                       └── systemctl start pi-streamer-c
```

The C11 rewrite currently runs a single-threaded run-to-completion event loop in `pipeline.c`. Per-thread CPU pinning, `SCHED_FIFO` priorities, `mlockall()`, and `isolcpus` kernel cmdline are planned — the systemd unit already grants `CAP_SYS_NICE`, `CAP_NET_RAW`, and `LimitMEMLOCK=infinity`, and `src/rt.c` + `test_rt.c` exist as the wiring hooks, but the multi-threaded split has not been merged into the main loop yet.

## Directory layout

```
RaspberryPi-WiFi-CSI/
├── pi/
│   ├── pi_streamer/         # Python MVP (stable, ~30 fps)
│   ├── pi_streamer_c/       # C11 rewrite (in progress)
│   ├── pi_ble/              # BLE provisioning daemon (PIN + WiFi)
│   ├── systemd/             # pi-streamer-c.service, pi-ble.service
│   ├── avahi/               # wificsi.service (Bonjour advertisement)
│   ├── scripts/             # install.sh, install_ble.sh, run.sh
│   ├── pyproject.toml
│   └── requirements.txt
├── iOS/PiVideoViewer/       # SwiftUI app, AVSampleBufferDisplayLayer H.264 path
├── protocol.md              # 28-byte UDP wire format (source of truth)
└── README.md
```

Python MVP modules under `pi/pi_streamer/`:

- `camera.py` — picamera2 wrapper, IMX708 sensor mode, optional low-res stream for inference
- `encoder.py` — PyAV + libx264, `ultrafast`/`zerolatency`, forced keyframe on demand
- `inference.py` — Hailo async workers for pose, object, and hand-landmark HEFs
- `postprocess.py` / `postprocess_yolo26.py` / `postprocess_hand.py` — decoders + dequant
- `overlay.py` — OpenCV COCO-17 skeleton, bounding boxes, labels, HUD
- `tracker.py` — EMA plus hysteresis smoothing for pose and object detections
- `hand_roi.py` — wrist-anchored 224x224 ROI derived from pose keypoints
- `wire_format.py` — 28-byte little-endian chunk header (`<IHHIIHHII`)
- `udp_protocol.py` — VID0/BEAT/PAWS/GONE state machine, force-IDR on IDLE to ACTIVE
- `main.py` — wires camera to inference to overlay to encoder to UDP and runs the main loop

C11 rewrite modules under `pi/pi_streamer_c/src/`:

- `ring_buffer.c` — SPSC lockless ring, cache-line aligned `_Atomic size_t` head/tail
- `wire_format.c` — byte-identical packer/unpacker for the 28-byte chunk header
- `state_machine.c` — flat Samek-style HSM with explicit transition lookup table
- `camera_mock.c` / `camera_libcamera.cpp` — mock + real libcamera backends
- `encoder_mock.c` / `encoder_x264.c` — mock + real libx264 backends
- `udp_sender_mock.c` / `udp_sender_iouring.c` — mock + real io_uring backends
- `inference_mock.c` / `inference_hailort.cpp` — mock + real HailoRT backends
- `overlay.c`, `postprocess.c` — overlay compositor and detection decoders
- `pipeline.c` — main event loop wiring all modules together
- `rt.c` — real-time scheduling hooks for thread pinning and `SCHED_FIFO`
- `logger.c` — RT-safe logger (no `malloc` on hot path)

## Build and run

### Pi side — Python MVP

Prerequisites on the Pi:

- Raspberry Pi OS Bookworm (12) or newer, 64-bit
- HailoRT 5.1.1 or newer (`libhailort.so` + `hailortcli` + `hailo_platform` python bindings)
- `picamera2`, `libcamera`, `x264` from apt
- Hailo HEFs — **not shipped in this repository, download them separately** from the Hailo Model Zoo ([hailo.ai/developer-zone](https://hailo.ai/developer-zone/)) and drop them at `/home/pi/models/`:
  - `yolov8m_pose.hef`
  - `yolo26m.hef`
  - `hand_landmark_lite.hef`
  - Any path is fine — override with `--hef`, `--object-hef`, `--hand-hef` on the command line if you place them elsewhere.

```bash
cd pi
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pi_streamer.main --log-level INFO
```

The `--system-site-packages` flag is required — `hailo_platform` is only installed system-wide via the Hailo apt package, it is not on PyPI.

Useful flags for bring-up:

| Flag | What it does |
|---|---|
| `--no-inference` | Skip all Hailo workers. Camera + encode + UDP only. |
| `--no-object` | Run pose only, skip YOLO26-m. |
| `--no-hands` | Run pose + object, skip hand landmarks. |
| `--no-smoothing` | Disable EMA + hysteresis. Raw decoder output every frame. |
| `--hand-roi wrist` | Wrist-anchored 224x224 ROI from pose keypoints (the only mode currently implemented). |
| `--bitrate` | Override the 2.5 Mbit/s default. |

### Pi side — C11 rewrite (experimental)

```bash
cd pi/pi_streamer_c
./scripts/install_deps.sh    # apt: libcamera-dev, libx264-dev, liburing-dev, clang-tidy, cppcheck, valgrind
./scripts/build.sh           # CMake + Ninja; auto-detects aarch64 and sets PI_TARGET=ON
./scripts/run_tests.sh       # host build, all unit tests under ASAN + UBSAN
```

Host build on Mac or x86 Linux for development — all backends swap to mocks automatically:

```bash
cd pi/pi_streamer_c
cmake -S . -B build-host -DPI_TARGET=OFF -DPI_ENABLE_ASAN=ON -DPI_ENABLE_UBSAN=ON
cmake --build build-host
ctest --test-dir build-host --output-on-failure
```

Compile flags are enforced in `CMakeLists.txt`: `-std=c11 -O3 -ftree-vectorize -Wall -Wextra -Wshadow -Wdouble-promotion -Wformat=2 -Wmissing-prototypes -Wstrict-prototypes -Wpointer-arith -Wcast-qual -Werror`, stack protector, hidden visibility. On `PI_TARGET=ON` it adds `-mcpu=cortex-a76 -mtune=cortex-a76 -march=armv8.2-a+crc+simd`.

### iOS viewer

```bash
cd iOS/PiVideoViewer
xcodegen generate               # regenerates PiVideoViewer.xcodeproj from project.yml
open PiVideoViewer.xcodeproj
# Set your Team ID in Signing & Capabilities, then run on a real iPhone
```

Deployment target is iOS 16. The app uses `AVSampleBufferDisplayLayer` directly for zero-copy H.264 rendering — no per-frame `VTDecompressionSession` + `CIImage` + `CGImage` round-trip in the hot path.

Command line build without signing:

```bash
xcodebuild \
  -project PiVideoViewer.xcodeproj \
  -scheme PiVideoViewer \
  -destination 'generic/platform=iOS' \
  -configuration Debug \
  CODE_SIGNING_ALLOWED=NO \
  build
```

### BLE provisioning

```bash
sudo ./pi/scripts/install_ble.sh
sudo systemctl enable --now pi-ble.service
```

First-run flow: open the iOS app, tap the gear icon, the app advertises over BLE, the Pi answers, you enter a PIN from the app and then WiFi SSID + password. The daemon hands the credentials to `nmcli`, the Pi associates, and the iOS side starts the stream over the Personal Hotspot subnet.

## Configuration

| Placeholder | Where | What to change |
|---|---|---|
| `YOUR_TEAM_ID` | `iOS/PiVideoViewer/project.yml`, `iOS/PiVideoViewer/PiVideoViewer.xcodeproj/project.pbxproj` | Apple Developer Team ID (10-char alphanumeric). Rerun `xcodegen generate` after editing `project.yml`. |
| `raspberrypi.local` / user `pi` | `pi/systemd/pi-streamer-c.service`, `pi/systemd/pi-ble.service`, `pi/pi_streamer_c/scripts/deploy.sh`, `pi/pi_ble/stream.py` | Pi hostname and Linux user. Change across all files if your Pi is not named `raspberrypi` or does not use the `pi` user. |
| `/home/pi/models/*.hef` | `pi/pi_streamer/main.py` defaults | Path to the three Hailo HEFs. Override at the command line with `--hef`, `--object-hef`, `--hand-hef`. |

## Wire protocol

28-byte chunk header, little-endian. Full details in `protocol.md`:

```
offset  size  field         notes
0       4     frame_id      u32, monotonic
4       2     width         u16, 1280 for MVP
6       2     height        u16, 720 for MVP
8       4     timestamp_ms  u32, monotonic
12      4     total_len     u32, total H.264 bytes for this frame
16      2     chunk_idx     u16, 0-based
18      2     chunk_count   u16, total chunks for this frame
20      4     frame_type    u32, 1 = IDR, 0 = P
24      4     reserved      u32, zero
-------
28      total header bytes
```

Python `struct` format: `<IHHIIHHII` (exactly 28 bytes). Each UDP datagram is header + up to 1400 bytes of H.264 Annex-B payload, so full datagram size is at most 1428 bytes. Chunks are paced with 200 us sleeps between sends to avoid flooding the hotspot TX queue.

Discovery over Bonjour: `_wificsi._udp` on port 3334, TXT records `codec=h264 width=1280 height=720 fps=30`. Fallback path: the iOS client sweeps `172.20.10.2` through `.20` with VID0 probes once per second and the Pi answers with VACK.

Control messages (4-byte ASCII, no NUL):

| Direction | Code | Meaning |
|---|---|---|
| Client -> Server | `VID0` | Register as viewer and request stream |
| Server -> Client | `VACK` | VID0 acknowledged, stream starts |
| Client -> Server | `BEAT` | Heartbeat, one per second |
| Client -> Server | `PAWS` | Pause (app entered background) |
| Client -> Server | `GONE` | Clean disconnect |

Server state machine: `IDLE` takes `VID0` or re-registering `BEAT` to `ACTIVE` (force-IDR set). `ACTIVE` takes `PAWS` to `PAUSED`, which comes back on the next `BEAT` (force-IDR set again). `GONE` or a 10-second heartbeat timeout drops back to `IDLE`.

## Testing

### Python MVP self-tests

Three modules ship with executable self-tests:

```bash
cd pi
source .venv/bin/activate
python -m pi_streamer.postprocess_hand   # hand decoder + affine back-projection
python -m pi_streamer.hand_roi           # wrist-anchored ROI geometry
python -m pi_streamer.tracker            # EMA + hysteresis smoothing
```

### C11 rewrite unit tests

13 CTest unit binaries, registered in `pi/pi_streamer_c/tests/unit/CMakeLists.txt`:

| Test | Covers |
|---|---|
| `test_logger` | RT-safe logger, no `malloc` on hot path |
| `test_ring_buffer` | SPSC lockless ring, single-producer + single-consumer stress |
| `test_wire_format` | Byte-identical pack/unpack vs. Python reference vector |
| `test_state_machine` | Flat HSM, full transition table coverage |
| `test_camera` | Camera interface + mock backend |
| `test_encoder` | Encoder interface + mock backend |
| `test_udp_sender` | UDP sender interface + mock backend |
| `test_inference` | Inference interface + mock backend |
| `test_inference_state` | Inference state tracking + latest-result fan-in |
| `test_overlay` | Overlay compositor geometry |
| `test_postprocess` | Detection decoders + dequant |
| `test_pipeline` | End-to-end camera -> inference -> encoder -> wire -> UDP flow |
| `test_rt` | Real-time scheduling hooks |

All tests run under `-DPI_ENABLE_ASAN=ON -DPI_ENABLE_UBSAN=ON` on both host (macOS + x86 Linux) and target (Pi 5 Cortex-A76).

## Status

- **Python MVP** — stable at ~30 fps on Pi 5 + Hailo-10H for pose + object. Hand landmark uses wrist-anchored ROI (`--hand-roi wrist`). Overlay is baked into the encoded frame; the iOS side does not render detections.
- **C11 rewrite** — skeleton, mock backends, and host tests are green. Real libcamera / libx264 / io_uring / HailoRT backends are partially implemented (the source files exist side-by-side with the mocks). Single-threaded event loop; multi-threaded split with `pthread_setaffinity_np` + `SCHED_FIFO` not yet merged into `pipeline.c`.
- **iOS viewer** — builds on Xcode 16+, iOS 16 deployment target. `AVSampleBufferDisplayLayer` renders H.264 directly. BLE provisioning, Bonjour discovery, and the hotspot-probe fallback path are wired in.
- **WiFi CSI side** — tracked in a separate repository. Not included here.

## License

MIT — see `LICENSE`.

## Acknowledgments

- Wire protocol design derived from an ESP32-P4 sibling streamer project so the iOS client stays byte-compatible across both hardware backends.
- HEFs for pose, object, and hand-landmark inference come from the Hailo Model Zoo (`yolov8m_pose`, `yolo26m`, `hand_landmark_lite`).
