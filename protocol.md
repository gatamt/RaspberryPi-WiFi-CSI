# Wire Protocol — RaspberryPi-WiFi-CSI

This is the single source of truth for the wire format between the Raspberry Pi streamer and the iOS `PiVideoViewer` app. Both sides must match this byte-for-byte.

The protocol is copied verbatim from the ESP32-P4 `P4-idf6` project so that the iOS client can remain nearly unchanged. Any deviation here is a bug.

## Network

- Transport: UDP
- Server port: **3334** (video control + stream)
- Subnet: iPhone Personal Hotspot, typically `172.20.10.0/28`
- Server (Pi) binds `INADDR_ANY:3334`
- Client (iPhone) uses an ephemeral source port

## Discovery

### mDNS (primary)

- Service type: `_wificsi._udp`
- Port: 3334
- TXT records:
  - `port=3334`
  - `codec=h264`
  - `width=1280`
  - `height=720`
  - `fps=30`
  - `proto=1`
  - `source=rpi5`
  - `overlay=baked`

### Hotspot probe (fallback)

iPhone app sweeps `172.20.10.2` through `172.20.10.20` with VID0 probes once per second. Pi must respond with VACK when VID0 is received.

## Control messages

All control messages are exactly 4 bytes of ASCII, no trailing NUL.

| Direction | Code | Bytes (hex) | Meaning |
|---|---|---|---|
| Client → Server | `VID0` | `56 49 44 30` | Start streaming / register client |
| Client → Server | `BEAT` | `42 45 41 54` | Heartbeat — send every 1 second |
| Client → Server | `PAWS` | `50 41 57 53` | Pause — app entered background |
| Client → Server | `GONE` | `47 4F 4E 45` | Disconnect — app shutting down |
| Server → Client | `VACK` | `56 41 43 4B` | VID0 acknowledged |

## State machine (server)

```
IDLE ──VID0──▶ ACTIVE ──PAWS──▶ PAUSED
  ▲              │                 │
  │         timeout/GONE        BEAT
  │              ▼                 ▼
  └──────────── IDLE             ACTIVE
                 ▲                 │
                 └─ timeout/GONE ──┘
```

### Transitions

- **IDLE → ACTIVE** on VID0 or BEAT (re-registration). Capture the client's address from the recvfrom. Send VACK. Set `force_idr = true`.
- **ACTIVE → PAUSED** on PAWS. Stop sending frames but keep the client address.
- **PAUSED → ACTIVE** on BEAT. Set `force_idr = true`.
- **any → IDLE** on GONE, on heartbeat timeout (10 s without BEAT or VID0), or on station disconnect.

### Heartbeat

- Client sends BEAT every 1 second after VACK.
- Server tracks `last_heartbeat_us` and resets to IDLE if more than 10 000 000 µs elapse.
- 10 s tolerates several lost UDP packets plus short network stalls.

### Force IDR

On transition IDLE → ACTIVE or PAUSED → ACTIVE, the server sets `force_idr = true`. The encoder must emit an IDR on its next frame so the client can rebuild format description and decode.

## Video chunk header (28 bytes, little-endian)

```
offset  size  field              notes
0       4     frame_id           u32 LE, monotonic
4       2     width              u16 LE, always 1280 for MVP
6       2     height             u16 LE, always 720 for MVP
8       4     timestamp          u32 LE, milliseconds (monotonic)
12      4     total_len          u32 LE, total H.264 bytes for this frame
16      2     chunk_idx          u16 LE, 0-based
18      2     chunk_count        u16 LE, number of chunks for this frame
20      4     frame_type         u32 LE, 1 = IDR/keyframe, 0 = P-frame
24      4     reserved           u32 LE, zero for now
-------
28      total header bytes
```

Python `struct` format: `<IHHIIHHII` (produces exactly 28 bytes — verify with `struct.calcsize`).

## Video payload

- Codec: H.264 Annex-B (start codes `00 00 00 01` before each NAL unit)
- Encoder settings (Pi side):
  - Resolution 1280×720
  - 30 FPS
  - Bitrate 2 500 000 bit/s
  - GOP 30
  - Preset `ultrafast`, tune `zerolatency`
  - Profile: baseline or main (whatever x264 ultrafast chooses)
- Each encoded frame is chunked into UDP datagrams with **max 1400 bytes of H.264 payload per chunk** (header is always 28 bytes — datagram total 1428).
- Chunks are sent in order. Between chunks, sleep 200 µs to avoid flooding the Wi-Fi TX queue.
- The SPS, PPS, and IDR slice together appear in the first frame after force_idr.

## Frame reassembly (client side)

- Client tracks `currentFrameId`, `expectedChunkCount`, and a set of `receivedChunks`.
- When `header.frame_id` differs from `currentFrameId`, the previous frame is discarded if incomplete (counted as a drop) and a new reassembly buffer is allocated based on `header.total_len`.
- Each chunk is written to `frame_buffer[chunk_idx * 1400 ...]`.
- When `len(receivedChunks) == expectedChunkCount`, the frame is complete and dispatched to the decoder.
- If a keyframe loses chunks, the client enters `waitingForIDR = true` and drops P-frames until the next complete IDR arrives.

## Encoding recovery

- On VideoToolbox decode failure, client resets the decompression session and marks `waitingForIDR = true`. Server must then emit an IDR — the client normally does this implicitly by being registered, so the next keyframe (GOP distance) will restore video.
- For faster recovery, the client can re-send VID0 which triggers another force_idr on the server.

## iOS Logger subsystem

All logging on the iOS side uses subsystem `com.gata.pivideoviewer` with categories:

- `discovery` — Bonjour + hotspot probe events
- `udp` — VID0/VACK/BEAT protocol events
- `reassembly` — Frame reassembly stats
- `decoder` — VideoToolbox session + decode events
- `frames` — Frame counts, FPS, bandwidth
- `ui` — View lifecycle + scene phase

## Python logger names (Pi side)

- `pi_streamer.main`
- `pi_streamer.camera`
- `pi_streamer.inference`
- `pi_streamer.overlay`
- `pi_streamer.encoder`
- `pi_streamer.udp_protocol`
- `pi_streamer.wire_format`
