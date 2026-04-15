import Foundation
import Network
import os

/// UDP video client — handles VID0/BEAT/PAWS/GONE protocol and reassembles chunked H.264 frames.
/// Protocol is byte-compatible with the ESP32-P4 firmware and the Raspberry Pi streamer.
final class UDPVideoClient: @unchecked Sendable {
  private let logger = Logger(subsystem: "com.gata.pivideoviewer", category: "udp")

  /// 28-byte chunk header — must match firmware h264_chunk_header_t exactly.
  struct H264Header {
    let frameId: UInt32
    let width: UInt16
    let height: UInt16
    let timestamp: UInt32
    let totalLen: UInt32
    let chunkIdx: UInt16
    let chunkCount: UInt16
    let frameType: UInt32  // 1 = keyframe
    let reserved: UInt32

    static let size = 28

    init?(data: Data) {
      guard data.count >= Self.size else { return nil }
      var offset = 0
      func read<T: FixedWidthInteger>() -> T {
        let value = data.withUnsafeBytes {
          $0.loadUnaligned(fromByteOffset: offset, as: T.self)
        }
        offset += MemoryLayout<T>.size
        return T(littleEndian: value)
      }
      frameId = read()
      width = read()
      height = read()
      timestamp = read()
      totalLen = read()
      chunkIdx = read()
      chunkCount = read()
      frameType = read()
      reserved = read()
    }

    var isKeyframe: Bool { frameType == 1 }
  }

  // MARK: - Connection state

  private var connection: NWConnection?
  private let queue = DispatchQueue(label: "com.gata.udpvideo", qos: .userInteractive)
  private var heartbeatTimer: DispatchSourceTimer?
  private var registrationAttempts: UInt32 = 0
  private var isRegistered = false

  // MARK: - Frame reassembly state

  /// Slot-based zero-allocation reassembler. Queue-confined to `queue`,
  /// no internal locking — see `UDPFrameReassembler` doc comment.
  private let reassembler = UDPFrameReassembler()

  /// Whether the currently-assembling frame is an IDR. Mirrored here
  /// (rather than read from the reassembler) so `onFrameReady` can pass
  /// it through after `finalizeFrame()` has already cleared the slot.
  private var currentIsKeyframe = false

  private var waitingForIDR = false  // drop P-frames after incomplete IDR

  // MARK: - Watchdog state (Tier 1 recovery)

  /// Monotonic timestamp of the last UDP datagram received from the Pi.
  /// Updated on every packet, including malformed ones — any traffic proves
  /// the network path is alive.
  private var lastPacketAt = DispatchTime.now()

  /// Monotonic timestamp of the last auto-rekey (throttle guard).
  private var lastForceRekeyAt = DispatchTime.now() - DispatchTimeInterval.seconds(60)

  /// Polling timer on `queue` that fires `forceRekey` when packet silence
  /// exceeds `packetSilenceThreshold`. Started alongside the heartbeat
  /// timer and torn down alongside it.
  private var watchdogTimer: DispatchSourceTimer?

  /// Circuit breaker for receive errors — resets on any successful packet.
  private var consecutiveReceiveErrors = 0

  /// Fire `forceRekey` after this many seconds of UDP silence. The
  /// inter-packet gap at 30 fps is ~33ms, so 2.0s is ~60× longer than
  /// healthy jitter and safely shorter than the Pi's 10s IDLE timeout.
  private let packetSilenceThreshold: TimeInterval = 2.0

  /// Minimum spacing between auto-rekeys, to avoid flooding the Pi with
  /// VID0 packets under sustained failure.
  private let forceRekeyMinInterval: TimeInterval = 1.5

  /// Trigger a rekey after this many consecutive NWConnection receive
  /// errors without any successful packet in between.
  private let maxConsecutiveReceiveErrors = 5

  // MARK: - Statistics

  private var totalPackets: UInt64 = 0
  private var totalFrames: UInt64 = 0
  private var assemblyTimeouts: UInt64 = 0

  // MARK: - Callbacks

  /// Called when a complete H.264 frame is reassembled.
  var onFrameReady: ((_ h264Data: Data, _ isKeyframe: Bool) -> Void)?

  /// Called when VACK is received (registration confirmed by firmware).
  var onRegistered: (() -> Void)?

  /// Called on the `queue` when the client autonomously rekeyed (packet
  /// silence or repeated receive errors). Allows the view model to update
  /// status/UI without racing against direct state access.
  var onForcedRekey: ((_ reason: String) -> Void)?

  // MARK: - Connection lifecycle

  func connect(host: String, port: UInt16) {
    disconnect()

    let endpoint = NWEndpoint.hostPort(
      host: NWEndpoint.Host(host),
      port: NWEndpoint.Port(rawValue: port)!
    )
    let params = NWParameters.udp
    let conn = NWConnection(to: endpoint, using: params)
    self.connection = conn

    conn.stateUpdateHandler = { [weak self] state in
      guard let self else { return }
      switch state {
      case .ready:
        self.logger.info("UDP connected to \(host):\(port)")
        self.startVID0RegistrationLoop()
        self.receiveLoop()
      case .failed(let error):
        self.logger.error("UDP connection failed: \(error.localizedDescription)")
      case .cancelled:
        self.logger.info("UDP connection cancelled")
      default:
        break
      }
    }

    conn.start(queue: queue)
  }

  func disconnect() {
    sendControl("GONE")
    stopHeartbeat()
    stopWatchdog()
    isRegistered = false
    registrationAttempts = 0
    consecutiveReceiveErrors = 0
    connection?.cancel()
    connection = nil
  }

  /// Force the Pi to send a fresh keyframe.
  ///
  /// Sends `VID0` (the Pi handles this from any state — IDLE, ACTIVE, or
  /// PAUSED — by forcing a new IDR on the next encode), then locally resets
  /// the reassembly state so that any partial-keyframe debris doesn't
  /// survive into the next frame. Throttled to one rekey per
  /// `forceRekeyMinInterval` seconds to avoid flooding the Pi.
  ///
  /// Safe to call from any thread — the work is hopped onto `queue`.
  func forceRekey(reason: String) {
    queue.async { [weak self] in
      guard let self else { return }
      guard self.connection != nil else { return }

      let now = DispatchTime.now()
      let elapsedNs = now.uptimeNanoseconds &- self.lastForceRekeyAt.uptimeNanoseconds
      let elapsed = Double(elapsedNs) / 1_000_000_000.0
      guard elapsed >= self.forceRekeyMinInterval else {
        self.logger.info(
          "forceRekey throttled (\(String(format: "%.2f", elapsed))s since last): \(reason)"
        )
        return
      }

      self.lastForceRekeyAt = now
      self.logger.warning("forceRekey: \(reason)")
      self.sendControl("VID0")

      // Wipe reassembly so a half-received IDR from the stalled connection
      // doesn't poison the next complete frame.
      self.reassembler.reset()
      self.currentIsKeyframe = false
      self.waitingForIDR = true

      self.onForcedRekey?(reason)
    }
  }

  /// Pause streaming — sends PAWS and stops heartbeat.
  func pause() {
    sendControl("PAWS")
    stopHeartbeat()
    logger.info("Sent PAWS — heartbeat stopped")
  }

  /// Resume streaming — re-sends VID0 to handle firmware IDLE timeout.
  func resume() {
    guard connection != nil else { return }
    /* Always send VID0 on resume — firmware may have timed out to IDLE
       while app was backgrounded, and BEAT is ignored in IDLE state. */
    sendControl("VID0")
    startHeartbeatTimer()
    logger.info("Sent VID0 — resuming stream")
  }

  /// Stop feeding P-frames until the next complete IDR arrives.
  func waitForNextIDR() {
    queue.async { [weak self] in
      guard let self else { return }
      self.waitingForIDR = true
      self.logger.warning("Decoder requested IDR resync")
    }
  }

  // MARK: - Control messages

  /// Send a 4-byte ASCII control message to the firmware.
  private func sendControl(_ msg: String) {
    guard let conn = connection, let data = msg.data(using: .ascii) else { return }
    conn.send(
      content: data,
      completion: .contentProcessed { [weak self] error in
        if let error {
          self?.logger.error("\(msg) send failed: \(error.localizedDescription)")
        }
      })
  }

  // MARK: - VID0 registration

  private func startVID0RegistrationLoop() {
    stopHeartbeat()
    registrationAttempts = 0
    isRegistered = false

    let timer = DispatchSource.makeTimerSource(queue: queue)
    timer.schedule(deadline: .now(), repeating: 1.0)
    timer.setEventHandler { [weak self] in
      guard let self, self.connection != nil else {
        self?.stopHeartbeat()
        return
      }

      self.registrationAttempts += 1
      self.sendControl("VID0")
      self.logger.info("VID0 registration sent (attempt \(self.registrationAttempts))")

      if self.registrationAttempts >= 10 {
        self.stopHeartbeat()
      }
    }
    heartbeatTimer = timer
    timer.resume()
  }

  // MARK: - Heartbeat

  private func startHeartbeatTimer() {
    stopHeartbeat()

    let timer = DispatchSource.makeTimerSource(queue: queue)
    timer.schedule(deadline: .now() + 1.0, repeating: 1.0)
    timer.setEventHandler { [weak self] in
      self?.sendControl("BEAT")
    }
    heartbeatTimer = timer
    timer.resume()

    // Reset the packet-silence clock so the watchdog doesn't fire on
    // stale timestamps from a previous session.
    lastPacketAt = DispatchTime.now()
    consecutiveReceiveErrors = 0
    startWatchdog()
  }

  private func stopHeartbeat() {
    heartbeatTimer?.cancel()
    heartbeatTimer = nil
    stopWatchdog()
  }

  // MARK: - Packet-silence watchdog (Tier 1)

  private func startWatchdog() {
    stopWatchdog()

    let timer = DispatchSource.makeTimerSource(queue: queue)
    // Poll every 0.5s — cheap and gives sub-second reaction time once
    // silence exceeds `packetSilenceThreshold`.
    timer.schedule(deadline: .now() + 0.5, repeating: 0.5)
    timer.setEventHandler { [weak self] in
      guard let self else { return }
      guard self.connection != nil, self.isRegistered else { return }

      let now = DispatchTime.now()
      let elapsedNs = now.uptimeNanoseconds &- self.lastPacketAt.uptimeNanoseconds
      let elapsed = Double(elapsedNs) / 1_000_000_000.0

      if elapsed >= self.packetSilenceThreshold {
        // Bump the clock so we don't retrigger on every 0.5s tick before
        // the rekey has had a chance to produce fresh traffic. The
        // throttle inside `forceRekey` is the real safety net.
        self.lastPacketAt = now
        self.forceRekey(
          reason: "packet silence \(String(format: "%.1f", elapsed))s"
        )
      }
    }
    watchdogTimer = timer
    timer.resume()
  }

  private func stopWatchdog() {
    watchdogTimer?.cancel()
    watchdogTimer = nil
  }

  // MARK: - Receive loop

  private func receiveLoop() {
    guard let conn = connection else { return }

    // Don't busy-loop on a dead socket. If NWConnection is cancelled or
    // permanently failed, the view model's hard recovery will rebuild it.
    switch conn.state {
    case .cancelled, .failed:
      logger.info("receiveLoop stopping — conn state \(String(describing: conn.state))")
      return
    default:
      break
    }

    conn.receiveMessage { [weak self] content, _, _, error in
      guard let self else { return }

      if let error {
        self.consecutiveReceiveErrors += 1
        if self.consecutiveReceiveErrors == 1
          || self.consecutiveReceiveErrors % 10 == 0
        {
          self.logger.error(
            "Receive error #\(self.consecutiveReceiveErrors): \(error.localizedDescription)"
          )
        }
        if self.consecutiveReceiveErrors >= self.maxConsecutiveReceiveErrors {
          self.forceRekey(
            reason: "\(self.consecutiveReceiveErrors) consecutive receive errors"
          )
          self.consecutiveReceiveErrors = 0
        }
        // Keep the loop alive — NWConnection errors on receiveMessage are
        // per-datagram, not fatal to the underlying socket.
        self.receiveLoop()
        return
      }

      if let data = content {
        self.processPacket(data)
      }

      self.receiveLoop()
    }
  }

  private func processPacket(_ data: Data) {
    // Any datagram — valid header or not — proves the network path is
    // alive, so reset the watchdog clock and the receive-error budget.
    lastPacketAt = DispatchTime.now()
    consecutiveReceiveErrors = 0

    /* Check for 4-byte control messages (VACK) */
    if data.count == 4, let msg = String(data: data, encoding: .ascii) {
      if msg == "VACK" {
        logger.info("VACK received — registration confirmed")
        isRegistered = true
        startHeartbeatTimer()
        onRegistered?()
        return
      }
    }

    totalPackets += 1

    /* Legacy fallback: stop VID0 retries on first video packet if no VACK */
    if !isRegistered {
      isRegistered = true
      startHeartbeatTimer()
      logger.info("First video packet — switching to heartbeat (no VACK)")
      onRegistered?()
    }

    guard let header = H264Header(data: data) else {
      logger.warning("Packet too small for header: \(data.count) bytes")
      return
    }

    let payload = data.dropFirst(H264Header.size)

    /* New frame detected — finalize assembly bookkeeping for the
     * previous frame and start a fresh slot. */
    if reassembler.currentFrameId != header.frameId {
      /* If we were mid-assembly on a different frame, that frame is now
       * abandoned — count the timeout and, if it was an IDR, force the
       * next complete IDR before resuming P-frame decode. */
      if let staleFrameId = reassembler.currentFrameId {
        assemblyTimeouts += 1
        if currentIsKeyframe {
          waitingForIDR = true  // lost IDR → drop P-frames until next complete IDR
        }
        if assemblyTimeouts % 100 == 1 {
          logger.warning(
            "Frame \(staleFrameId) incomplete before frame \(header.frameId) arrived"
          )
        }
        reassembler.reset()
      }

      currentIsKeyframe = header.frameType == 1
      _ = reassembler.startFrame(
        frameId: header.frameId,
        chunkCount: Int(header.chunkCount),
        expectedLen: Int(header.totalLen),
        isKeyframe: currentIsKeyframe
      )
    }

    /* Place chunk at correct offset in the active slot. The reassembler
     * memcpys directly into its preallocated buffer — no per-chunk
     * Data allocation. */
    payload.withUnsafeBytes { bytes in
      reassembler.placeChunk(
        chunkIdx: Int(header.chunkIdx),
        offset: Int(header.chunkIdx) * 1400,
        payload: bytes
      )
    }

    /* Check if frame is complete */
    if reassembler.isComplete() {
      totalFrames += 1

      if waitingForIDR && !currentIsKeyframe {
        /* Drop P-frames until we get a complete IDR. Still finalize the
         * slot so it's free for the next frame. */
        _ = reassembler.finalizeFrame()
        return
      }
      if currentIsKeyframe {
        waitingForIDR = false  // got a complete IDR — resume decoding
      }

      if let frameData = reassembler.finalizeFrame() {
        if totalFrames % 300 == 0 {
          logger.info(
            "Frame \(self.totalFrames): \(frameData.count) bytes \(self.currentIsKeyframe ? "[IDR]" : "")"
          )
        }
        onFrameReady?(frameData, currentIsKeyframe)
      }
    }
  }
}
