import Foundation

/// Slot-based H.264 frame reassembler.
///
/// Maintains 5 fixed-capacity scratch buffers (100_000 bytes each) and copies
/// incoming UDP chunk payloads directly into them via `memcpy`, avoiding the
/// per-chunk `Data.replaceSubrange` allocation churn the previous implementation
/// suffered from. Only one slot is "active" at a time — frames complete and
/// finalize before the next one starts — but having multiple physical slots
/// gives us headroom for trivial buffer reuse without reallocation.
///
/// # Thread-safety
///
/// **This class has no internal locking.** It is queue-confined: the caller
/// (UDPVideoClient) must only invoke methods from its serial dispatch queue
/// (`com.gata.udpvideo`). Calling from multiple threads is undefined behavior.
final class UDPFrameReassembler {

  // MARK: - Configuration

  /// Maximum number of slots held in flight. Only one is active at a time;
  /// the rest are scratch space available for reuse without reallocation.
  static let slotCount = 5

  /// Capacity of each slot in bytes. 100 kB is comfortably above the largest
  /// observed H.264 frame from the Pi streamer (P-frames ~2-10 kB,
  /// IDRs ~30-60 kB at 720p).
  static let slotSize = 100_000

  /// MTU-aligned chunk payload size used by the firmware/streamer protocol.
  /// Kept here as a reference constant — the caller passes the chunk's
  /// absolute byte offset directly into `placeChunk`.
  static let chunkPayloadSize = 1400

  // MARK: - Slot storage

  /// Per-slot bookkeeping. Each slot owns one `UnsafeMutableRawBufferPointer`
  /// of length `slotSize` allocated once at init and freed in `deinit`.
  private struct Slot {
    var buffer: UnsafeMutableRawBufferPointer
    var inUse: Bool
    var frameId: UInt32
    var chunkCount: Int
    var expectedLen: Int
    var receivedChunks: Set<Int>
    var isKeyframe: Bool
  }

  private var slots: [Slot]

  /// Index of the currently active slot (the one being filled), or `nil`
  /// if no frame is in flight.
  private var activeSlotIdx: Int?

  // MARK: - Init / deinit

  init() {
    var initial: [Slot] = []
    initial.reserveCapacity(Self.slotCount)
    for _ in 0..<Self.slotCount {
      let buf = UnsafeMutableRawBufferPointer.allocate(
        byteCount: Self.slotSize,
        alignment: MemoryLayout<UInt64>.alignment
      )
      initial.append(
        Slot(
          buffer: buf,
          inUse: false,
          frameId: 0,
          chunkCount: 0,
          expectedLen: 0,
          receivedChunks: [],
          isKeyframe: false
        )
      )
    }
    self.slots = initial
  }

  deinit {
    for slot in slots {
      slot.buffer.deallocate()
    }
  }

  // MARK: - Public API

  /// FrameId of the currently active slot, or nil if no frame is in flight.
  var currentFrameId: UInt32? {
    guard let idx = activeSlotIdx else { return nil }
    return slots[idx].frameId
  }

  /// Begin a new frame. Picks a free slot, populates its metadata, and
  /// returns the slot index. Returns `nil` if all slots are in use (which
  /// in practice should never happen because frames complete before the
  /// next one starts — but the caller can use `nil` to detect a stuck slot).
  ///
  /// - Parameters:
  ///   - frameId: 32-bit frame id from the chunk header.
  ///   - chunkCount: total number of chunks expected for this frame.
  ///   - expectedLen: total H.264 payload length in bytes.
  ///   - isKeyframe: true for IDRs.
  /// - Returns: the slot index that now holds the frame, or nil on failure.
  @discardableResult
  func startFrame(
    frameId: UInt32,
    chunkCount: Int,
    expectedLen: Int,
    isKeyframe: Bool
  ) -> Int? {
    /* Find a free slot. activeSlotIdx is None at this point in normal
     * flow, so the first free slot we find is fine. */
    guard let freeIdx = slots.firstIndex(where: { !$0.inUse }) else {
      return nil
    }

    /* Refuse frames that wouldn't fit the slot — silently dropping is
     * safer than corrupting memory. The reassembler can never recover
     * a frame larger than slotSize, and the watchdog will eventually
     * rekey if the stream gets stuck. */
    guard expectedLen <= Self.slotSize else {
      return nil
    }

    slots[freeIdx].inUse = true
    slots[freeIdx].frameId = frameId
    slots[freeIdx].chunkCount = chunkCount
    slots[freeIdx].expectedLen = expectedLen
    slots[freeIdx].isKeyframe = isKeyframe
    slots[freeIdx].receivedChunks.removeAll(keepingCapacity: true)

    activeSlotIdx = freeIdx
    return freeIdx
  }

  /// Copy a chunk payload into the active slot at the given byte offset.
  ///
  /// Bounds-checked against both `slotSize` and the active frame's
  /// declared `expectedLen` — out-of-range chunks are silently dropped to
  /// avoid memory corruption from malformed or spoofed packets.
  ///
  /// - Parameters:
  ///   - chunkIdx: chunk index (used to track completeness).
  ///   - offset: absolute byte offset into the frame buffer.
  ///   - payload: raw UDP payload (already with the header stripped).
  func placeChunk(
    chunkIdx: Int,
    offset: Int,
    payload: UnsafeRawBufferPointer
  ) {
    guard let idx = activeSlotIdx else { return }
    guard offset >= 0 else { return }

    let slot = slots[idx]

    /* Clamp the copy length so we never write past the slot or past
     * the declared frame length. */
    let maxLen = min(Self.slotSize, slot.expectedLen) - offset
    guard maxLen > 0 else { return }
    let copyLen = min(payload.count, maxLen)
    guard copyLen > 0 else { return }
    guard let payloadBase = payload.baseAddress else { return }

    let dest = slot.buffer.baseAddress!.advanced(by: offset)
    dest.copyMemory(from: payloadBase, byteCount: copyLen)

    slots[idx].receivedChunks.insert(chunkIdx)
  }

  /// True when all expected chunks have been received for the active frame.
  func isComplete() -> Bool {
    guard let idx = activeSlotIdx else { return false }
    let slot = slots[idx]
    return slot.receivedChunks.count == slot.chunkCount
  }

  /// Snapshot the active slot into a heap-allocated `Data`, free the slot,
  /// and clear the active pointer. Returns nil if there is no active frame.
  ///
  /// The returned `Data` owns its own copy of the bytes — the slot can be
  /// safely reused for the next frame as soon as this returns.
  func finalizeFrame() -> Data? {
    guard let idx = activeSlotIdx else { return nil }
    let slot = slots[idx]
    let length = min(slot.expectedLen, Self.slotSize)

    /* Copy out into a fresh Data — the decoder owns its own buffer
     * lifetime, decoupled from this slot's reuse. */
    let data = Data(bytes: slot.buffer.baseAddress!, count: length)

    slots[idx].inUse = false
    slots[idx].receivedChunks.removeAll(keepingCapacity: true)
    activeSlotIdx = nil

    return data
  }

  /// Mark all slots free and clear the active pointer. Called on
  /// `forceRekey` to wipe partial-frame debris from a stalled connection.
  func reset() {
    for i in 0..<slots.count {
      slots[i].inUse = false
      slots[i].receivedChunks.removeAll(keepingCapacity: true)
    }
    activeSlotIdx = nil
  }
}
