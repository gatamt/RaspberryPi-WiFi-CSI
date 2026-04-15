import XCTest

@testable import PiVideoViewer

/// Behavioral unit tests for `UDPFrameReassembler`.
///
/// All tests assume the queue-confined contract — they invoke methods
/// from the test thread, which is fine because the reassembler has no
/// internal locking and these tests don't share instances across threads.
final class UDPFrameReassemblerTests: XCTestCase {

  // MARK: - Helpers

  /// Build a `UnsafeRawBufferPointer` view over a byte array for use
  /// with `placeChunk(payload:)`. The closure-style is necessary because
  /// `withUnsafeBytes` only exposes the pointer for its lifetime.
  private func withChunk(_ length: Int, _ body: (UnsafeRawBufferPointer) -> Void) {
    var bytes = [UInt8](repeating: 0xAB, count: length)
    bytes.withUnsafeBytes(body)
  }

  // MARK: - Tests

  /// Three chunks placed in order: 0, 1, 2 → frame should complete and
  /// finalize cleanly.
  func testInOrderChunksComplete() {
    let r = UDPFrameReassembler()
    let chunkSize = 1400
    let chunkCount = 3
    let totalLen = chunkSize * chunkCount

    let slot = r.startFrame(
      frameId: 42, chunkCount: chunkCount, expectedLen: totalLen, isKeyframe: true)
    XCTAssertNotNil(slot)
    XCTAssertEqual(r.currentFrameId, 42)
    XCTAssertFalse(r.isComplete())

    for idx in 0..<chunkCount {
      withChunk(chunkSize) { bytes in
        r.placeChunk(chunkIdx: idx, offset: idx * chunkSize, payload: bytes)
      }
    }

    XCTAssertTrue(r.isComplete())

    let data = r.finalizeFrame()
    XCTAssertNotNil(data)
    XCTAssertEqual(data?.count, totalLen)
    XCTAssertNil(r.currentFrameId)
  }

  /// Same three chunks placed in the order 2, 0, 1 — the reassembler
  /// must not depend on arrival order.
  func testOutOfOrderChunksComplete() {
    let r = UDPFrameReassembler()
    let chunkSize = 1400
    let chunkCount = 3
    let totalLen = chunkSize * chunkCount

    XCTAssertNotNil(
      r.startFrame(
        frameId: 7, chunkCount: chunkCount, expectedLen: totalLen, isKeyframe: false))

    for idx in [2, 0, 1] {
      withChunk(chunkSize) { bytes in
        r.placeChunk(chunkIdx: idx, offset: idx * chunkSize, payload: bytes)
      }
    }

    XCTAssertTrue(r.isComplete())
    XCTAssertNotNil(r.finalizeFrame())
  }

  /// A frame with one chunk missing (chunk 2 of 3) must never report
  /// complete, regardless of how many times we re-receive chunks 0 and 1.
  func testMissingChunkNeverCompletes() {
    let r = UDPFrameReassembler()
    let chunkSize = 1400
    let chunkCount = 3
    let totalLen = chunkSize * chunkCount

    XCTAssertNotNil(
      r.startFrame(
        frameId: 100, chunkCount: chunkCount, expectedLen: totalLen, isKeyframe: false))

    withChunk(chunkSize) { bytes in
      r.placeChunk(chunkIdx: 0, offset: 0, payload: bytes)
    }
    withChunk(chunkSize) { bytes in
      r.placeChunk(chunkIdx: 1, offset: chunkSize, payload: bytes)
    }
    XCTAssertFalse(r.isComplete())

    /* Re-receiving the same chunks must not flip completeness. */
    withChunk(chunkSize) { bytes in
      r.placeChunk(chunkIdx: 0, offset: 0, payload: bytes)
    }
    XCTAssertFalse(r.isComplete())
  }

  /// `reset()` after a partial frame must clear the active slot so that
  /// `currentFrameId` returns nil.
  func testResetWipesState() {
    let r = UDPFrameReassembler()

    XCTAssertNotNil(
      r.startFrame(frameId: 555, chunkCount: 2, expectedLen: 2800, isKeyframe: true))
    withChunk(1400) { bytes in
      r.placeChunk(chunkIdx: 0, offset: 0, payload: bytes)
    }

    XCTAssertEqual(r.currentFrameId, 555)
    XCTAssertFalse(r.isComplete())

    r.reset()

    XCTAssertNil(r.currentFrameId)
    XCTAssertFalse(r.isComplete())
  }

  /// Finalize frame A, immediately start frame B — the reassembler must
  /// reuse a slot and accept the new frame without leaking state from A.
  func testTwoFramesInFlightViaEviction() {
    let r = UDPFrameReassembler()
    let chunkSize = 1400

    /* Frame A: single chunk, complete and finalize. */
    XCTAssertNotNil(
      r.startFrame(frameId: 1, chunkCount: 1, expectedLen: chunkSize, isKeyframe: true))
    withChunk(chunkSize) { bytes in
      r.placeChunk(chunkIdx: 0, offset: 0, payload: bytes)
    }
    XCTAssertTrue(r.isComplete())
    let dataA = r.finalizeFrame()
    XCTAssertNotNil(dataA)
    XCTAssertEqual(dataA?.count, chunkSize)
    XCTAssertNil(r.currentFrameId)

    /* Frame B: should land in a fresh (or reused) slot, no leakage. */
    XCTAssertNotNil(
      r.startFrame(
        frameId: 2, chunkCount: 2, expectedLen: chunkSize * 2, isKeyframe: false))
    XCTAssertEqual(r.currentFrameId, 2)
    XCTAssertFalse(r.isComplete())

    withChunk(chunkSize) { bytes in
      r.placeChunk(chunkIdx: 0, offset: 0, payload: bytes)
    }
    withChunk(chunkSize) { bytes in
      r.placeChunk(chunkIdx: 1, offset: chunkSize, payload: bytes)
    }
    XCTAssertTrue(r.isComplete())

    let dataB = r.finalizeFrame()
    XCTAssertNotNil(dataB)
    XCTAssertEqual(dataB?.count, chunkSize * 2)
  }
}
