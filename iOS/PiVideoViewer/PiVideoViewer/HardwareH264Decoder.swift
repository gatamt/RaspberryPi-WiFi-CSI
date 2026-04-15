import AVFoundation
import CoreMedia
import Foundation
import os

/// H.264 sample buffer producer. Despite the name, this class no longer
/// runs its own `VTDecompressionSession`. It:
///
///   1. Extracts SPS/PPS from every keyframe
///   2. Builds a `CMVideoFormatDescription` from the parameter sets
///   3. Converts the Annex-B bytestream to AVCC (length-prefixed)
///   4. Wraps the AVCC bytes in a `CMBlockBuffer` that owns its own copy
///   5. Creates a `CMSampleBuffer` and passes it to `onFrameDecoded`
///
/// `VideoViewModel` enqueues the sample buffer into an
/// `AVSampleBufferDisplayLayer`, which does the actual VideoToolbox decode
/// and composites the output via CoreAnimation. The old name stayed to
/// avoid churning every call site; the contract (feed H.264, get something
/// displayable) is the same.
final class HardwareH264Decoder: @unchecked Sendable {
  private let logger = Logger(subsystem: "com.gata.pivideoviewer",
                              category: "decoder")

  private var formatDescription: CMVideoFormatDescription?
  private var spsData: Data?
  private var ppsData: Data?

  /// Called with each ready-to-display sample buffer on an arbitrary thread.
  /// The receiver is expected to dispatch to main and call
  /// `displayLayer.enqueue(_:)`.
  var onFrameDecoded: ((_ sampleBuffer: CMSampleBuffer) -> Void)?
  var onDecodeFailure: (() -> Void)?

  private var buildFailures: UInt64 = 0

  /// Consume a complete H.264 frame (one or more NAL units) and emit a
  /// sample buffer via `onFrameDecoded` when one can be built.
  func decode(h264Data: Data, isKeyframe: Bool) {
    if isKeyframe {
      extractParameterSets(from: h264Data)
    }

    guard formatDescription != nil else {
      return  // still waiting for the first keyframe with SPS/PPS
    }

    guard let avccData = annexBToAVCC(h264Data) else { return }

    guard let sampleBuffer = makeSampleBuffer(from: avccData) else {
      buildFailures += 1
      if buildFailures % 50 == 1 {
        logger.error("sample buffer build failed (count: \(self.buildFailures))")
      }
      onDecodeFailure?()
      return
    }

    onFrameDecoded?(sampleBuffer)
  }

  func stop() {
    formatDescription = nil
    spsData = nil
    ppsData = nil
  }

  /// Kept for source compatibility with the previous VT-session-based
  /// implementation. In the new pipeline there is no per-decoder session
  /// to reset — we just drop the format description so the next keyframe
  /// rebuilds it from fresh SPS/PPS.
  func resetSession() {
    formatDescription = nil
    spsData = nil
    ppsData = nil
  }

  // MARK: - Private

  /// Build a CMSampleBuffer that owns its own copy of the AVCC payload.
  private func makeSampleBuffer(from avccData: Data) -> CMSampleBuffer? {
    guard let formatDescription else { return nil }

    // 1) Allocate a CMBlockBuffer whose backing memory is owned by
    //    kCFAllocatorDefault. The AssureMemoryNow flag forces the backing
    //    store to materialize now (not lazily).
    var blockBuffer: CMBlockBuffer?
    let blockStatus = CMBlockBufferCreateWithMemoryBlock(
      allocator: kCFAllocatorDefault,
      memoryBlock: nil,
      blockLength: avccData.count,
      blockAllocator: kCFAllocatorDefault,
      customBlockSource: nil,
      offsetToData: 0,
      dataLength: avccData.count,
      flags: kCMBlockBufferAssureMemoryNowFlag,
      blockBufferOut: &blockBuffer
    )
    guard blockStatus == kCMBlockBufferNoErr, let block = blockBuffer else {
      return nil
    }

    // 2) Copy the AVCC bytes into the block buffer so the sample buffer
    //    is safe to pass outside this function (avccData would otherwise
    //    be deallocated when this scope ends).
    let copyStatus = avccData.withUnsafeBytes { rawBuf -> OSStatus in
      guard let baseAddr = rawBuf.baseAddress else {
        return kCMBlockBufferBadPointerParameterErr
      }
      return CMBlockBufferReplaceDataBytes(
        with: baseAddr,
        blockBuffer: block,
        offsetIntoDestination: 0,
        dataLength: avccData.count
      )
    }
    guard copyStatus == kCMBlockBufferNoErr else { return nil }

    // 3) Wrap the block buffer in a CMSampleBuffer with the format
    //    description we extracted from SPS/PPS.
    var sampleBuffer: CMSampleBuffer?
    var sampleSize = avccData.count
    let sampleStatus = CMSampleBufferCreateReady(
      allocator: kCFAllocatorDefault,
      dataBuffer: block,
      formatDescription: formatDescription,
      sampleCount: 1,
      sampleTimingEntryCount: 0,
      sampleTimingArray: nil,
      sampleSizeEntryCount: 1,
      sampleSizeArray: &sampleSize,
      sampleBufferOut: &sampleBuffer
    )
    guard sampleStatus == noErr, let sample = sampleBuffer else {
      return nil
    }

    // 4) Mark the sample for immediate display. Without this key the
    //    display layer waits for the PTS to catch up to its internal
    //    clock — we don't care about PTS since the stream is live.
    if let attachments = CMSampleBufferGetSampleAttachmentsArray(
      sample, createIfNecessary: true) as? [NSMutableDictionary],
       let first = attachments.first {
      first[kCMSampleAttachmentKey_DisplayImmediately as String] = true
    }

    return sample
  }

  /// Find and extract SPS and PPS NAL units from an Annex-B bytestream.
  private func extractParameterSets(from data: Data) {
    let nalus = findNALUnits(in: data)
    for nalu in nalus {
      let naluType = nalu[0] & 0x1F
      if naluType == 7 {  // SPS
        spsData = nalu
      } else if naluType == 8 {  // PPS
        ppsData = nalu
      }
    }
    if let sps = spsData, let pps = ppsData, formatDescription == nil {
      createFormatDescription(sps: sps, pps: pps)
    }
  }

  private func createFormatDescription(sps: Data, pps: Data) {
    var desc: CMVideoFormatDescription?
    let status = sps.withUnsafeBytes { spsPtr -> OSStatus in
      pps.withUnsafeBytes { ppsPtr -> OSStatus in
        let parameterSetPointers: [UnsafePointer<UInt8>] = [
          spsPtr.baseAddress!.assumingMemoryBound(to: UInt8.self),
          ppsPtr.baseAddress!.assumingMemoryBound(to: UInt8.self),
        ]
        let parameterSetSizes: [Int] = [sps.count, pps.count]
        return CMVideoFormatDescriptionCreateFromH264ParameterSets(
          allocator: kCFAllocatorDefault,
          parameterSetCount: 2,
          parameterSetPointers: parameterSetPointers,
          parameterSetSizes: parameterSetSizes,
          nalUnitHeaderLength: 4,
          formatDescriptionOut: &desc
        )
      }
    }
    if status == noErr, let desc {
      formatDescription = desc
      logger.info("format description built from SPS/PPS")
    } else {
      logger.error("failed to create format description: \(status)")
    }
  }

  /// Find NAL unit boundaries in Annex-B data (00 00 00 01 or 00 00 01).
  private func findNALUnits(in data: Data) -> [Data] {
    var nalus: [Data] = []
    var i = 0
    let bytes = Array(data)
    var startPositions: [Int] = []

    while i < bytes.count - 3 {
      if bytes[i] == 0 && bytes[i + 1] == 0 {
        if bytes[i + 2] == 1 {
          startPositions.append(i + 3)
          i += 3
        } else if i < bytes.count - 4 && bytes[i + 2] == 0 && bytes[i + 3] == 1 {
          startPositions.append(i + 4)
          i += 4
        } else {
          i += 1
        }
      } else {
        i += 1
      }
    }

    for j in 0..<startPositions.count {
      let start = startPositions[j]
      let end =
        j + 1 < startPositions.count
        ? findStartCodeBefore(startPositions[j + 1], in: bytes)
        : bytes.count
      if start < end {
        nalus.append(Data(bytes[start..<end]))
      }
    }
    return nalus
  }

  private func findStartCodeBefore(_ pos: Int, in bytes: [UInt8]) -> Int {
    if pos >= 4 && bytes[pos - 4] == 0 && bytes[pos - 3] == 0
      && bytes[pos - 2] == 0 && bytes[pos - 1] == 1
    {
      return pos - 4
    }
    if pos >= 3 && bytes[pos - 3] == 0 && bytes[pos - 2] == 0
      && bytes[pos - 1] == 1
    {
      return pos - 3
    }
    return pos
  }

  /// Convert Annex-B (start code) format to AVCC (length-prefixed) format.
  /// SPS and PPS are stripped — they live in the format description.
  private func annexBToAVCC(_ data: Data) -> Data? {
    let nalus = findNALUnits(in: data)
    guard !nalus.isEmpty else { return nil }
    var result = Data()
    for nalu in nalus {
      let naluType = nalu[0] & 0x1F
      if naluType == 7 || naluType == 8 { continue }
      var length = UInt32(nalu.count).bigEndian
      result.append(Data(bytes: &length, count: 4))
      result.append(nalu)
    }
    return result.isEmpty ? nil : result
  }
}
