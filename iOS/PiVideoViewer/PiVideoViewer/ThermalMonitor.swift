import Foundation
import os

/// Polls `ProcessInfo.processInfo.thermalState` and publishes transitions
/// so the pipeline can shed work (pause UDP, stop requesting frames) when
/// the device is under thermal pressure. Transitions are edge-triggered,
/// so `onStateChange` fires exactly once per change.
///
/// iOS does not deliver a real-time thermal notification, so we poll on
/// a 2s cadence. SoC temperature moves slowly enough that this is fine.
@MainActor
final class ThermalMonitor: ObservableObject {
  @Published private(set) var currentState: ProcessInfo.ThermalState

  /// Called once per state transition on the main actor.
  var onStateChange: ((ProcessInfo.ThermalState) -> Void)?

  private let logger = Logger(subsystem: "com.gata.pivideoviewer",
                              category: "thermal")
  private var timer: Timer?
  private let pollInterval: TimeInterval

  init(pollInterval: TimeInterval = 2.0) {
    self.pollInterval = pollInterval
    self.currentState = ProcessInfo.processInfo.thermalState
  }

  func start() {
    stop()
    timer = Timer.scheduledTimer(withTimeInterval: pollInterval,
                                 repeats: true) { [weak self] _ in
      Task { @MainActor in self?.tick() }
    }
    // Give one immediate reading so listeners don't have to wait 2s.
    tick()
  }

  func stop() {
    timer?.invalidate()
    timer = nil
  }

  private func tick() {
    let newState = ProcessInfo.processInfo.thermalState
    guard newState != currentState else { return }
    let oldLabel = Self.label(for: currentState)
    let newLabel = Self.label(for: newState)
    logger.info("thermal state \(oldLabel) -> \(newLabel)")
    currentState = newState
    onStateChange?(newState)
  }

  static func label(for state: ProcessInfo.ThermalState) -> String {
    switch state {
    case .nominal:  return "nominal"
    case .fair:     return "fair"
    case .serious:  return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
  }
}
