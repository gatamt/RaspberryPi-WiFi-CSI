import Combine
import Foundation
import os

/// Central app state — drives UI and coordinates discovery → connection → video flow.
///
/// Simpler than the P4-idf6 version because this app does not render its own
/// detection overlays (they are baked into the video on the Pi side) and does
/// not handle audio.
@MainActor
final class AppState: ObservableObject {
  enum ConnectionState: String {
    case discovering = "Searching for Pi..."
    case probing = "Probing hotspot..."
    case resolving = "Resolving service..."
    case registering = "Sending VID0..."
    case waitingForKeyframe = "Waiting for keyframe..."
    case streaming = "Streaming"
    case paused = "Paused"
    case reconnecting = "Reconnecting..."
    case idle = "Ready"
    case error = "Error"
  }

  @Published var connectionState: ConnectionState = .discovering
  @Published var statusMessage: String = ""
  @Published var frameCount: UInt64 = 0
  @Published var bytesReceived: UInt64 = 0
  @Published var keyframeAgeSeconds: TimeInterval = 0
  @Published var measuredFPS: Double = 0
  @Published var lastError: String?

  /// Resolved Pi host and port from Bonjour or hotspot probe.
  @Published var resolvedHost: String?
  @Published var resolvedPort: UInt16?
  @Published var discoverySource: String = "(none)"

  let logger = Logger(subsystem: "com.gata.pivideoviewer", category: "app")

  func updateStatus(_ state: ConnectionState, message: String = "") {
    connectionState = state
    statusMessage = message
    logger.info("State: \(state.rawValue) \(message, privacy: .public)")
  }

  func logError(_ message: String) {
    lastError = message
    logger.error("\(message, privacy: .public)")
  }
}
