import SwiftUI

/// Translucent debug HUD shown on top of the video.
///
/// Intentionally loud and always on by default so that the first time the app
/// runs against a live Pi we can immediately see whether packets are arriving,
/// the decoder is producing frames, the FPS is sane, and so on. Tap anywhere
/// to toggle visibility.
struct DebugOverlayView: View {
  @ObservedObject var appState: AppState

  var body: some View {
    VStack(alignment: .leading, spacing: 2) {
      labelLine("STATE", appState.connectionState.rawValue)
      if !appState.statusMessage.isEmpty {
        labelLine("INFO", appState.statusMessage)
      }
      labelLine("HOST", "\(appState.resolvedHost ?? "—"):\(appState.resolvedPort.map(String.init) ?? "—")")
      labelLine("SRC", appState.discoverySource)
      labelLine("FRAMES", "\(appState.frameCount)")
      labelLine("FPS", String(format: "%.1f", appState.measuredFPS))
      labelLine("RX", formatBytes(appState.bytesReceived))
      labelLine("IDR AGE", String(format: "%.1fs", appState.keyframeAgeSeconds))
      if let error = appState.lastError {
        Text(error)
          .font(.system(size: 11, weight: .semibold, design: .monospaced))
          .foregroundColor(.red)
          .lineLimit(3)
      }
    }
    .padding(10)
    .background(.black.opacity(0.55))
    .foregroundColor(.white)
    .cornerRadius(8)
    .font(.system(size: 11, weight: .regular, design: .monospaced))
  }

  private func labelLine(_ key: String, _ value: String) -> some View {
    HStack(spacing: 6) {
      Text(key)
        .foregroundColor(.green)
        .frame(width: 55, alignment: .leading)
      Text(value)
        .foregroundColor(.white)
    }
  }

  private func formatBytes(_ bytes: UInt64) -> String {
    let kb = Double(bytes) / 1024.0
    if kb < 1024 {
      return String(format: "%.1f KB", kb)
    }
    return String(format: "%.2f MB", kb / 1024.0)
  }
}
