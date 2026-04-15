import SwiftUI

struct SettingsMenuView: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var bleManager: BLEManager
    @EnvironmentObject var router: AppRouter
    @EnvironmentObject var viewModel: VideoViewModel

    @State private var showChangePassword = false
    @State private var showChangeWiFi = false
    @State private var streamErrorMessage: String?

    var body: some View {
        Menu {
            Button {
                showChangePassword = true
            } label: {
                Label("Change Password", systemImage: "key.fill")
            }

            Button {
                showChangeWiFi = true
            } label: {
                Label("Change WiFi", systemImage: "wifi")
            }

            Button {
                toggleStream()
            } label: {
                Label(
                    bleManager.isStreamRunning ? "Stop Videostream" : "Start Videostream",
                    systemImage: bleManager.isStreamRunning ? "stop.circle" : "play.circle"
                )
            }
        } label: {
            Image(systemName: "gearshape.fill")
                .font(.title2)
                .foregroundColor(.white)
                .padding(12)
                .background(Color(.systemGray5).opacity(0.5))
                .clipShape(Circle())
        }
        .sheet(isPresented: $showChangePassword) {
            ChangePasswordSheet()
                .environmentObject(bleManager)
        }
        .sheet(isPresented: $showChangeWiFi) {
            ChangeWiFiSheet()
                .environmentObject(bleManager)
        }
        .alert("Stream Error", isPresented: Binding(
            get: { streamErrorMessage != nil },
            set: { if !$0 { streamErrorMessage = nil } }
        )) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(streamErrorMessage ?? "Unknown stream error")
        }
    }

    private func toggleStream() {
        let wasRunning = bleManager.isStreamRunning

        Task {
            do {
                if wasRunning {
                    let response = try await bleManager.sendCommand(BLECommand.streamStop())
                    let parsed = BLEResponse.parse(response)
                    guard parsed.isOK else {
                        appState.logError(parsed.message)
                        streamErrorMessage = parsed.message
                        return
                    }
                    appState.updateStatus(.idle)
                } else {
                    appState.updateStatus(.idle, message: "Starting streamer on Pi...")
                    let response = try await bleManager.sendCommand(BLECommand.streamStart(), timeout: 15)
                    let parsed = BLEResponse.parse(response)
                    guard parsed.isOK else {
                        appState.logError(parsed.message)
                        appState.updateStatus(.idle)
                        streamErrorMessage = parsed.message
                        return
                    }
                    viewModel.streamWasFreshStart = parsed.message == "STARTED"
                }
                // Fallback: if Pi's proactive push hasn't flipped state within 2s, query explicitly
                let expectedRunning = !wasRunning
                Task {
                    try await Task.sleep(for: .seconds(2))
                    if bleManager.isStreamRunning != expectedRunning {
                        let running = try? await bleManager.queryStreamStatus()
                        if running != expectedRunning {
                            let message = expectedRunning
                                ? "Pi did not report a running videostream."
                                : "Pi did not confirm that the videostream stopped."
                            await MainActor.run {
                                appState.logError(message)
                                appState.updateStatus(.idle)
                                streamErrorMessage = message
                            }
                        }
                    }
                }
            } catch {
                appState.logError(error.localizedDescription)
                appState.updateStatus(.idle)
                streamErrorMessage = error.localizedDescription
            }
        }
    }
}
