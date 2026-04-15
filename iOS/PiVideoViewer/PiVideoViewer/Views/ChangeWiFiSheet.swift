import SwiftUI

struct ChangeWiFiSheet: View {
    @Environment(\.dismiss) var dismiss
    @EnvironmentObject var bleManager: BLEManager

    @State private var isScanning = false
    @State private var showPasswordAlert = false
    @State private var selectedSSID = ""
    @State private var password = ""
    @State private var isConnecting = false
    @State private var errorMessage: String?
    @State private var showSavedAction = false
    @State private var selectedSavedSSID = ""

    var body: some View {
        NavigationStack {
            ZStack {
                if isScanning && bleManager.wifiNetworks.isEmpty {
                    VStack {
                        ProgressView().tint(.blue)
                        Text("Scanning...").foregroundColor(.gray)
                    }
                } else {
                    List(bleManager.wifiNetworks) { network in
                        Button {
                            if network.saved {
                                selectedSavedSSID = network.ssid
                                showSavedAction = true
                            } else {
                                selectedSSID = network.ssid
                                password = ""
                                showPasswordAlert = true
                            }
                        } label: {
                            HStack {
                                VStack(alignment: .leading) {
                                    Text(network.ssid)
                                    if network.saved {
                                        Text("Saved")
                                            .font(.caption)
                                            .foregroundColor(.green)
                                    }
                                }
                                Spacer()
                                if network.isSecured {
                                    Image(systemName: "lock.fill")
                                        .font(.caption)
                                        .foregroundColor(.gray)
                                }
                            }
                        }
                    }
                }

                if isConnecting {
                    Color.black.opacity(0.4)
                    VStack {
                        ProgressView().tint(.white)
                        Text("Connecting...").foregroundColor(.white)
                    }
                }
            }
            .navigationTitle("Change WiFi")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
            .safeAreaInset(edge: .bottom) {
                if let error = errorMessage {
                    Text(error)
                        .foregroundColor(.red)
                        .font(.callout)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Color(.systemBackground))
                }
            }
            .alert("WiFi Password", isPresented: $showPasswordAlert) {
                SecureField("Password", text: $password)
                Button("Cancel", role: .cancel) {}
                Button("Connect") {
                    connect(ssid: selectedSSID, password: password)
                }
            } message: {
                Text("Enter password for \"\(selectedSSID)\"")
            }
            .confirmationDialog(selectedSavedSSID, isPresented: $showSavedAction) {
                Button("Connect") { reconnect(ssid: selectedSavedSSID) }
                Button("Forget", role: .destructive) { forgetNetwork(ssid: selectedSavedSSID) }
                Button("Cancel", role: .cancel) {}
            }
            .onAppear {
                Task { await scan() }
            }
        }
    }

    @MainActor
    private func scan() async {
        isScanning = true
        do {
            let _ = try await bleManager.sendCommand(BLECommand.wifiScan())
            try await Task.sleep(for: .seconds(5))
        } catch {}
        isScanning = false
    }

    private func reconnect(ssid: String) {
        selectedSSID = ssid
        isConnecting = true
        errorMessage = nil
        Task {
            do {
                let response = try await bleManager.sendCommand(
                    BLECommand.wifiReconnect(ssid: ssid), timeout: 30
                )
                handleResult(response)
            } catch {
                errorMessage = error.localizedDescription
                isConnecting = false
            }
        }
    }

    private func connect(ssid: String, password: String) {
        selectedSSID = ssid
        isConnecting = true
        errorMessage = nil
        Task {
            do {
                let response = try await bleManager.sendCommand(
                    BLECommand.wifiConnect(ssid: ssid, password: password), timeout: 30
                )
                handleResult(response)
            } catch {
                errorMessage = error.localizedDescription
                isConnecting = false
            }
        }
    }

    private func forgetNetwork(ssid: String) {
        Task {
            let _ = try? await bleManager.sendCommand(BLECommand.wifiForget(ssid: ssid))
            await scan()
        }
    }

    private func handleResult(_ response: String) {
        let parsed = BLEResponse.parse(response)
        isConnecting = false
        if parsed.isOK, parsed.message.hasPrefix("CONNECTED:") {
            let parts = parsed.message.split(separator: ":").map(String.init)
            let ztIP = parts.count > 1 && parts[1] != "NONE" ? parts[1] : ""
            let wlanIP = parts.count > 2 && parts[2] != "NONE" ? parts[2] : nil
            bleManager.wifiConnected = true
            bleManager.connectedSSID = selectedSSID
            bleManager.piZTIP = ztIP.isEmpty ? nil : ztIP
            bleManager.piWlanIP = wlanIP
            PiStorage.updateIPs(zt: ztIP, wlan: wlanIP)
            dismiss()
        } else {
            let errorCode = parsed.message.replacingOccurrences(of: "CONNECT_FAILED:", with: "")
            errorMessage = WiFiError.userMessage(for: errorCode)
            if errorCode == "WRONG_PASSWORD" {
                password = ""
                showPasswordAlert = true
            }
        }
    }
}
