import SwiftUI

struct WiFiSetupView: View {
    @ObservedObject var router: AppRouter
    @ObservedObject var bleManager: BLEManager

    @State private var isScanning = false
    @State private var showPasswordAlert = false
    @State private var selectedSSID = ""
    @State private var password = ""
    @State private var isConnecting = false
    @State private var errorMessage: String?
    @State private var showSavedAction = false
    @State private var selectedSavedSSID = ""

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack(spacing: 0) {
                HStack {
                    Button {
                        router.currentScreen = .pinSetup
                    } label: {
                        Image(systemName: "chevron.left")
                            .font(.title3)
                            .foregroundColor(.blue)
                    }
                    Spacer()
                    Text("Connect Your Pi to WiFi")
                        .font(.headline)
                        .foregroundColor(.white)
                    Spacer()
                    Image(systemName: "chevron.left").opacity(0)
                }
                .padding()

                if isScanning && bleManager.wifiNetworks.isEmpty {
                    Spacer()
                    VStack(spacing: 16) {
                        ProgressView().tint(.blue).scaleEffect(1.5)
                        Text("Scanning for WiFi networks...")
                            .foregroundColor(.gray)
                    }
                    Spacer()
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
                                        .foregroundColor(.white)
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
                                signalBarsView(bars: network.signalBars)
                            }
                            .padding(.vertical, 4)
                        }
                        .listRowBackground(Color(.systemGray6).opacity(0.15))
                    }
                    .listStyle(.plain)
                    .scrollContentBackground(.hidden)
                    .refreshable { await scanNetworks() }
                }

                if let error = errorMessage {
                    Text(error)
                        .foregroundColor(.red)
                        .font(.callout)
                        .padding()
                }
            }

            if isConnecting {
                Color.black.opacity(0.6).ignoresSafeArea()
                VStack(spacing: 16) {
                    ProgressView().tint(.white).scaleEffect(1.5)
                    Text("Connecting to \(selectedSSID)...")
                        .foregroundColor(.white)
                }
            }
        }
        .alert("WiFi Password", isPresented: $showPasswordAlert) {
            SecureField("Password", text: $password)
            Button("Cancel", role: .cancel) {}
            Button("Connect") {
                connectWithPassword(ssid: selectedSSID, password: password)
            }
        } message: {
            Text("Enter password for \"\(selectedSSID)\"")
        }
        .confirmationDialog(selectedSavedSSID, isPresented: $showSavedAction) {
            Button("Connect") { connectToSaved(ssid: selectedSavedSSID) }
            Button("Forget", role: .destructive) { forgetNetwork(ssid: selectedSavedSSID) }
            Button("Cancel", role: .cancel) {}
        }
        .onAppear {
            Task { await scanNetworks() }
        }
    }

    @MainActor
    private func scanNetworks() async {
        isScanning = true
        errorMessage = nil
        do {
            let _ = try await bleManager.sendCommand(BLECommand.wifiScan())
            try await Task.sleep(for: .seconds(5))
        } catch {
            errorMessage = error.localizedDescription
        }
        isScanning = false
    }

    private func connectToSaved(ssid: String) {
        selectedSSID = ssid
        isConnecting = true
        errorMessage = nil
        Task {
            do {
                let response = try await bleManager.sendCommand(
                    BLECommand.wifiReconnect(ssid: ssid), timeout: 30
                )
                handleConnectResponse(response)
            } catch {
                errorMessage = error.localizedDescription
                isConnecting = false
            }
        }
    }

    private func connectWithPassword(ssid: String, password: String) {
        selectedSSID = ssid
        isConnecting = true
        errorMessage = nil
        Task {
            do {
                let response = try await bleManager.sendCommand(
                    BLECommand.wifiConnect(ssid: ssid, password: password), timeout: 30
                )
                handleConnectResponse(response)
            } catch {
                errorMessage = error.localizedDescription
                isConnecting = false
            }
        }
    }

    private func forgetNetwork(ssid: String) {
        Task {
            let _ = try? await bleManager.sendCommand(BLECommand.wifiForget(ssid: ssid))
            await scanNetworks()
        }
    }

    private func handleConnectResponse(_ response: String) {
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
            router.currentScreen = .main
        } else {
            // Map error codes to user-friendly messages
            let errorCode = parsed.message.replacingOccurrences(of: "CONNECT_FAILED:", with: "")
            errorMessage = WiFiError.userMessage(for: errorCode)
            // On wrong password, re-show password popup
            if errorCode == "WRONG_PASSWORD" {
                password = ""
                showPasswordAlert = true
            }
        }
    }

    private func signalBarsView(bars: Int) -> some View {
        HStack(spacing: 2) {
            ForEach(1...4, id: \.self) { bar in
                RoundedRectangle(cornerRadius: 1)
                    .fill(bar <= bars ? Color.blue : Color.gray.opacity(0.3))
                    .frame(width: 4, height: CGFloat(bar * 4 + 4))
            }
        }
    }
}
