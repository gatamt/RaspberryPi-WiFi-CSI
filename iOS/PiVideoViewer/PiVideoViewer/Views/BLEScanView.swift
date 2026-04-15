import SwiftUI

struct BLEScanView: View {
    @ObservedObject var router: AppRouter
    @ObservedObject var bleManager: BLEManager

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack(spacing: 0) {
                // Header
                HStack {
                    Button {
                        bleManager.stopScanning()
                        router.handleScanCancelled()
                    } label: {
                        Image(systemName: "chevron.left")
                            .font(.title3)
                            .foregroundColor(.blue)
                    }
                    Spacer()
                    Text("Select Your Raspberry Pi")
                        .font(.headline)
                        .foregroundColor(.white)
                    Spacer()
                    Image(systemName: "chevron.left")
                        .font(.title3)
                        .opacity(0)
                }
                .padding()

                // State-aware content
                switch bleManager.connectionState {
                case .poweredOff:
                    errorState(
                        icon: "bluetooth.slash",
                        title: "Bluetooth is turned off",
                        detail: "Enable Bluetooth in Settings to continue."
                    )

                case .unauthorized:
                    errorState(
                        icon: "lock.shield",
                        title: "Bluetooth permission denied",
                        detail: "Go to Settings > PiVideoViewer and enable Bluetooth."
                    )

                case .unsupported:
                    errorState(
                        icon: "exclamationmark.triangle",
                        title: "Bluetooth not supported",
                        detail: "This device does not support Bluetooth LE."
                    )

                default:
                    if bleManager.discoveredPis.isEmpty {
                        Spacer()
                        VStack(spacing: 16) {
                            if bleManager.isScanning {
                                ProgressView()
                                    .progressViewStyle(.circular)
                                    .tint(.blue)
                                    .scaleEffect(1.5)
                                Text("Searching for devices...")
                                    .foregroundColor(.gray)
                            } else {
                                Image(systemName: "antenna.radiowaves.left.and.right.slash")
                                    .font(.system(size: 48))
                                    .foregroundColor(.gray)
                                Text("Not scanning")
                                    .foregroundColor(.gray)
                                Button("Retry") {
                                    bleManager.startScanning()
                                }
                                .foregroundColor(.blue)
                            }
                        }
                        Spacer()
                    } else {
                        List(bleManager.discoveredPis) { pi in
                            Button {
                                router.connectToDiscoveredPi(pi)
                            } label: {
                                HStack {
                                    Image(systemName: "cpu")
                                        .foregroundColor(.green)
                                    VStack(alignment: .leading) {
                                        Text(pi.name)
                                            .foregroundColor(.white)
                                        Text("Signal: \(pi.rssi) dBm")
                                            .font(.caption)
                                            .foregroundColor(.gray)
                                    }
                                    Spacer()
                                    Image(systemName: signalIcon(rssi: pi.rssi))
                                        .foregroundColor(.blue)
                                }
                                .padding(.vertical, 4)
                            }
                            .listRowBackground(Color(.systemGray6).opacity(0.15))
                        }
                        .listStyle(.plain)
                        .scrollContentBackground(.hidden)
                    }
                }
            }
        }
        .onAppear {
            bleManager.startScanning()
        }
        .onDisappear {
            bleManager.stopScanning()
        }
    }

    private func errorState(icon: String, title: String, detail: String) -> some View {
        VStack {
            Spacer()
            VStack(spacing: 16) {
                Image(systemName: icon)
                    .font(.system(size: 48))
                    .foregroundColor(.red)
                Text(title)
                    .foregroundColor(.white)
                Text(detail)
                    .foregroundColor(.gray)
                    .font(.callout)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 40)
            }
            Spacer()
        }
    }

    private func signalIcon(rssi: Int) -> String {
        switch rssi {
        case (-50)...: return "wifi"
        case (-65)...: return "wifi"
        case (-75)...: return "wifi.exclamationmark"
        default: return "wifi.slash"
        }
    }
}
