import SwiftUI

enum AppScreen {
    case loading
    case start
    case bleScan
    case pinSetup
    case wifiSetup
    case main
}

@MainActor
final class AppRouter: ObservableObject {
    @Published var currentScreen: AppScreen = .loading

    let bleManager: BLEManager
    private var savedPi: SavedPi?
    private var startupTask: Task<Void, Never>?

    init(bleManager: BLEManager) {
        self.bleManager = bleManager
    }

    func determineInitialScreen() {
        guard startupTask == nil else { return }
        startupTask = Task { [weak self] in
            await self?.bootstrapKnownPi()
        }
    }

    func connectToDiscoveredPi(_ pi: DiscoveredPi) {
        startupTask?.cancel()
        currentScreen = .loading
        bleManager.connect(to: pi)
        startupTask = Task { [weak self] in
            await self?.continueAfterBLEReady(fallbackScreen: .bleScan)
        }
    }

    func handlePINEntryRequired() {
        currentScreen = .pinSetup
    }

    func handleScanCancelled() {
        startupTask?.cancel()
        startupTask = nil
        currentScreen = .start
    }

    func handleSuccessfulAuth() async {
        persistConnectedPi()

        do {
            let wifiStatus = try await bleManager.queryWiFiStatus()
            if wifiStatus.connected {
                _ = try? await bleManager.queryStreamStatus()
                currentScreen = .main
            } else {
                currentScreen = .wifiSetup
            }
        } catch {
            currentScreen = .wifiSetup
        }
    }

    private func bootstrapKnownPi() async {
        defer { startupTask = nil }

        savedPi = PiStorage.load()
        guard let savedPi else {
            currentScreen = .start
            return
        }

        currentScreen = .loading

        if !bleManager.connectToKnownPi(savedIdentifier: savedPi.peripheralIdentifier) {
            bleManager.startScanning()
            defer { bleManager.stopScanning() }

            let deadline = Date().addingTimeInterval(3.0)
            while Date() < deadline {
                if let candidate = bleManager.knownScanCandidate(savedIdentifier: savedPi.peripheralIdentifier) {
                    bleManager.connect(to: candidate)
                    break
                }
                try? await Task.sleep(for: .milliseconds(100))
                if Task.isCancelled { return }
            }
        }

        await continueAfterBLEReady(fallbackScreen: .start)
    }

    private func continueAfterBLEReady(fallbackScreen: AppScreen) async {
        guard await waitForBLEReady(timeout: 5.0) else {
            currentScreen = fallbackScreen
            return
        }

        do {
            let hasPIN = try await bleManager.queryAuthStatus()
            if hasPIN {
                guard let savedPIN = PiStorage.loadPin() else {
                    currentScreen = .pinSetup
                    return
                }

                let response = try await bleManager.sendCommand(BLECommand.authVerify(pin: savedPIN))
                guard BLEResponse.parse(response).isOK else {
                    currentScreen = .pinSetup
                    return
                }
                await handleSuccessfulAuth()
            } else {
                currentScreen = .pinSetup
            }
        } catch {
            currentScreen = fallbackScreen
        }
    }

    private func waitForBLEReady(timeout: TimeInterval) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if Task.isCancelled { return false }
            switch bleManager.connectionState {
            case .ready:
                return true
            case .error, .disconnected:
                return false
            default:
                break
            }
            try? await Task.sleep(for: .milliseconds(100))
        }
        return bleManager.connectionState == .ready
    }

    private func persistConnectedPi() {
        guard let peripheral = bleManager.connectedPeripheral else { return }
        let existing = PiStorage.load()
        let pi = SavedPi(
            peripheralIdentifier: peripheral.identifier,
            name: peripheral.name ?? existing?.name ?? "GataPi5",
            ztIP: existing?.ztIP ?? "",
            wlanIP: existing?.wlanIP
        )
        savedPi = pi
        PiStorage.save(pi)
    }
}

struct RootView: View {
    @StateObject var bleManager = BLEManager()
    @StateObject var appState = AppState()
    @StateObject var router: AppRouter

    init() {
        let ble = BLEManager()
        _bleManager = StateObject(wrappedValue: ble)
        _appState = StateObject(wrappedValue: AppState())
        _router = StateObject(wrappedValue: AppRouter(bleManager: ble))
    }

    var body: some View {
        Group {
            switch router.currentScreen {
            case .loading:
                ProgressView("Connecting...")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(Color.black)
                    .foregroundColor(.white)

            case .start:
                StartView(router: router)

            case .bleScan:
                BLEScanView(router: router, bleManager: bleManager)

            case .pinSetup:
                PINSetupView(router: router, bleManager: bleManager)

            case .wifiSetup:
                WiFiSetupView(router: router, bleManager: bleManager)

            case .main:
                ContentView()
                    .environmentObject(appState)
                    .environmentObject(bleManager)
                    .environmentObject(router)
            }
        }
        .onAppear {
            router.determineInitialScreen()
        }
    }
}
