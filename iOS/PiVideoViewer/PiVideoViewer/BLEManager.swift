import CoreBluetooth
import Foundation
import os

private actor BLECommandGate {
    private var isBusy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func acquire() async {
        if !isBusy {
            isBusy = true
            return
        }

        await withCheckedContinuation { continuation in
            waiters.append(continuation)
        }
    }

    func release() {
        if let next = waiters.first {
            waiters.removeFirst()
            next.resume()
        } else {
            isBusy = false
        }
    }

    /// Unblock all queued waiters on BLE disconnect so tasks don't hang.
    /// Unblocked tasks will hit sendCommand's nil-guard and throw .notConnected.
    func cancelAll() {
        let pending = waiters
        waiters.removeAll()
        isBusy = false
        for waiter in pending {
            waiter.resume()
        }
    }
}

struct DiscoveredPi: Identifiable {
    let peripheral: CBPeripheral
    let name: String
    let rssi: Int
    var id: UUID { peripheral.identifier }
}

enum BLEConnectionState: String {
    case poweredOff
    case unauthorized
    case unsupported
    case disconnected
    case scanning
    case connecting
    case discoveringServices
    case ready
    case error
}

/// BLE central manager — NO @MainActor. Delegates run on main queue (queue: nil).
/// UI state published via @Published on main thread.
final class BLEManager: NSObject, ObservableObject {
    private let logger = Logger(subsystem: "com.gata.pivideoviewer", category: "BLE")

    // Published state (updated on main thread)
    @Published var connectionState: BLEConnectionState = .disconnected
    @Published var discoveredPis: [DiscoveredPi] = []
    @Published var wifiNetworks: [WiFiNetwork] = []
    @Published var isStreamRunning = false
    @Published var isScanning = false
    @Published var isReconnecting = false
    @Published var wifiConnected = false
    @Published var connectedSSID: String?
    @Published var piZTIP: String?
    @Published var piWlanIP: String?
    @Published var lastError: String?

    // CoreBluetooth
    private var centralManager: CBCentralManager!
    private(set) var connectedPeripheral: CBPeripheral?

    // Characteristics
    private var commandChrc: CBCharacteristic?
    private var responseChrc: CBCharacteristic?
    private var wifiNetworksChrc: CBCharacteristic?
    private var wifiStatusChrc: CBCharacteristic?
    private var streamStatusChrc: CBCharacteristic?

    // Command/response
    private var pendingContinuation: CheckedContinuation<String, Error>?
    private var pendingCommandID: UUID?
    private var pendingCommandName: String?
    private var pendingTimeoutTask: Task<Void, Never>?
    private let chunkAssembler = ChunkAssembler()
    private let wifiChunkAssembler = ChunkAssembler()
    private let commandGate = BLECommandGate()

    // Scan intent
    private var wantsScan = false

    // Disconnect tracking
    private enum DisconnectReason { case userInitiated, unexpected }
    private var disconnectReason: DisconnectReason = .unexpected

    // Auto-reconnect
    private var reconnectTask: Task<Void, Never>?
    private var reconnectAttempt = 0
    private static let maxReconnectAttempts = 5

    override init() {
        super.init()
        centralManager = CBCentralManager(
            delegate: self,
            queue: nil,  // main queue
            options: [
                CBCentralManagerOptionShowPowerAlertKey: true,
            ]
        )
    }

    // MARK: - Public API

    func startScanning() {
        wantsScan = true
        if centralManager.state == .poweredOn {
            beginScan()
        } else {
            logger.info("BLE scan requested, BT state=\(self.centralManager.state.rawValue)")
        }
    }

    func stopScanning() {
        wantsScan = false
        if isScanning {
            centralManager.stopScan()
            isScanning = false
            logger.info("BLE scanning stopped")
        }
        if connectionState == .scanning {
            connectionState = .disconnected
        }
    }

    func connect(to pi: DiscoveredPi) {
        connect(peripheral: pi.peripheral, displayName: pi.name)
    }

    func disconnect() {
        disconnectReason = .userInitiated
        reconnectTask?.cancel()
        reconnectTask = nil
        if let p = connectedPeripheral {
            centralManager.cancelPeripheralConnection(p)
        }
        resetState()
    }

    func reconnectToSaved(identifier: UUID) {
        if !connectToKnownPi(savedIdentifier: identifier) {
            logger.warning("Saved peripheral not found, starting scan")
            startScanning()
        }
    }

    @MainActor
    func sendCommand(_ command: String, timeout: TimeInterval = 10) async throws -> String {
        await commandGate.acquire()
        defer { Task { await self.commandGate.release() } }

        guard let chrc = commandChrc, let peripheral = connectedPeripheral else {
            throw BLEError.notConnected
        }
        guard let data = command.data(using: .utf8) else {
            throw BLEError.notConnected
        }

        return try await withCheckedThrowingContinuation { continuation in
            let requestID = UUID()
            self.pendingCommandID = requestID
            self.pendingCommandName = command
            self.pendingContinuation = continuation
            self.pendingTimeoutTask?.cancel()
            self.pendingTimeoutTask = Task { [weak self] in
                try? await Task.sleep(for: .seconds(timeout))
                await MainActor.run {
                    self?.timeoutPendingCommand(id: requestID)
                }
            }
            peripheral.writeValue(data, for: chrc, type: .withResponse)
        }
    }

    @MainActor
    func queryAuthStatus() async throws -> Bool {
        let response = try await sendCommand(BLECommand.authStatus())
        return BLEResponse.parse(response).message == "HAS_PIN"
    }

    @MainActor
    func queryWiFiStatus() async throws -> WiFiConnectionStatus {
        let response = try await sendCommand(BLECommand.wifiStatus())
        let payload = BLEResponse.parse(response).message
        handleWifiStatus(payload)
        return WiFiConnectionStatus.parse(payload)
    }

    @MainActor
    func queryStreamStatus() async throws -> Bool {
        let response = try await sendCommand(BLECommand.streamStatus())
        let payload = BLEResponse.parse(response).message
        let isRunning = payload.trimmingCharacters(in: .whitespacesAndNewlines) == "RUNNING"
        self.isStreamRunning = isRunning
        return isRunning
    }

    func connectToKnownPi(savedIdentifier: UUID?) -> Bool {
        let connected = centralManager.retrieveConnectedPeripherals(withServices: [PiBLE.serviceUUID])
        if let savedIdentifier {
            if let peripheral = connected.first(where: { $0.identifier == savedIdentifier }) {
                connect(peripheral: peripheral, displayName: peripheral.name ?? "Saved Pi")
                return true
            }
            let peripherals = centralManager.retrievePeripherals(withIdentifiers: [savedIdentifier])
            if let peripheral = peripherals.first {
                connect(peripheral: peripheral, displayName: peripheral.name ?? "Saved Pi")
                return true
            }
        }

        if let peripheral = connected.first {
            connect(peripheral: peripheral, displayName: peripheral.name ?? "Connected Pi")
            return true
        }
        return false
    }

    func knownScanCandidate(savedIdentifier: UUID?) -> DiscoveredPi? {
        if let savedIdentifier,
           let match = discoveredPis.first(where: { $0.peripheral.identifier == savedIdentifier }) {
            return match
        }
        return discoveredPis.first(where: { $0.name.hasPrefix(PiBLE.advertisedNamePrefix) })
    }

    // MARK: - Auto-Reconnect

    func attemptReconnect() {
        reconnectTask?.cancel()

        guard let savedPi = PiStorage.load() else {
            isReconnecting = false
            return
        }

        isReconnecting = true
        reconnectAttempt = 0
        connectionState = .disconnected

        reconnectTask = Task { [weak self] in
            guard let self else { return }

            while !Task.isCancelled && self.reconnectAttempt < Self.maxReconnectAttempts {
                self.reconnectAttempt += 1
                let delay = pow(2.0, Double(self.reconnectAttempt - 1))
                self.logger.info("BLE reconnect attempt \(self.reconnectAttempt)/\(Self.maxReconnectAttempts) in \(delay)s")

                try? await Task.sleep(for: .seconds(delay))
                if Task.isCancelled { break }

                // Try CoreBluetooth cache first
                if self.connectToKnownPi(savedIdentifier: savedPi.peripheralIdentifier) {
                    if await self.waitForBLEReady(timeout: 5.0) {
                        await self.autoAuthAfterReconnect()
                        return
                    }
                }

                // Cache miss: brief scan
                self.startScanning()
                try? await Task.sleep(for: .seconds(3))
                if Task.isCancelled { break }

                if let candidate = self.knownScanCandidate(savedIdentifier: savedPi.peripheralIdentifier) {
                    self.connect(to: candidate)
                    if await self.waitForBLEReady(timeout: 5.0) {
                        await self.autoAuthAfterReconnect()
                        return
                    }
                }
                self.stopScanning()
            }

            // All attempts exhausted
            await MainActor.run {
                self.isReconnecting = false
                self.logger.warning("BLE reconnect gave up after \(Self.maxReconnectAttempts) attempts")
            }
        }
    }

    @MainActor
    private func autoAuthAfterReconnect() async {
        guard let pin = PiStorage.loadPin() else {
            logger.warning("Reconnected but no saved PIN — skipping auth")
            isReconnecting = false
            return
        }

        do {
            let hasPIN = try await queryAuthStatus()
            if hasPIN {
                let response = try await sendCommand(BLECommand.authVerify(pin: pin))
                guard BLEResponse.parse(response).isOK else {
                    logger.error("Reconnect auth failed — PIN rejected")
                    isReconnecting = false
                    return
                }
            }
            // Re-sync stream status with Pi's actual state
            _ = try? await queryStreamStatus()
            logger.info("BLE reconnected and authenticated successfully")
            isReconnecting = false
        } catch {
            logger.error("Reconnect auth error: \(error.localizedDescription)")
            isReconnecting = false
        }
    }

    private func waitForBLEReady(timeout: TimeInterval) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if Task.isCancelled { return false }
            switch connectionState {
            case .ready:
                return true
            case .error:
                return false
            default:
                break
            }
            try? await Task.sleep(for: .milliseconds(100))
        }
        return connectionState == .ready
    }

    // MARK: - Private

    private func beginScan() {
        discoveredPis.removeAll()
        isScanning = true
        connectionState = .scanning
        centralManager.scanForPeripherals(
            withServices: [PiBLE.serviceUUID],
            options: [CBCentralManagerScanOptionAllowDuplicatesKey: false]
        )
        logger.info("BLE scanning started (service UUID filter)")
    }

    private func connect(peripheral: CBPeripheral, displayName: String) {
        stopScanning()
        connectionState = .connecting
        connectedPeripheral = peripheral
        peripheral.delegate = self
        centralManager.connect(peripheral, options: nil)
        logger.info("Connecting to \(displayName)")
    }

    private func clearPendingCommand() {
        pendingTimeoutTask?.cancel()
        pendingTimeoutTask = nil
        pendingCommandID = nil
        pendingCommandName = nil
        pendingContinuation = nil
    }

    private func timeoutPendingCommand(id: UUID) {
        guard pendingCommandID == id, let pending = pendingContinuation else { return }
        let commandName = pendingCommandName
        clearPendingCommand()
        logger.error("BLE command timed out: \(commandName ?? "unknown", privacy: .public)")
        pending.resume(throwing: BLEError.timeout(command: commandName))
    }

    private func resetState() {
        pendingTimeoutTask?.cancel()
        pendingTimeoutTask = nil
        connectedPeripheral = nil
        commandChrc = nil
        responseChrc = nil
        wifiNetworksChrc = nil
        wifiStatusChrc = nil
        streamStatusChrc = nil
        pendingContinuation = nil
        pendingCommandID = nil
        pendingCommandName = nil
        chunkAssembler.reset()
        wifiChunkAssembler.reset()
        isStreamRunning = false
        isReconnecting = false
        wifiConnected = false
        connectedSSID = nil
        piZTIP = nil
        piWlanIP = nil
        connectionState = centralManager.state == .poweredOn ? .disconnected : .poweredOff
    }

    /// Partial reset: clear BLE handles only, preserve stream/wifi state.
    /// Used for unexpected BLE drops where video pipeline should survive.
    private func resetBLEHandles() {
        pendingTimeoutTask?.cancel()
        pendingTimeoutTask = nil
        connectedPeripheral = nil
        commandChrc = nil
        responseChrc = nil
        wifiNetworksChrc = nil
        wifiStatusChrc = nil
        streamStatusChrc = nil
        pendingContinuation = nil
        pendingCommandID = nil
        pendingCommandName = nil
        chunkAssembler.reset()
        wifiChunkAssembler.reset()
        // Deliberately NOT touching: isStreamRunning, wifiConnected,
        // connectedSSID, piZTIP, piWlanIP — video pipeline stays alive
    }
}

// MARK: - CBCentralManagerDelegate

extension BLEManager: CBCentralManagerDelegate {
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        logger.info("BT state changed: \(central.state.rawValue)")

        switch central.state {
        case .poweredOn:
            if wantsScan {
                beginScan()
            } else {
                connectionState = .disconnected
            }

        case .poweredOff:
            isScanning = false
            connectionState = .poweredOff

        case .unauthorized:
            isScanning = false
            connectionState = .unauthorized
            lastError = "Bluetooth permission denied"

        case .unsupported:
            isScanning = false
            connectionState = .unsupported
            lastError = "Bluetooth not supported"

        case .resetting:
            isScanning = false
            logger.info("Bluetooth resetting, will re-scan when ready")

        case .unknown:
            logger.info("Bluetooth state unknown, waiting")

        @unknown default:
            break
        }
    }

    func centralManager(
        _ central: CBCentralManager,
        didDiscover peripheral: CBPeripheral,
        advertisementData: [String: Any],
        rssi RSSI: NSNumber
    ) {
        let name = advertisementData[CBAdvertisementDataLocalNameKey] as? String
            ?? peripheral.name
            ?? "Unknown Device"

        if !discoveredPis.contains(where: { $0.peripheral.identifier == peripheral.identifier }) {
            let pi = DiscoveredPi(peripheral: peripheral, name: name, rssi: RSSI.intValue)
            discoveredPis.append(pi)
            logger.info("Discovered: \(name) RSSI=\(RSSI)")
        }
    }

    func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        connectionState = .discoveringServices
        peripheral.discoverServices([PiBLE.serviceUUID])
        logger.info("Connected, discovering services")
    }

    func centralManager(
        _ central: CBCentralManager,
        didFailToConnect peripheral: CBPeripheral,
        error: Error?
    ) {
        connectionState = .error
        lastError = error?.localizedDescription ?? "Connection failed"
        logger.error("Failed to connect: \(error?.localizedDescription ?? "unknown")")
    }

    func centralManager(
        _ central: CBCentralManager,
        didDisconnectPeripheral peripheral: CBPeripheral,
        error: Error?
    ) {
        let reason = disconnectReason
        disconnectReason = .unexpected // reset for next time
        logger.info("Disconnected from Pi (reason: \(String(describing: reason), privacy: .public))")

        // Always resolve any in-flight command
        if let pending = pendingContinuation {
            clearPendingCommand()
            pending.resume(throwing: BLEError.disconnected)
        }
        Task { await commandGate.cancelAll() }

        if reason == .userInitiated {
            resetState()
        } else {
            resetBLEHandles()
            attemptReconnect()
        }
    }
}

// MARK: - CBPeripheralDelegate

extension BLEManager: CBPeripheralDelegate {
    func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        guard let service = peripheral.services?.first(where: { $0.uuid == PiBLE.serviceUUID }) else {
            connectionState = .error
            lastError = "Pi Setup service not found"
            return
        }
        peripheral.discoverCharacteristics(PiBLE.allCharacteristicUUIDs, for: service)
    }

    func peripheral(
        _ peripheral: CBPeripheral,
        didDiscoverCharacteristicsFor service: CBService,
        error: Error?
    ) {
        guard let chrcs = service.characteristics else { return }
        for chrc in chrcs {
            switch chrc.uuid {
            case PiBLE.commandUUID:
                commandChrc = chrc
            case PiBLE.responseUUID:
                responseChrc = chrc
                peripheral.setNotifyValue(true, for: chrc)
            case PiBLE.wifiNetworksUUID:
                wifiNetworksChrc = chrc
                peripheral.setNotifyValue(true, for: chrc)
            case PiBLE.wifiStatusUUID:
                wifiStatusChrc = chrc
                peripheral.setNotifyValue(true, for: chrc)
            case PiBLE.streamStatusUUID:
                streamStatusChrc = chrc
                peripheral.setNotifyValue(true, for: chrc)
            default:
                break
            }
        }

        if commandChrc != nil && responseChrc != nil {
            connectionState = .ready
            logger.info("All characteristics discovered, BLE ready")
        } else {
            connectionState = .error
            lastError = "Missing required characteristics"
        }
    }

    func peripheral(
        _ peripheral: CBPeripheral,
        didUpdateValueFor characteristic: CBCharacteristic,
        error: Error?
    ) {
        guard let data = characteristic.value, let string = String(data: data, encoding: .utf8) else { return }

        switch characteristic.uuid {
        case PiBLE.responseUUID:
            handleResponse(string)
        case PiBLE.wifiNetworksUUID:
            handleWifiNetworks(string)
        case PiBLE.wifiStatusUUID:
            handleWifiStatus(string)
        case PiBLE.streamStatusUUID:
            isStreamRunning = (string.trimmingCharacters(in: .whitespacesAndNewlines) == "RUNNING")
        default:
            break
        }
    }

    func peripheral(
        _ peripheral: CBPeripheral,
        didWriteValueFor characteristic: CBCharacteristic,
        error: Error?
    ) {
        if let error = error {
            if let pending = pendingContinuation {
                clearPendingCommand()
                pending.resume(throwing: error)
            }
        }
    }
}

// MARK: - Response Handling

private extension BLEManager {
    func handleResponse(_ string: String) {
        let response = BLEResponse.parse(string)

        switch response {
        case .chunk(let idx, let total, let data):
            if let assembled = chunkAssembler.addChunk(index: idx, total: total, data: data) {
                resolvePending(with: assembled)
            }
        default:
            resolvePending(with: string)
        }
    }

    func handleWifiNetworks(_ string: String) {
        let response = BLEResponse.parse(string)

        switch response {
        case .chunk(let idx, let total, let data):
            if let assembled = wifiChunkAssembler.addChunk(index: idx, total: total, data: data) {
                parseAndPublishNetworks(assembled)
            }
        default:
            parseAndPublishNetworks(string)
        }
    }

    func parseAndPublishNetworks(_ json: String) {
        guard let data = json.data(using: .utf8),
              let networks = try? JSONDecoder().decode([WiFiNetwork].self, from: data) else {
            logger.error("Failed to parse WiFi networks JSON")
            return
        }
        wifiNetworks = networks
        logger.info("Received \(networks.count) WiFi networks")
    }

    func handleWifiStatus(_ string: String) {
        let status = WiFiConnectionStatus.parse(string)
        wifiConnected = status.connected
        connectedSSID = status.ssid
        piZTIP = status.ztIP
        piWlanIP = status.wlanIP
        // Only overwrite stored IPs with non-nil values
        if let zt = status.ztIP {
            PiStorage.updateIPs(zt: zt, wlan: status.wlanIP)
        } else if let wlan = status.wlanIP {
            var saved = PiStorage.load()
            saved?.wlanIP = wlan
            if let s = saved { PiStorage.save(s) }
        }
        logger.info("WiFi status: connected=\(status.connected) ssid=\(status.ssid ?? "none") zt=\(status.ztIP ?? "none")")
    }

    func resolvePending(with response: String) {
        if let pending = pendingContinuation {
            clearPendingCommand()
            pending.resume(returning: response)
        }
    }
}

// MARK: - Errors

enum BLEError: LocalizedError {
    case notConnected
    case timeout(command: String?)
    case disconnected

    var errorDescription: String? {
        switch self {
        case .notConnected: return "Not connected to Pi"
        case .timeout(let command):
            if let command {
                return "\(command) timed out"
            }
            return "Command timed out"
        case .disconnected: return "Pi disconnected"
        }
    }
}
