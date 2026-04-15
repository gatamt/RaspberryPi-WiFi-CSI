import CoreBluetooth
import Foundation

// MARK: - Service & Characteristic UUIDs

enum PiBLE {
    static let serviceUUID       = CBUUID(string: "A0000001-B000-C000-D000-E00000000000")
    static let commandUUID       = CBUUID(string: "A0000002-B000-C000-D000-E00000000000")
    static let responseUUID      = CBUUID(string: "A0000003-B000-C000-D000-E00000000000")
    static let wifiNetworksUUID  = CBUUID(string: "A0000004-B000-C000-D000-E00000000000")
    static let wifiStatusUUID    = CBUUID(string: "A0000005-B000-C000-D000-E00000000000")
    static let streamStatusUUID  = CBUUID(string: "A0000006-B000-C000-D000-E00000000000")

    static let allCharacteristicUUIDs: [CBUUID] = [
        commandUUID, responseUUID, wifiNetworksUUID, wifiStatusUUID, streamStatusUUID,
    ]

    static let advertisedNamePrefix = "GataPi"
}

// MARK: - Command Builders

enum BLECommand {
    static func authStatus() -> String { "AUTH:STATUS" }
    static func authSet(pin: String) -> String { "AUTH:SET:\(pin)" }
    static func authVerify(pin: String) -> String { "AUTH:VERIFY:\(pin)" }
    static func authChange(old: String, new: String) -> String { "AUTH:CHANGE:\(old):\(new)" }

    static func wifiScan() -> String { "WIFI:SCAN" }
    static func wifiConnect(ssid: String, password: String) -> String { "WIFI:CONNECT:\(ssid):\(password)" }
    static func wifiReconnect(ssid: String) -> String { "WIFI:RECONNECT:\(ssid)" }
    static func wifiStatus() -> String { "WIFI:STATUS" }
    static func wifiSaved() -> String { "WIFI:SAVED" }
    static func wifiForget(ssid: String) -> String { "WIFI:FORGET:\(ssid)" }

    static func streamStart() -> String { "STREAM:START" }
    static func streamStop() -> String { "STREAM:STOP" }
    static func streamStatus() -> String { "STREAM:STATUS" }
}

// MARK: - Response Parsing

enum BLEResponse {
    case ok(String)
    case error(String)
    case chunk(index: Int, total: Int, data: String)
    case raw(String)

    var isOK: Bool {
        if case .ok = self { return true }
        return false
    }

    var message: String {
        switch self {
        case .ok(let msg): return msg
        case .error(let msg): return msg
        case .chunk(_, _, let data): return data
        case .raw(let msg): return msg
        }
    }

    static func parse(_ string: String) -> BLEResponse {
        if string.hasPrefix("OK:") {
            return .ok(String(string.dropFirst(3)))
        } else if string.hasPrefix("ERR:") {
            return .error(String(string.dropFirst(4)))
        } else if string.hasPrefix("CHUNK:") {
            // CHUNK:1/3:data...
            let parts = string.split(separator: ":", maxSplits: 2)
            if parts.count >= 3 {
                let indexParts = parts[1].split(separator: "/")
                if indexParts.count == 2,
                   let idx = Int(indexParts[0]),
                   let total = Int(indexParts[1]) {
                    return .chunk(index: idx, total: total, data: String(parts[2]))
                }
            }
            return .raw(string)
        }
        return .raw(string)
    }
}

// MARK: - Chunk Reassembly

class ChunkAssembler {
    private var chunks: [Int: String] = [:]
    private var expectedTotal: Int = 0

    func reset() {
        chunks.removeAll()
        expectedTotal = 0
    }

    func addChunk(index: Int, total: Int, data: String) -> String? {
        expectedTotal = total
        chunks[index] = data

        if chunks.count == expectedTotal {
            let assembled = (1...expectedTotal).compactMap { chunks[$0] }.joined()
            reset()
            return assembled
        }
        return nil
    }
}

// MARK: - WiFi Network Model

struct WiFiNetwork: Codable, Identifiable {
    let ssid: String
    let signal: Int
    let security: String
    let saved: Bool

    var id: String { ssid }

    var signalBars: Int {
        switch signal {
        case (-50)...: return 4
        case (-65)...: return 3
        case (-75)...: return 2
        default: return 1
        }
    }

    var isSecured: Bool { security != "Open" && !security.isEmpty }
}

// MARK: - WiFi Error Mapping

enum WiFiError {
    static func userMessage(for code: String) -> String {
        switch code {
        case "WRONG_PASSWORD": return "Wrong password. Try again."
        case "NETWORK_NOT_FOUND": return "Network not found."
        case "TIMEOUT": return "Connection timed out. Try again."
        default: return "Connection failed."
        }
    }
}

// MARK: - WiFi Status

struct WiFiConnectionStatus {
    let connected: Bool
    let ssid: String?
    let ztIP: String?
    let wlanIP: String?

    static func parse(_ string: String) -> WiFiConnectionStatus {
        let payload = BLEResponse.parse(string).message
        if payload.hasPrefix("CONNECTED:") {
            let parts = payload.split(separator: ":", maxSplits: 3).map(String.init)
            return WiFiConnectionStatus(
                connected: true,
                ssid: parts.count > 1 ? parts[1] : nil,
                ztIP: parts.count > 2 && parts[2] != "NONE" ? parts[2] : nil,
                wlanIP: parts.count > 3 && parts[3] != "NONE" ? parts[3] : nil
            )
        }
        return WiFiConnectionStatus(connected: false, ssid: nil, ztIP: nil, wlanIP: nil)
    }
}
