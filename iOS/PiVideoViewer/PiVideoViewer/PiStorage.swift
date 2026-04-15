import Foundation
import Security

struct SavedPi: Codable {
    var peripheralIdentifier: UUID
    var name: String
    var ztIP: String
    var wlanIP: String?
}

class PiStorage {
    private static let piKey = "com.gata.pivideoviewer.saved-pi"
    private static let pinService = "com.gata.pivideoviewer.pi-pin"

    // MARK: - Pi info (UserDefaults)

    static func save(_ pi: SavedPi) {
        if let data = try? JSONEncoder().encode(pi) {
            UserDefaults.standard.set(data, forKey: piKey)
        }
    }

    static func load() -> SavedPi? {
        guard let data = UserDefaults.standard.data(forKey: piKey) else { return nil }
        return try? JSONDecoder().decode(SavedPi.self, from: data)
    }

    static func updateIPs(zt: String, wlan: String?) {
        guard var pi = load() else { return }
        pi.ztIP = zt
        pi.wlanIP = wlan
        save(pi)
    }

    static func clear() {
        UserDefaults.standard.removeObject(forKey: piKey)
        deletePin()
    }

    // MARK: - PIN (Keychain)

    static func savePin(_ pin: String) {
        deletePin()
        let data = pin.data(using: .utf8)!
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: pinService,
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlocked,
        ]
        SecItemAdd(query as CFDictionary, nil)
    }

    static func loadPin() -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: pinService,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        guard status == errSecSuccess, let data = result as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    static func deletePin() {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: pinService,
        ]
        SecItemDelete(query as CFDictionary)
    }
}
