import Foundation
import Network
import os
import Darwin

/// Discovers _wificsi._udp services on the local network via Bonjour/mDNS.
/// No hardcoded IP — discovery is the sole source of truth per plan requirement.
final class BonjourDiscoveryManager: @unchecked Sendable {
  private let logger = Logger(subsystem: "com.gata.pivideoviewer", category: "discovery")
  private var browser: NWBrowser?
  private var connection: NWConnection?

  /// Called on main thread when a service is resolved.
  var onServiceResolved: ((_ host: String, _ port: UInt16) -> Void)?
  /// Called on main thread when service disappears.
  var onServiceLost: (() -> Void)?

  func startBrowsing() {
    let params = NWParameters()
    params.includePeerToPeer = true

    let descriptor = NWBrowser.Descriptor.bonjour(type: "_wificsi._udp", domain: nil)
    let browser = NWBrowser(for: descriptor, using: params)
    self.browser = browser

    browser.stateUpdateHandler = { [weak self] state in
      guard let self else { return }
      switch state {
      case .ready:
        self.logger.info("Bonjour browser ready")
      case .failed(let error):
        self.logger.error("Browser failed: \(error.localizedDescription)")
      case .cancelled:
        self.logger.info("Browser cancelled")
      default:
        break
      }
    }

    browser.browseResultsChangedHandler = { [weak self] results, changes in
      guard let self else { return }
      for change in changes {
        switch change {
        case .added(let result):
          self.logger.info("Service found: \(result.endpoint.debugDescription)")
          self.resolveEndpoint(result.endpoint)
        case .removed:
          self.logger.info("Service lost")
          DispatchQueue.main.async {
            self.onServiceLost?()
          }
        default:
          break
        }
      }
    }

    browser.start(queue: .global(qos: .userInitiated))
    logger.info("Started browsing for _wificsi._udp")
  }

  func stopBrowsing() {
    browser?.cancel()
    browser = nil
    connection?.cancel()
    connection = nil
    logger.info("Stopped browsing")
  }

  private func resolveEndpoint(_ endpoint: NWEndpoint) {
    let conn = NWConnection(to: endpoint, using: .udp)
    self.connection = conn

    conn.stateUpdateHandler = { [weak self] state in
      guard let self else { return }
      switch state {
      case .ready:
        if let path = conn.currentPath,
          let remoteEndpoint = path.remoteEndpoint
        {
          self.extractHostPort(from: remoteEndpoint)
        }
      case .failed(let error):
        self.logger.error("Resolve failed: \(error.localizedDescription)")
      default:
        break
      }
    }

    conn.start(queue: .global(qos: .userInitiated))
  }

  private func extractHostPort(from endpoint: NWEndpoint) {
    switch endpoint {
    case .hostPort(let host, let port):
      let hostStr: String
      switch host {
      case .ipv4(let addr):
        hostStr = "\(addr)"
      case .ipv6(let addr):
        hostStr = "\(addr)"
      case .name(let name, _):
        hostStr = name
      @unknown default:
        hostStr = "\(host)"
      }
      let portVal = port.rawValue
      logger.info("Resolved: \(hostStr):\(portVal)")
      DispatchQueue.main.async {
        self.onServiceResolved?(hostStr, portVal)
      }
    default:
      logger.warning("Unexpected endpoint type: \(endpoint.debugDescription)")
    }
  }
}

/// Fallback discovery for hotspot mode.
/// Sends repeated VID0 probes across likely hotspot client IPs and waits for any UDP reply.
final class HotspotProbeDiscoveryManager: @unchecked Sendable {
  private let logger = Logger(subsystem: "com.gata.pivideoviewer", category: "probe")
  private let queue = DispatchQueue(label: "com.gata.pivideoviewer.probe", qos: .userInitiated)

  private var socketFD: Int32 = -1
  private var probeTimer: DispatchSourceTimer?
  private var isRunning = false
  private var probeRound: UInt32 = 0
  private var targetPort: UInt16 = 0
  private var candidateHosts = [String]()

  var onHostDetected: ((_ host: String, _ port: UInt16) -> Void)?
  var onProbeUpdate: ((_ message: String) -> Void)?

  func startProbing(port: UInt16) {
    guard !isRunning else { return }
    isRunning = true
    targetPort = port
    candidateHosts = buildCandidateHosts()

    guard openSocket() else {
      onProbeUpdate?("Probe socket failed")
      return
    }

    logger.info("Hotspot probe starting with \(self.candidateHosts.count) candidates")
    onProbeUpdate?("Trying hotspot fallback")

    startReadLoop()
    startProbeTimer()
  }

  func stopProbing() {
    isRunning = false
    probeTimer?.cancel()
    probeTimer = nil

    if socketFD >= 0 {
      close(socketFD)
      socketFD = -1
    }
  }

  private func openSocket() -> Bool {
    socketFD = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)
    guard socketFD >= 0 else {
      logger.error("Failed to create probe socket: \(errno)")
      return false
    }

    var reuse: Int32 = 1
    setsockopt(
      socketFD, SOL_SOCKET, SO_REUSEADDR, &reuse,
      socklen_t(MemoryLayout<Int32>.size))

    let flags = fcntl(socketFD, F_GETFL, 0)
    _ = fcntl(socketFD, F_SETFL, flags | O_NONBLOCK)

    var bindAddr = sockaddr_in()
    bindAddr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    bindAddr.sin_family = sa_family_t(AF_INET)
    bindAddr.sin_port = 0
    bindAddr.sin_addr.s_addr = INADDR_ANY.bigEndian

    let bindResult = withUnsafePointer(to: &bindAddr) {
      $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
        Darwin.bind(socketFD, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
      }
    }

    if bindResult < 0 {
      logger.error("Failed to bind probe socket: \(errno)")
      close(socketFD)
      socketFD = -1
      return false
    }

    return true
  }

  private func startProbeTimer() {
    let timer = DispatchSource.makeTimerSource(queue: queue)
    timer.schedule(deadline: .now(), repeating: 1.0)
    timer.setEventHandler { [weak self] in
      guard let self, self.isRunning else { return }

      self.probeRound += 1
      if self.probeRound == 1 || self.probeRound % 5 == 0 {
        self.logger.info("Probe round \(self.probeRound) across \(self.candidateHosts.count) hosts")
      }

      for host in self.candidateHosts {
        self.sendVID0(to: host)
      }
    }
    probeTimer = timer
    timer.resume()
  }

  private func startReadLoop() {
    queue.async { [weak self] in
      guard let self, self.isRunning, self.socketFD >= 0 else { return }

      var buffer = [UInt8](repeating: 0, count: 2048)
      var srcAddr = sockaddr_in()
      var addrLen = socklen_t(MemoryLayout<sockaddr_in>.size)

      let readLen = withUnsafeMutablePointer(to: &srcAddr) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
          recvfrom(self.socketFD, &buffer, buffer.count, 0, $0, &addrLen)
        }
      }

      if readLen > 0 {
        var ipBuf = [CChar](repeating: 0, count: Int(INET_ADDRSTRLEN))
        inet_ntop(AF_INET, &srcAddr.sin_addr, &ipBuf, socklen_t(INET_ADDRSTRLEN))
        let host = String(cString: ipBuf)

        self.logger.info("Probe received UDP payload from \(host)")
        DispatchQueue.main.async {
          self.onHostDetected?(host, self.targetPort)
        }
        self.stopProbing()
        return
      }

      if errno != EAGAIN && errno != EWOULDBLOCK {
        self.logger.error("Probe recvfrom failed: \(errno)")
      }

      usleep(5_000)
      self.startReadLoop()
    }
  }

  private func sendVID0(to host: String) {
    guard socketFD >= 0 else { return }

    var dstAddr = sockaddr_in()
    dstAddr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    dstAddr.sin_family = sa_family_t(AF_INET)
    dstAddr.sin_port = targetPort.bigEndian

    let ipParseResult = host.withCString { inet_pton(AF_INET, $0, &dstAddr.sin_addr) }
    guard ipParseResult == 1 else {
      logger.warning("Skipping invalid probe host \(host)")
      return
    }

    let payload: [UInt8] = [0x56, 0x49, 0x44, 0x30]
    payload.withUnsafeBytes { bytes in
      withUnsafePointer(to: &dstAddr) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPtr in
          _ = sendto(
            socketFD,
            bytes.baseAddress,
            bytes.count,
            0,
            sockaddrPtr,
            socklen_t(MemoryLayout<sockaddr_in>.size))
        }
      }
    }
  }

  private func buildCandidateHosts() -> [String] {
    var orderedHosts = [String]()
    var seen = Set<String>()

    func appendRange(prefix: String, range: ClosedRange<Int>) {
      for host in range {
        let candidate = "\(prefix).\(host)"
        if seen.insert(candidate).inserted {
          orderedHosts.append(candidate)
        }
      }
    }

    for prefix in detectPrivateIPv4Prefixes() {
      appendRange(prefix: prefix, range: 2...20)
    }

    appendRange(prefix: "172.20.10", range: 2...20)
    return orderedHosts
  }

  private func detectPrivateIPv4Prefixes() -> [String] {
    var ifaddr: UnsafeMutablePointer<ifaddrs>?
    guard getifaddrs(&ifaddr) == 0, let first = ifaddr else {
      return []
    }
    defer { freeifaddrs(ifaddr) }

    var prefixes = [String]()
    var seen = Set<String>()
    var ptr: UnsafeMutablePointer<ifaddrs>? = first

    while let current = ptr {
      let interface = current.pointee
      guard let addr = interface.ifa_addr, addr.pointee.sa_family == UInt8(AF_INET) else {
        ptr = interface.ifa_next
        continue
      }

      var addrCopy = addr.pointee
      var hostBuf = [CChar](repeating: 0, count: Int(NI_MAXHOST))
      let result = getnameinfo(
        &addrCopy, socklen_t(addr.pointee.sa_len), &hostBuf, socklen_t(hostBuf.count),
        nil, 0, NI_NUMERICHOST)
      if result == 0 {
        let ip = String(cString: hostBuf)
        let octets = ip.split(separator: ".")
        if octets.count == 4, isPrivateIPv4(octets), octets[0] != "127" {
          let prefix = octets[0...2].joined(separator: ".")
          if seen.insert(prefix).inserted {
            prefixes.append(prefix)
          }
        }
      }

      ptr = interface.ifa_next
    }

    return prefixes
  }

  private func isPrivateIPv4(_ octets: [Substring]) -> Bool {
    guard octets.count == 4,
      let a = Int(octets[0]),
      let b = Int(octets[1])
    else {
      return false
    }

    if a == 10 || a == 192 && b == 168 {
      return true
    }

    return a == 172 && (16...31).contains(b)
  }
}
