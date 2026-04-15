import AVFoundation
import SwiftUI
import os

/// Fullscreen video view for the baked-in pose stream from the Raspberry Pi.
/// All bounding boxes, pose keypoints and labels are rendered on the Pi before
/// encoding, so the iOS side simply decodes and displays — no overlay layer.
struct ContentView: View {
  @EnvironmentObject private var appState: AppState
  @EnvironmentObject private var bleManager: BLEManager
  @EnvironmentObject private var router: AppRouter
  @StateObject private var viewModel = VideoViewModel()
  @Environment(\.scenePhase) private var scenePhase
  @State private var showDebug: Bool = true

  var body: some View {
    ZStack {
      Color.black.ignoresSafeArea()

      if viewModel.hasFirstFrame {
        VideoDisplayView(displayLayer: viewModel.displayLayer)
          .ignoresSafeArea()
      } else if bleManager.isStreamRunning {
        VStack(spacing: 16) {
          ProgressView()
            .tint(.white)
            .scaleEffect(1.5)
          Text(appState.connectionState.rawValue)
            .foregroundColor(.white)
          Text(appState.statusMessage.isEmpty ? "Waiting for the first video frame." : appState.statusMessage)
            .font(.callout)
            .foregroundColor(.gray)
            .multilineTextAlignment(.center)
            .padding(.horizontal, 40)
        }
      } else if !appState.statusMessage.isEmpty {
        VStack(spacing: 16) {
          ProgressView()
            .tint(.white)
            .scaleEffect(1.5)
          Text(appState.statusMessage)
            .foregroundColor(.white)
            .multilineTextAlignment(.center)
            .padding(.horizontal, 40)
        }
      } else {
        VStack(spacing: 16) {
          Image(systemName: "video.slash")
            .font(.system(size: 48))
            .foregroundColor(.gray)
          Text("Stream not running")
            .foregroundColor(.gray)
          Text("Tap the gear icon to start the video stream.")
            .font(.callout)
            .foregroundColor(.gray.opacity(0.7))
            .multilineTextAlignment(.center)
            .padding(.horizontal, 40)
        }
      }

      // Settings gear (top-right)
      VStack {
        HStack {
          if showDebug {
            DebugOverlayView(appState: appState)
              .padding(.horizontal, 12)
              .padding(.top, 12)
          }
          Spacer()
          SettingsMenuView()
            .environmentObject(viewModel)
            .padding(.trailing, 12)
            .padding(.top, 12)
        }
        Spacer()

        // BLE reconnect indicator (bottom-left, non-blocking)
        if bleManager.isReconnecting {
          HStack {
            HStack(spacing: 8) {
              ProgressView()
                .tint(.orange)
                .scaleEffect(0.8)
              Text("Reconnecting BLE...")
                .font(.caption)
                .foregroundColor(.orange)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(.black.opacity(0.7))
            .cornerRadius(8)
            .padding(.leading, 12)
            .padding(.bottom, 12)
            Spacer()
          }
        }
      }
    }
    .statusBarHidden(true)
    .contentShape(Rectangle())
    .onTapGesture(count: 2) {
      showDebug.toggle()
    }
    .onAppear {
      Task {
        let running = (try? await bleManager.queryStreamStatus()) ?? bleManager.isStreamRunning
        if running {
          appState.updateStatus(.registering, message: "Connecting to stream...")
          viewModel.attachToStream(appState: appState)
        } else {
          appState.updateStatus(.idle)
        }
      }
    }
    .onDisappear {
      viewModel.stop()
    }
    .onChange(of: scenePhase) { newPhase in
      switch newPhase {
      case .active:
        viewModel.handleForeground()
        // If BLE dropped while backgrounded and reconnect task was suspended, retry
        if bleManager.connectionState == .disconnected && !bleManager.isReconnecting {
          bleManager.attemptReconnect()
        }
      case .background:
        viewModel.handleBackground()
      case .inactive:
        break
      @unknown default:
        break
      }
    }
    .onChange(of: bleManager.isStreamRunning) { running in
      if running {
        appState.updateStatus(.registering, message: "Connecting to stream...")
        let delay: Duration = viewModel.streamWasFreshStart ? .seconds(3) : .milliseconds(500)
        Task {
          try? await Task.sleep(for: delay)
          viewModel.attachToStream(appState: appState)
          viewModel.streamWasFreshStart = false
        }
      } else {
        viewModel.stop()
        appState.updateStatus(.idle)
      }
    }
  }
}

// MARK: - VideoViewModel

/// Coordinates the video pipeline: discovery → UDP → decode → display.
@MainActor
final class VideoViewModel: ObservableObject {
  /// Flips to true the first time the decoder emits a sample buffer that
  /// made it into the display layer. Used by ContentView to swap from
  /// the "Waiting for first video frame" overlay to the actual video.
  /// Reset to false on stop() / disconnect / decoder reset.
  @Published var hasFirstFrame: Bool = false

  /// Layer owned by the view model so the decoder callback can enqueue
  /// sample buffers directly. VideoDisplayView hosts it as a SwiftUI
  /// subview; thermal monitor + watchdogs can also call flush / stop
  /// on it without a detour through the view hierarchy.
  let displayLayer = AVSampleBufferDisplayLayer()

  /// Gate for the decoder callback. Set true by every path that opens a
  /// fresh pipeline (start, connectDirectCandidates, etc) and false by
  /// `resetDisplay()`. Prevents a race where the decoder's MainActor
  /// Task enqueues a sample buffer onto a display layer that was just
  /// flushed by a concurrent reset.
  private var acceptFrames: Bool = false

  /// True when the UDP client was paused by ThermalMonitor. Lets the
  /// thermal recovery path (`.nominal` / `.fair`) know it owns the
  /// resume; otherwise a user- or watchdog-initiated pause would be
  /// silently overridden.
  private var thermalPausedUDP: Bool = false

  var streamWasFreshStart = false

  private let logger = Logger(subsystem: "com.gata.pivideoviewer", category: "viewmodel")

  private let discovery = BonjourDiscoveryManager()
  private let hotspotProbe = HotspotProbeDiscoveryManager()
  private let udpClient = UDPVideoClient()
  private let decoder = HardwareH264Decoder()
  private let thermalMonitor = ThermalMonitor()

  private weak var appState: AppState?
  private var isRunning = false
  private var didResolveEndpoint = false
  private var didRegisterCurrentEndpoint = false
  private var endpointWatchdogTask: Task<Void, Never>?
  private var directCandidates: [StreamEndpoint] = []
  private var activeDirectCandidateIndex = 0
  private var fallbackToDiscovery = false

  // FPS measurement
  private var fpsWindowStart = Date()
  private var fpsWindowFrames = 0

  // Keyframe age tracking
  private var lastKeyframeAt = Date()
  private var keyframeTimer: Timer?

  // MARK: - Frame watchdog (Tier 2+3 recovery)

  /// Wall-clock timestamp of the last decoded frame that actually reached
  /// the UI. Updated inside `decoder.onFrameDecoded`. This is the
  /// load-bearing "pipeline healthy" signal — we are only considered
  /// healthy when the decoder emits frames, not merely when UDP packets
  /// arrive.
  private var lastFrameAt = Date()

  /// Soft recovery threshold — rekey + decoder reset.
  /// Longer than UDPVideoClient's 2.0s packet watchdog so Tier 1 gets
  /// first crack at recovery.
  private let frameStallSoftSeconds: TimeInterval = 3.0

  /// Hard recovery threshold — full disconnect + reconnect. Chosen to
  /// fire before the Pi's 10s IDLE timeout so we can recover proactively
  /// without losing Pi-side state.
  private let frameStallHardSeconds: TimeInterval = 8.0

  /// Cooldowns prevent the watchdog from re-firing on every 0.5s tick
  /// while a recovery is in flight.
  private var lastSoftRecoveryAt = Date.distantPast
  private var lastHardRecoveryAt = Date.distantPast

  private struct StreamEndpoint {
    let host: String
    let port: UInt16
    let source: String
  }

  func start(appState: AppState) {
    if isRunning {
      stop()
    }
    isRunning = true
    self.appState = appState
    didResolveEndpoint = false
    didRegisterCurrentEndpoint = false
    configureCallbacks()
    resetFrameWatchdogClocks()
    resetDisplay()
    acceptFrames = true

    // ── Start listeners ──────────────────────────────────────────────────
    logger.info("starting discovery + hotspot probe")
    appState.updateStatus(.discovering)
    discovery.startBrowsing()
    hotspotProbe.startProbing(port: 3334)

    startKeyframeTicker()
    startThermalMonitoring()
  }

  /// Reset the display layer + first-frame flag. Call this whenever we
  /// disconnect, force a reconnect, or want the UI to return to the
  /// "Waiting for first video frame" state.
  ///
  /// Also:
  ///   1. Closes the `acceptFrames` gate so an in-flight decoder Task
  ///      that was scheduled BEFORE this reset will drop its payload
  ///      instead of enqueuing onto the freshly-flushed layer.
  ///   2. Resets the frame watchdog clocks so the next candidate gets a
  ///      full grace period before Tier 2/3 recovery fires. Without
  ///      this, `handleDirectCandidateTimeout` + `onServiceLost` paths
  ///      can trigger a false watchdog on the NEXT candidate because
  ///      `lastFrameAt` still reflects the start of the previous one.
  private func resetDisplay() {
    acceptFrames = false
    displayLayer.flush()
    hasFirstFrame = false
    resetFrameWatchdogClocks()
  }

  /// Wire up thermal monitoring so the pipeline can react to sustained
  /// thermal pressure (e.g. pause video under `.critical`). The callback
  /// runs on the main actor.
  private func startThermalMonitoring() {
    thermalMonitor.onStateChange = { [weak self] state in
      guard let self else { return }
      let label = ThermalMonitor.label(for: state)
      self.logger.warning("thermal transition: \(label, privacy: .public)")
      switch state {
      case .critical:
        // Hard shed: pause UDP so the radio gets to rest. Track that
        // WE paused so recovery knows to resume (vs a user-initiated
        // pause that we shouldn't touch).
        self.appState?.updateStatus(
          .reconnecting,
          message: "Thermal critical — pausing video"
        )
        self.udpClient.pause()
        self.thermalPausedUDP = true
      case .serious:
        self.appState?.updateStatus(
          self.appState?.connectionState ?? .streaming,
          message: "Thermal serious — reducing load"
        )
      case .fair, .nominal:
        // Thermal has recovered. If WE paused the stream, resume it.
        // Otherwise leave alone — the watchdogs/user own the pause.
        if self.thermalPausedUDP {
          self.udpClient.resume()
          self.thermalPausedUDP = false
          self.appState?.updateStatus(
            .waitingForKeyframe,
            message: "Thermal recovered — resuming"
          )
          // Reset the watchdog clocks so we get a full grace period
          // before the next Tier 2/3 fire.
          self.resetFrameWatchdogClocks()
        }
      @unknown default:
        break
      }
    }
    thermalMonitor.start()
  }

  func stop() {
    isRunning = false
    endpointWatchdogTask?.cancel()
    endpointWatchdogTask = nil
    directCandidates.removeAll()
    activeDirectCandidateIndex = 0
    fallbackToDiscovery = false
    didResolveEndpoint = false
    didRegisterCurrentEndpoint = false
    discovery.stopBrowsing()
    hotspotProbe.stopProbing()
    udpClient.disconnect()
    decoder.stop()
    keyframeTimer?.invalidate()
    keyframeTimer = nil
    resetDisplay()
    thermalMonitor.stop()
    logger.info("stopped pipeline")
  }

  func handleBackground() {
    guard isRunning, didResolveEndpoint else { return }
    appState?.updateStatus(.paused)
    udpClient.pause()
    decoder.stop()
  }

  func handleForeground() {
    guard isRunning, didResolveEndpoint else { return }
    guard appState?.connectionState == .paused else { return }
    appState?.updateStatus(.waitingForKeyframe, message: "Resuming...")
    udpClient.resume()
    // Avoid the foreground transition itself triggering a frame watchdog
    // recovery before the first resumed frame arrives.
    resetFrameWatchdogClocks()
  }

  /// Attach to a running stream using saved Pi IPs (wlanIP → ztIP → Bonjour fallback).
  func attachToStream(appState: AppState) {
    guard let savedPi = PiStorage.load() else { return }
    var candidates = [StreamEndpoint]()
    if let wlan = savedPi.wlanIP, !wlan.isEmpty {
      candidates.append(StreamEndpoint(host: wlan, port: 3334, source: "wifi"))
    }
    if !savedPi.ztIP.isEmpty && !candidates.contains(where: { $0.host == savedPi.ztIP }) {
      candidates.append(StreamEndpoint(host: savedPi.ztIP, port: 3334, source: "zerotier"))
    }

    if candidates.isEmpty {
      start(appState: appState)
      return
    }
    connectDirectCandidates(candidates, appState: appState, fallbackToDiscovery: true)
  }

  /// Connect directly to a known host:port (e.g. ZeroTier IP), bypassing discovery.
  func connectDirect(host: String, port: UInt16, appState: AppState) {
    connectDirectCandidates(
      [StreamEndpoint(host: host, port: port, source: "direct")],
      appState: appState,
      fallbackToDiscovery: false
    )
  }

  // MARK: - Helpers

  private func configureCallbacks() {
    discovery.onServiceResolved = { [weak self] host, port in
      guard let self else { return }
      Task { @MainActor in
        self.connectToResolvedHost(host: host, port: port, source: "bonjour")
      }
    }

    discovery.onServiceLost = { [weak self] in
      guard let self else { return }
      Task { @MainActor in
        self.appState?.updateStatus(.reconnecting)
        self.didResolveEndpoint = false
        self.didRegisterCurrentEndpoint = false
        self.endpointWatchdogTask?.cancel()
        self.udpClient.disconnect()
        self.decoder.stop()
        self.resetDisplay()
        self.discovery.startBrowsing()
        self.hotspotProbe.startProbing(port: 3334)
      }
    }

    hotspotProbe.onProbeUpdate = { [weak self] message in
      guard let self else { return }
      Task { @MainActor in
        guard !self.didResolveEndpoint else { return }
        self.appState?.updateStatus(.probing, message: message)
      }
    }

    hotspotProbe.onHostDetected = { [weak self] host, port in
      guard let self else { return }
      Task { @MainActor in
        self.connectToResolvedHost(host: host, port: port, source: "probe")
      }
    }

    udpClient.onFrameReady = { [weak self] h264Data, isKeyframe in
      guard let self else { return }
      Task { @MainActor in
        self.didRegisterCurrentEndpoint = true
        self.endpointWatchdogTask?.cancel()
        self.recordFrame(bytes: UInt64(h264Data.count), isKeyframe: isKeyframe)
      }
      self.decoder.decode(h264Data: h264Data, isKeyframe: isKeyframe)
    }

    udpClient.onRegistered = { [weak self] in
      guard let self else { return }
      Task { @MainActor in
        self.didRegisterCurrentEndpoint = true
        self.endpointWatchdogTask?.cancel()
        if self.appState?.connectionState == .registering {
          self.appState?.updateStatus(.waitingForKeyframe)
        }
      }
    }

    udpClient.onForcedRekey = { [weak self] reason in
      guard let self else { return }
      Task { @MainActor in
        // Surface autonomous recovery to the HUD instead of the user
        // staring at a still frame in `.streaming` state.
        self.appState?.updateStatus(
          .waitingForKeyframe,
          message: "Recovering: \(reason)"
        )
      }
    }

    decoder.onFrameDecoded = { [weak self] sampleBuffer in
      guard let self else { return }
      Task { @MainActor in
        // Gate check: reject this frame if a concurrent reset happened
        // between the callback being scheduled and this Task running.
        // Without this gate we would enqueue onto a just-flushed layer
        // and race with the next connection attempt.
        guard self.acceptFrames else { return }

        // AVSampleBufferDisplayLayer is thread-safe for enqueue, but the
        // layer is part of the CoreAnimation tree so we touch it on main
        // for safety.
        if self.displayLayer.status == .failed {
          // The layer has shut down on us — flush and continue. If the
          // stream keeps failing the Tier 2/3 watchdog will kick a hard
          // recovery which rebuilds the whole pipeline.
          self.logger.error("display layer failed — flushing")
          self.displayLayer.flush()
        }
        self.displayLayer.enqueue(sampleBuffer)
        if !self.hasFirstFrame {
          self.hasFirstFrame = true
        }
        // Pipeline-healthy signal: a decoded frame actually reached the UI.
        self.lastFrameAt = Date()
      }
    }

    decoder.onDecodeFailure = { [weak self] in
      guard let self else { return }
      self.udpClient.waitForNextIDR()
      self.decoder.resetSession()
      Task { @MainActor in
        self.appState?.updateStatus(.waitingForKeyframe, message: "Decoder resync")
      }
    }
  }

  private func connectDirectCandidates(
    _ candidates: [StreamEndpoint],
    appState: AppState,
    fallbackToDiscovery: Bool
  ) {
    stop()
    isRunning = true
    self.appState = appState
    self.directCandidates = candidates
    self.activeDirectCandidateIndex = 0
    self.fallbackToDiscovery = fallbackToDiscovery
    self.didResolveEndpoint = false
    self.didRegisterCurrentEndpoint = false
    configureCallbacks()
    resetFrameWatchdogClocks()
    resetDisplay()
    acceptFrames = true
    startKeyframeTicker()
    startThermalMonitoring()
    attemptNextDirectCandidate()
  }

  private func attemptNextDirectCandidate() {
    guard let appState else { return }

    if activeDirectCandidateIndex >= directCandidates.count {
      if fallbackToDiscovery {
        appState.updateStatus(.discovering, message: "Falling back to discovery...")
        start(appState: appState)
      } else {
        appState.updateStatus(.error, message: "Unable to reach Pi stream endpoint")
      }
      return
    }

    let endpoint = directCandidates[activeDirectCandidateIndex]
    didResolveEndpoint = false
    didRegisterCurrentEndpoint = false
    connectToResolvedHost(host: endpoint.host, port: endpoint.port, source: endpoint.source)

    endpointWatchdogTask?.cancel()
    endpointWatchdogTask = Task { [weak self] in
      try? await Task.sleep(for: .seconds(3))
      await MainActor.run {
        self?.handleDirectCandidateTimeout()
      }
    }
  }

  private func handleDirectCandidateTimeout() {
    guard isRunning, !didRegisterCurrentEndpoint else { return }
    guard activeDirectCandidateIndex < directCandidates.count else { return }

    let endpoint = directCandidates[activeDirectCandidateIndex]
    logger.warning("direct endpoint timeout: \(endpoint.source, privacy: .public) \(endpoint.host, privacy: .public)")
    appState?.updateStatus(.reconnecting, message: "\(endpoint.source) unavailable, trying fallback...")
    udpClient.disconnect()
    decoder.stop()
    resetDisplay()
    activeDirectCandidateIndex += 1
    attemptNextDirectCandidate()
  }

  private func connectToResolvedHost(host: String, port: UInt16, source: String) {
    guard !didResolveEndpoint else { return }
    didResolveEndpoint = true

    logger.info("resolved host \(host, privacy: .public):\(port) via \(source, privacy: .public)")

    appState?.updateStatus(.registering, message: "\(source): \(host):\(port)")
    appState?.resolvedHost = host
    appState?.resolvedPort = port
    appState?.discoverySource = source

    discovery.stopBrowsing()
    hotspotProbe.stopProbing()
    udpClient.connect(host: host, port: port)

    // Endpoint may have changed — reset the frame watchdog so the new
    // connection's startup latency doesn't trip an immediate recovery.
    resetFrameWatchdogClocks()
  }

  private func recordFrame(bytes: UInt64, isKeyframe: Bool) {
    guard let appState else { return }

    if appState.connectionState == .waitingForKeyframe && isKeyframe {
      appState.updateStatus(.streaming)
    }
    appState.frameCount += 1
    appState.bytesReceived &+= bytes

    if isKeyframe {
      lastKeyframeAt = Date()
    }

    fpsWindowFrames += 1
    let elapsed = Date().timeIntervalSince(fpsWindowStart)
    if elapsed >= 1.0 {
      appState.measuredFPS = Double(fpsWindowFrames) / elapsed
      fpsWindowFrames = 0
      fpsWindowStart = Date()
    }
  }

  private func startKeyframeTicker() {
    keyframeTimer?.invalidate()
    let timer = Timer(timeInterval: 0.5, repeats: true) { [weak self] _ in
      guard let self else { return }
      Task { @MainActor in
        self.appState?.keyframeAgeSeconds =
          Date().timeIntervalSince(self.lastKeyframeAt)
        self.tickFrameWatchdog()
      }
    }
    RunLoop.main.add(timer, forMode: .common)
    keyframeTimer = timer
  }

  // MARK: - Frame watchdog (Tier 2+3)

  /// Gated 0.5s tick that escalates from soft (rekey + decoder reset) to
  /// hard (full reconnect) recovery based on how long it's been since a
  /// decoded frame actually reached the UI.
  private func tickFrameWatchdog() {
    // Only watch while the pipeline is supposed to be producing frames.
    // `.paused` and `.registering` are legitimate quiet periods; don't
    // false-positive during them.
    guard let state = appState?.connectionState else { return }
    guard state == .streaming || state == .waitingForKeyframe else { return }

    let stall = Date().timeIntervalSince(lastFrameAt)
    let now = Date()

    if stall >= frameStallHardSeconds
      && now.timeIntervalSince(lastHardRecoveryAt) >= frameStallHardSeconds
    {
      lastHardRecoveryAt = now
      logger.warning(
        "frame watchdog: HARD stall \(stall, privacy: .public)s — full reconnect"
      )
      performHardRecovery()
    } else if stall >= frameStallSoftSeconds
      && now.timeIntervalSince(lastSoftRecoveryAt) >= frameStallSoftSeconds
    {
      lastSoftRecoveryAt = now
      logger.warning(
        "frame watchdog: SOFT stall \(stall, privacy: .public)s — rekey + decoder reset"
      )
      performSoftRecovery()
    }
  }

  /// Tier 2: cheap recovery — force a fresh keyframe from the Pi and
  /// rebuild the decoder's internal state. Run this before tearing the
  /// whole UDP connection down.
  private func performSoftRecovery() {
    udpClient.forceRekey(reason: "frame stall (soft)")
    decoder.resetSession()
    appState?.updateStatus(.waitingForKeyframe, message: "Resync...")
  }

  /// Tier 3: expensive recovery — fully disconnect and reconnect against
  /// the SAME verified endpoint. Only fires if the soft recovery didn't
  /// unblock frames within `frameStallHardSeconds`.
  ///
  /// We deliberately reuse `appState.resolvedHost`/`resolvedPort` instead
  /// of restarting the candidate carousel (wlanIP → ztIP → Bonjour): if
  /// this endpoint worked before, we trust it still works and the stall
  /// is not a routing problem.
  private func performHardRecovery() {
    guard let host = appState?.resolvedHost,
          let port = appState?.resolvedPort
    else {
      logger.error("hard recovery: no resolved endpoint, falling back to discovery")
      if let appState { start(appState: appState) }
      return
    }

    appState?.updateStatus(.reconnecting, message: "Stream stalled, reconnecting...")
    udpClient.disconnect()
    decoder.stop()
    resetDisplay()

    // `udpClient.connect` internally calls `disconnect` too, so this is
    // idempotent; the explicit call above is for deterministic decoder
    // and display-layer teardown.
    udpClient.connect(host: host, port: port)

    resetFrameWatchdogClocks()
  }

  /// Reset all watchdog clocks so that the next half-second of grace
  /// won't falsely trigger recovery. Called on start, on endpoint change,
  /// on foreground resume, and after hard recovery.
  private func resetFrameWatchdogClocks() {
    lastFrameAt = Date()
    lastSoftRecoveryAt = Date.distantPast
    lastHardRecoveryAt = Date.distantPast
  }
}
