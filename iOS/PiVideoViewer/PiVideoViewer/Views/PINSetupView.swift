import SwiftUI

struct PINSetupView: View {
    @ObservedObject var router: AppRouter
    @ObservedObject var bleManager: BLEManager

    @State private var pin = ""
    @State private var hasPIN: Bool?
    @State private var errorMessage: String?
    @State private var isLoading = false
    @State private var shake = false

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack(spacing: 24) {
                Spacer()

                Image(systemName: hasPIN == true ? "lock.fill" : "lock.open.fill")
                    .font(.system(size: 48))
                    .foregroundColor(.blue)

                Text(titleText)
                    .font(.title2.bold())
                    .foregroundColor(.white)

                Text(bodyText)
                    .font(.body)
                    .foregroundColor(.gray)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 40)

                if hasPIN == nil {
                    ProgressView()
                        .tint(.white)
                } else {
                    SecureField("PIN", text: $pin)
                        .keyboardType(.numberPad)
                        .textContentType(.password)
                        .font(.title)
                        .multilineTextAlignment(.center)
                        .padding()
                        .background(Color(.systemGray5).opacity(0.3))
                        .cornerRadius(12)
                        .padding(.horizontal, 60)
                        .modifier(ShakeEffect(shakes: shake ? 2 : 0))
                }

                if let error = errorMessage {
                    Text(error)
                        .foregroundColor(.red)
                        .font(.callout)
                }

                Button {
                    submitPIN()
                } label: {
                    HStack {
                        if isLoading {
                            ProgressView()
                                .tint(.white)
                        }
                        Text(buttonTitle)
                    }
                    .font(.headline)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(canSubmit ? Color.blue : Color.gray)
                    .cornerRadius(14)
                }
                .disabled(!canSubmit || isLoading)
                .padding(.horizontal, 40)

                Spacer()
                Spacer()
            }
        }
        .onAppear {
            checkPINStatus()
        }
    }

    private var titleText: String {
        switch hasPIN {
        case .some(true):
            return "Enter PIN"
        case .some(false):
            return "Set a PIN for your Pi"
        case .none:
            return "Checking Pi Security"
        }
    }

    private var bodyText: String {
        switch hasPIN {
        case .some(true):
            return "Enter the PIN code to unlock your Pi."
        case .some(false):
            return "Choose a 4-8 digit PIN to secure your Pi."
        case .none:
            return "Reading the Pi's current PIN status before showing the correct flow."
        }
    }

    private var buttonTitle: String {
        hasPIN == true ? "Unlock" : "Set PIN"
    }

    private var canSubmit: Bool {
        hasPIN != nil && pin.count >= 4
    }

    private func checkPINStatus() {
        Task {
            // Wait for BLE to be ready
            for _ in 0..<30 {
                if bleManager.connectionState == .ready { break }
                try await Task.sleep(for: .milliseconds(100))
            }
            guard bleManager.connectionState == .ready else { return }

            do {
                hasPIN = try await bleManager.queryAuthStatus()
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    private func submitPIN() {
        guard pin.count >= 4 && pin.count <= 8 else {
            errorMessage = "PIN must be 4-8 digits"
            return
        }
        guard pin.allSatisfy(\.isNumber) else {
            errorMessage = "PIN must contain only digits"
            return
        }

        isLoading = true
        errorMessage = nil

        Task {
            do {
                let command: String
                if hasPIN == true {
                    command = BLECommand.authVerify(pin: pin)
                } else {
                    command = BLECommand.authSet(pin: pin)
                }

                let response = try await bleManager.sendCommand(command)
                let parsed = BLEResponse.parse(response)

                if parsed.isOK {
                    PiStorage.savePin(pin)
                    await router.handleSuccessfulAuth()
                } else {
                    errorMessage = parsed.message
                    withAnimation(.default) { shake.toggle() }
                    pin = ""
                }
            } catch {
                errorMessage = error.localizedDescription
            }
            isLoading = false
        }
    }
}

struct ShakeEffect: GeometryEffect {
    var shakes: CGFloat
    var animatableData: CGFloat {
        get { shakes }
        set { shakes = newValue }
    }

    func effectValue(size: CGSize) -> ProjectionTransform {
        ProjectionTransform(
            CGAffineTransform(translationX: 10 * sin(shakes * .pi * 2), y: 0)
        )
    }
}
