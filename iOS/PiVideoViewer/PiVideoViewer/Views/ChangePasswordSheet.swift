import SwiftUI

struct ChangePasswordSheet: View {
    @Environment(\.dismiss) var dismiss
    @EnvironmentObject var bleManager: BLEManager

    @State private var oldPin = ""
    @State private var newPin = ""
    @State private var errorMessage: String?
    @State private var isLoading = false

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    SecureField("Old Password", text: $oldPin)
                        .keyboardType(.numberPad)
                    SecureField("New Password", text: $newPin)
                        .keyboardType(.numberPad)
                } header: {
                    Text("Change PIN Code")
                }

                if let error = errorMessage {
                    Section {
                        Text(error)
                            .foregroundColor(.red)
                    }
                }

                Section {
                    Button {
                        changePassword()
                    } label: {
                        HStack {
                            Spacer()
                            if isLoading {
                                ProgressView()
                            } else {
                                Text("Change")
                            }
                            Spacer()
                        }
                    }
                    .disabled(oldPin.count < 4 || newPin.count < 4 || isLoading)
                }
            }
            .navigationTitle("Change Password")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
    }

    private func changePassword() {
        guard oldPin.count >= 4 && oldPin.count <= 8 && oldPin.allSatisfy(\.isNumber) else {
            errorMessage = "Old PIN must be 4-8 digits"
            return
        }
        guard newPin.count >= 4 && newPin.count <= 8 && newPin.allSatisfy(\.isNumber) else {
            errorMessage = "New PIN must be 4-8 digits"
            return
        }

        isLoading = true
        errorMessage = nil

        Task {
            do {
                let response = try await bleManager.sendCommand(
                    BLECommand.authChange(old: oldPin, new: newPin)
                )
                let parsed = BLEResponse.parse(response)
                if parsed.isOK {
                    PiStorage.savePin(newPin)
                    dismiss()
                } else {
                    errorMessage = parsed.message == "WRONG_PIN" ? "Incorrect old password" : parsed.message
                }
            } catch {
                errorMessage = error.localizedDescription
            }
            isLoading = false
        }
    }
}
