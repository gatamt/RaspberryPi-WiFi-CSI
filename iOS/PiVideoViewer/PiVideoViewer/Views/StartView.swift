import SwiftUI

struct StartView: View {
    @ObservedObject var router: AppRouter

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack(spacing: 32) {
                Spacer()

                Image(systemName: "antenna.radiowaves.left.and.right")
                    .font(.system(size: 64))
                    .foregroundColor(.blue)

                Text("PiVideoViewer")
                    .font(.largeTitle.bold())
                    .foregroundColor(.white)

                Text("Connect to your Raspberry Pi via Bluetooth to get started.")
                    .font(.body)
                    .foregroundColor(.gray)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 40)

                Spacer()

                Button {
                    router.currentScreen = .bleScan
                } label: {
                    HStack {
                        Image(systemName: "wave.3.right")
                        Text("Connect to Raspberry Pi")
                    }
                    .font(.headline)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .cornerRadius(14)
                }
                .padding(.horizontal, 40)

                Spacer().frame(height: 60)
            }
        }
    }
}
