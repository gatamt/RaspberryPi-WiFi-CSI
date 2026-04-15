import AVFoundation
import SwiftUI
import UIKit

/// SwiftUI wrapper for an `AVSampleBufferDisplayLayer`. The layer is owned
/// by `VideoViewModel` so the decoder callback can enqueue into it directly.
///
/// The older pipeline went VT decode → CVPixelBuffer → CIImage →
/// CIContext.createCGImage → SwiftUI `Image`, which allocated per frame,
/// touched user-space pixels twice and blocked on `CIContext`. Handing a
/// `CMSampleBuffer` straight to `AVSampleBufferDisplayLayer` lets
/// VideoToolbox decode and CoreAnimation composite without any of that.
struct VideoDisplayView: UIViewRepresentable {
  let displayLayer: AVSampleBufferDisplayLayer

  func makeUIView(context: Context) -> VideoDisplayContainerView {
    let view = VideoDisplayContainerView()
    view.backgroundColor = .black
    view.displayLayer = displayLayer
    return view
  }

  func updateUIView(_ uiView: VideoDisplayContainerView, context: Context) {
    // The layer itself is stable; updates come via enqueue(). We only
    // refresh the hosted layer when the identity changes (e.g. after a
    // decoder reset that swapped the layer).
    if uiView.displayLayer !== displayLayer {
      uiView.displayLayer = displayLayer
    }
  }
}

/// UIView that hosts an AVSampleBufferDisplayLayer and resizes it to
/// match the view's bounds on layout.
final class VideoDisplayContainerView: UIView {
  var displayLayer: AVSampleBufferDisplayLayer? {
    didSet {
      if let oldValue, oldValue !== displayLayer {
        oldValue.removeFromSuperlayer()
      }
      if let displayLayer, displayLayer.superlayer !== self.layer {
        displayLayer.videoGravity = .resizeAspect
        self.layer.addSublayer(displayLayer)
        setNeedsLayout()
      }
    }
  }

  override func layoutSubviews() {
    super.layoutSubviews()
    displayLayer?.frame = bounds
  }
}
