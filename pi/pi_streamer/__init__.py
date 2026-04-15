"""Raspberry Pi video streamer for PiVideoViewer iOS app.

Captures from Pi Cam 3 Wide, runs Hailo pose inference, bakes overlays
into the frame, encodes to H.264, and streams chunked UDP to the iPhone
using the same wire protocol as the ESP32-P4 project.
"""

__version__ = "0.1.0"
