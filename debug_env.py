import sys
try:
    import mediapipe
    print(f"MediaPipe version: {mediapipe.__version__}")
except ImportError:
    print("MediaPipe not installed")

try:
    import google.protobuf
    print(f"Protobuf version: {google.protobuf.__version__}")
except ImportError:
    print("Protobuf not installed")

print(f"Python version: {sys.version}")
