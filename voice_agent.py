import io
import pygame
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

client = ElevenLabs(api_key="sk_3952c2bc91a980c3d6816017dfe1df468a5b7072584359c5")

# Rate / Speed control:
# 1.0 = normal speed
# > 1.0 = faster (e.g., 1.15 or 1.2)
# < 1.0 = slower (e.g., 0.85 or 0.9)
# Recommended range: 0.7 to 1.2
SPEECH_SPEED = 1.2

# Generate natural audio stream with rate control
audio = client.text_to_speech.convert(
    text="Hi, we are team prime syndicate. We introduce UNMUTED!",
    voice_id="JBFqnCBsd6RMkjVDRZzb", # e.g. "George" or another free voice
    model_id="eleven_flash_v2_5",      # Lowest latency model
    voice_settings=VoiceSettings(
        stability=0.5,
        similarity_boost=0.75,
        speed=SPEECH_SPEED
    )
)

# Play audio cleanly on Windows without needing external CLI tools (mpv/ffmpeg)
audio_bytes = b"".join(audio)
pygame.mixer.init()
pygame.mixer.music.load(io.BytesIO(audio_bytes))
pygame.mixer.music.play()
while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)
