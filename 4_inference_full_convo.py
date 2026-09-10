import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import time
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import threading
import subprocess
import re

# --- CONFIGURATION ---
terminal_logs = []
log_lock = threading.Lock()

def add_terminal_log(msg):
    with log_lock:
        clean_msg = str(msg).encode('ascii', 'replace').decode('ascii')
        terminal_logs.append(clean_msg)
        if len(terminal_logs) > 15:
            del terminal_logs[:-15]

def get_terminal_logs_snapshot():
    with log_lock:
        return list(terminal_logs)

DATA_PATH = os.path.join('data')

# Automatically detect actions from the data folder (must match training)
if os.path.exists(DATA_PATH):
    actions = np.array(sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]))
else:
    print(f"Error: Data folder '{DATA_PATH}' not found.")
    actions = np.array([])

print(f"Loaded {len(actions)} actions: {actions.tolist()}")

# Display names mapping (folder name -> display text)
display_names = {
    'good_morning': 'Good Morning',
    'hello': 'Hello',
    'help': 'Help',
    'hi': 'Hi',
    'how_are_you_': 'How Are You?',
    'i': 'I',
    'introduce': 'Introduce',
    'present': 'Present',
    'thankyou': 'Thank You',
    'judges': 'Introduce',
    'unmuted': 'Unmuted',
    'we': 'We',
}

def get_display_name(action):
    """Get the display-friendly name for an action."""
    return display_names.get(action, action.replace('_', ' ').title())

def compile_conversational_sentence(tokens):
    """
    Transforms raw sign glosses into natural, warm, conversational spoken English.
    - Hardcodes 'Unmuted' so it NEVER appears in between words; always placed strictly at the end.
    - Cleans up repetitive tokens and adds natural conversational punctuation.
    - E.g.: ['Hi', 'Hello', 'How Are You?'] -> 'Hey! Hello, how are you doing?'
    - E.g.: ['We', 'Judges', 'Introduce', 'Unmuted'] -> 'We, judges, would like to introduce... Unmuted!'
    """
    if not tokens:
        return ""

    # 1. Hardcode 'Unmuted': Strip from body so it ONLY comes at the end
    has_unmuted = any(t.strip().lower() == 'unmuted' for t in tokens)
    clean_tokens = [t.strip() for t in tokens if t.strip().lower() != 'unmuted']
    lower_tokens = [t.lower().replace('_', ' ').replace('?', '').strip() for t in clean_tokens]

    phrases = []
    i = 0
    # Group leading greetings
    greetings = []
    while i < len(lower_tokens) and lower_tokens[i] in ['hi', 'hello', 'good morning', 'how are you']:
        greetings.append(lower_tokens[i])
        i += 1

    if greetings:
        if 'hi' in greetings and 'hello' in greetings and 'how are you' in greetings:
            phrases.append("Hey! Hello, how are you doing?")
        elif 'hi' in greetings and 'hello' in greetings:
            phrases.append("Hey! Hello there,")
        elif 'hi' in greetings and 'how are you' in greetings:
            phrases.append("Hey! How are you doing?")
        elif 'hello' in greetings and 'how are you' in greetings:
            phrases.append("Hello! How are you doing?")
        elif 'good morning' in greetings and 'how are you' in greetings:
            phrases.append("Good morning! How are you doing today?")
        elif 'good morning' in greetings:
            phrases.append("Good morning,")
        elif 'how are you' in greetings:
            phrases.append("How are you doing?")
        elif 'hi' in greetings:
            phrases.append("Hey there!")
        elif 'hello' in greetings:
            phrases.append("Hello!")

    rest_tokens = lower_tokens[i:]
    if rest_tokens:
        rest_str = ' '.join(rest_tokens)
        if rest_str in ['we judges introduce', 'we judges present introduce']:
            phrases.append("we, judges, would like to introduce")
        elif rest_str in ['we judges present']:
            phrases.append("we, judges, proudly present")
        elif rest_str in ['we introduce']:
            phrases.append("we would like to introduce")
        elif rest_str in ['we present']:
            phrases.append("we are proud to present")
        elif rest_str in ['i introduce']:
            phrases.append("I would like to introduce")
        elif rest_str == 'help':
            phrases.append("could you please help me?")
        elif rest_str in ['thankyou', 'thank you']:
            phrases.append("thank you so much.")
        elif rest_str == 'judges':
            phrases.append("honorable judges,")
        else:
            mapped = []
            for idx, t in enumerate(rest_tokens):
                if t == 'we': mapped.append("we")
                elif t == 'judges': mapped.append("judges,")
                elif t == 'introduce': mapped.append("would like to introduce....Unmuted!")
                elif t == 'present': mapped.append("present")
                elif t in ['thankyou', 'thank you']: mapped.append("thank you")
                elif t == 'i': mapped.append("I")
                elif t == 'help': mapped.append("need help")
                else: mapped.append(t)
            body = ' '.join(mapped)
            phrases.append(body)

    full = ' '.join(phrases).strip()
    if full:
        # Clean up any spacing / punctuation glitches
        full = full.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?')
        full = full.replace(',,', ',').replace(',.', '.')
        # Capitalize sentence starts after punctuation
        full = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), full)
        full = full[0].upper() + full[1:]

    # Hardcode 'Unmuted' strictly at the end with a dramatic conversational pause
    if has_unmuted:
        if full:
            full = full.rstrip(',.!? ') + "... Unmuted!"
        else:
            full = "Unmuted!"
    elif full:
        full = full.rstrip(', ')
        if not full.endswith(('.', '!', '?')):
            full += "."

    return full

sequence_length = 30
threshold = 0.45          # Prediction confidence threshold (lowered significantly for fast locking)
CONFIDENCE_MARGIN = 0.05  # Lowered margin so close runner-ups do not delay locking
CONSISTENCY_FRAMES = 3    # Anti-jitter frames (fallback)
MIN_ACTIVE_VELOCITY = 0.003 # Active movement gate; highly sensitive for fast stationary sign locking
SIGN_HOLD_DURATION = 0.25 # Fast lock: hold for only ~0.25s (~250ms) to confirm sign
SPEAK_COOLDOWN = 0.6      # Seconds before repeating the same word
IDLE_TIMEOUT_HANDS_DOWN = 1.2 # 1.2s after hands genuinely stay down (>0.6s) to speak sentence
IDLE_TIMEOUT_HANDS_UP = 2.5   # 2.5s pause between words while signing (gives ample time to chain signs like judges!)
CAMERA_INDEX = None   # Set to 0, 1, 2 for a specific camera, or None for auto-detect

# Check if camera index passed via command line argument (e.g. `python 3_inference_pc.py 1`)
import sys
if len(sys.argv) > 1 and sys.argv[1].isdigit():
    CAMERA_INDEX = int(sys.argv[1])

# --- CAMERA EXPOSURE & BRIGHTNESS TUNING ---
# Tweak these values if the room lighting changes:
EXPOSURE_ALPHA = 0.88   # Contrast scale (1.0 = normal, 0.85-0.90 = reduces harsh glare)
EXPOSURE_BETA = -30     # Brightness offset (0 = normal, -15 to -30 = slightly dimmer)

# --- TTS Setup (ElevenLabs) ---
import io
import pygame
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

# Initialize pygame mixer for audio playback
pygame.mixer.init()
el_client = ElevenLabs(api_key="sk_3952c2bc91a980c3d6816017dfe1df468a5b7072584359c5")

# --- KINEMATICS & PROSODY TUNING ---
# Lower VELOCITY_MAX means less hand shaking required to reach peak Style and Speed
VELOCITY_MIN = 0.15        # Calm gesture (m/s)
VELOCITY_MAX = 0.70        # High-urgency / Fast sign (m/s)
THRESHOLD_EXCITED = 0.65   # Velocity threshold to trigger excited exclamation (m/s)

MIN_STYLE, MAX_STYLE = 0.05, 1.00
MIN_STABILITY, MAX_STABILITY = 0.25, 0.75
MIN_SPEED, MAX_SPEED = 0.85, 1.15

class MetricHandTracker:
    def __init__(self, frame_width=640, frame_height=480):
        self.focal_length = frame_width
        self.cx = frame_width / 2.0
        self.cy = frame_height / 2.0
        
        # EMA State
        self.ema_speed = 0.0
        self.alpha = 0.6
        self.last_P = None
        self.last_time = None
        self.current_Z = 0.0
        
    def update(self, u0, v0, u9, v9, current_time):
        pixel_dist = np.sqrt((u9 - u0)**2 + (v9 - v0)**2)
        
        # 3D Metric Depth (Z) Estimation
        Z = (self.focal_length * 0.09) / max(pixel_dist, 1e-5)
        self.current_Z = Z
        
        # Back-project 2D to 3D Metric
        X = (u0 - self.cx) * Z / self.focal_length
        Y = (v0 - self.cy) * Z / self.focal_length
        P_metric = np.array([X, Y, Z])
        
        speed_mps = 0.0
        if self.last_P is not None and self.last_time is not None:
            dt = current_time - self.last_time
            if dt > 0:
                delta_P = P_metric - self.last_P
                raw_speed = np.linalg.norm(delta_P) / dt
                
                # 1D Kalman / EMA Filter
                self.ema_speed = self.alpha * raw_speed + (1 - self.alpha) * self.ema_speed
                speed_mps = self.ema_speed
                
        self.last_P = P_metric
        self.last_time = current_time
        return speed_mps, Z

metric_tracker = MetricHandTracker(640, 480)

tts_lock = threading.Lock()
is_speaking = False

def speak(text, velocity=0.0):
    """Speaks natural conversational text using ElevenLabs with expressive warm delivery."""
    global is_speaking
    if is_speaking or not text.strip():
        return

    def _speak():
        global is_speaking
        with tts_lock:
            is_speaking = True
            try:
                # 1. DYNAMIC EMOTION & PARAMETER TUNING
                norm_v = np.clip((velocity - VELOCITY_MIN) / (VELOCITY_MAX - VELOCITY_MIN), 0.0, 1.0)
                
                # Conversational tuning parameters requested by user:
                # stability = 0.35 (lower stability = richer human intonation, prevents monotone read)
                dynamic_stability = round(float(np.clip(0.35 - (norm_v - 0.5) * 0.06, 0.28, 0.38)), 2)
                
                # similarity_boost = 0.80
                dynamic_similarity = 0.80
                
                # style = 0.55 (boosts expressive delivery)
                dynamic_style = round(float(np.clip(0.55 + (norm_v - 0.5) * 0.12, 0.50, 0.68)), 2)
                
                # speed = 1.0 (clamped strictly between 0.90 and 1.15)
                dynamic_speed = round(float(np.clip(1.00 + (norm_v - 0.5) * 0.15, 0.90, 1.15)), 2)
                
                # 2. NATURAL CONVERSATIONAL TEXT (NO FLAT UPPERCASE)
                spoken_text = text.strip()
                if velocity >= THRESHOLD_EXCITED:
                    log_prefix = "[expressive]"
                else:
                    log_prefix = "[conversational]"
                    
                msg1 = f"[CONVO ENGINE] Phrase: \"{spoken_text}\" | Total Words: {len(spoken_text.split())} | Overall Avg Vel: {velocity:.4f} -> Speed: {dynamic_speed:.2f}"
                msg2 = f"[Voice Agent] Synthesizing: '{log_prefix} {spoken_text}' | Style: {dynamic_style:.2f} | Stab: {dynamic_stability:.2f} | Sim: {dynamic_similarity:.2f} | Spd: {dynamic_speed:.2f}"
                print(msg1)
                print(msg2)
                add_terminal_log(msg1)
                add_terminal_log(msg2)

                # 3. VOICE SETTINGS PAYLOAD WITH RETRY
                for attempt in range(2):
                    try:
                        audio = el_client.text_to_speech.convert(
                            text=spoken_text,
                            voice_id="JBFqnCBsd6RMkjVDRZzb",
                            model_id="eleven_multilingual_v2",  # Crucial for emotional inflection
                            voice_settings=VoiceSettings(
                                stability=dynamic_stability,
                                similarity_boost=dynamic_similarity,
                                style=dynamic_style,
                                use_speaker_boost=True,
                                speed=dynamic_speed
                            )
                        )
                        audio_bytes = b"".join(audio)
                        pygame.mixer.music.load(io.BytesIO(audio_bytes))
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            pygame.time.Clock().tick(10)
                        break
                    except Exception as net_err:
                        if attempt == 0:
                            time.sleep(0.4)
                            continue
                        print(f"[TTS ERROR] Network/Connection dropped: {net_err}")
            except Exception as e:
                print(f"[TTS ERROR] {e}")
            finally:
                is_speaking = False
                
    threading.Thread(target=_speak, daemon=True).start()

# --- MEDIAPIPE SETUP (HANDS ONLY) ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

class HandLockTracker:
    """
    Persistent 2-hand ID-locking registry to prevent 3rd-party hands from interfering.
    Maintains active slots {0: None, 1: None}.
    """
    def __init__(self, max_match_dist=0.25, max_lost_frames=15):
        self.max_match_dist = max_match_dist
        self.max_lost_frames = max_lost_frames
        self.locked_hands = {0: None, 1: None}
        
    def update(self, detected_hands):
        # detected_hands: list of dicts: {'wrist': (x, y), 'landmarks': hand_curr, 'handedness': str, 'bbox': (x1, y1, x2, y2)}
        matched_detect_indices = set()
        active_slots = [s for s in (0, 1) if self.locked_hands[s] is not None]
        
        # 1. Match active locked slots to nearest incoming detection based on Euclidean distance
        pairs = []
        for s in active_slots:
            last_w = self.locked_hands[s]['wrist']
            for d_idx, d_hand in enumerate(detected_hands):
                w = d_hand['wrist']
                dist = np.sqrt((w[0] - last_w[0])**2 + (w[1] - last_w[1])**2)
                if dist < self.max_match_dist:
                    pairs.append((dist, s, d_idx))
                    
        # Greedy assignment (closest first)
        pairs.sort(key=lambda x: x[0])
        matched_slots = set()
        for dist, s, d_idx in pairs:
            if s not in matched_slots and d_idx not in matched_detect_indices:
                matched_slots.add(s)
                matched_detect_indices.add(d_idx)
                d = detected_hands[d_idx]
                self.locked_hands[s]['wrist'] = d['wrist']
                self.locked_hands[s]['landmarks'] = d['landmarks']
                self.locked_hands[s]['handedness'] = d['handedness']
                self.locked_hands[s]['bbox'] = d['bbox']
                self.locked_hands[s]['lost_frames'] = 0
                
        # 2. Increment lost frames for unmatched active slots
        for s in active_slots:
            if s not in matched_slots:
                self.locked_hands[s]['lost_frames'] += 1
                self.locked_hands[s]['landmarks'] = None
                if self.locked_hands[s]['lost_frames'] > self.max_lost_frames:
                    self.locked_hands[s] = None
                    
        # 3. For unmatched detected hands: lock into open slots if available
        unmatched_detections = [d for idx, d in enumerate(detected_hands) if idx not in matched_detect_indices]
        rejected_hands = []
        
        for d in unmatched_detections:
            open_slots = [s for s in (0, 1) if self.locked_hands[s] is None]
            if open_slots:
                s = open_slots[0]
                self.locked_hands[s] = {
                    'id': s,
                    'wrist': d['wrist'],
                    'lost_frames': 0,
                    'landmarks': d['landmarks'],
                    'handedness': d['handedness'],
                    'bbox': d['bbox']
                }
            else:
                # Both slots occupied -> strictly reject 3rd party hand
                rejected_hands.append(d)
                
        return self.locked_hands, rejected_hands

def draw_target_reticle(image, center_pt, size=20, color=(0, 255, 0), thickness=2):
    """Renders high-tech target reticle brackets around center_pt."""
    cx, cy = center_pt
    arm = max(6, size // 3)
    # Top-left
    cv2.line(image, (cx - size, cy - size), (cx - size + arm, cy - size), color, thickness)
    cv2.line(image, (cx - size, cy - size), (cx - size, cy - size + arm), color, thickness)
    # Top-right
    cv2.line(image, (cx + size, cy - size), (cx + size - arm, cy - size), color, thickness)
    cv2.line(image, (cx + size, cy - size), (cx + size, cy - size + arm), color, thickness)
    # Bottom-left
    cv2.line(image, (cx - size, cy + size), (cx - size + arm, cy + size), color, thickness)
    cv2.line(image, (cx - size, cy + size), (cx - size, cy + size - arm), color, thickness)
    # Bottom-right
    cv2.line(image, (cx + size, cy + size), (cx + size - arm, cy + size), color, thickness)
    cv2.line(image, (cx + size, cy + size), (cx + size, cy + size - arm), color, thickness)
    # Center dot
    cv2.circle(image, (cx, cy), 2, color, -1)

def draw_styled_landmarks(image, results):
    """Fallback landmark renderer."""
    if results and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                     mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(250,44,250), thickness=2, circle_radius=2))

def draw_locked_and_rejected_hands(image, locked_hands, rejected_hands):
    """
    Renders Electric Cyan / Bright Green annotations for locked user hands,
    and subtle red warning boxes for rejected 3rd-party interference.
    """
    h, w, c = image.shape
    
    # 1. Draw LOCKED USER HANDS
    for slot in (0, 1):
        info = locked_hands.get(slot)
        if info is not None and info.get('landmarks') is not None:
            hand_lm = info['landmarks']
            # Draw landmark skeleton in electric cyan / bright green
            mp_drawing.draw_landmarks(
                image, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)
            )
            
            # Bounding box
            x1, y1, x2, y2 = info['bbox']
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Reticle around wrist
            wrist_lm = hand_lm.landmark[0]
            wx, wy = int(wrist_lm.x * w), int(wrist_lm.y * h)
            draw_target_reticle(image, (wx, wy), size=18, color=(0, 255, 0), thickness=2)
            
            # High-tech HUD pill label
            label = f"[LOCKED TARGET: USER_H{slot}]"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            label_x1 = max(0, min(x1, w - tw - 10))
            label_y1 = max(0, y1 - th - 8)
            cv2.rectangle(image, (label_x1, label_y1), (label_x1 + tw + 6, max(th + 8, y1)), (0, 40, 0), -1)
            cv2.rectangle(image, (label_x1, label_y1), (label_x1 + tw + 6, max(th + 8, y1)), (0, 255, 0), 1)
            cv2.putText(image, label, (label_x1 + 3, max(th + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    # 2. Draw REJECTED 3RD-PARTY HANDS
    for rej in rejected_hands:
        rx1, ry1, rx2, ry2 = rej['bbox']
        # Subtle red bounding box
        cv2.rectangle(image, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
        
        # Subtle red landmarks
        if rej.get('landmarks') is not None:
            mp_drawing.draw_landmarks(
                image, rej['landmarks'], mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 180), thickness=1, circle_radius=2),
                mp_drawing.DrawingSpec(color=(50, 50, 220), thickness=1, circle_radius=1)
            )
            
        label = "[IGNORED // 3RD_PARTY_INTERFERENCE]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        label_x1 = max(0, min(rx1, w - tw - 10))
        label_y1 = max(0, ry1 - th - 8)
        cv2.rectangle(image, (label_x1, label_y1), (label_x1 + tw + 6, max(th + 8, ry1)), (0, 0, 50), -1)
        cv2.rectangle(image, (label_x1, label_y1), (label_x1 + tw + 6, max(th + 8, ry1)), (0, 0, 255), 1)
        cv2.putText(image, label, (label_x1 + 3, max(th + 2, ry1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)

def extract_keypoints(locked_hands):
    """
    Extracts keypoints STRICTLY from locked hands (Slot 0 and Slot 1).
    3rd-party hands are completely excluded.
    """
    lh = np.zeros(21*3)
    rh = np.zeros(21*3)
    used_lh = False
    
    for slot in (0, 1):
        info = locked_hands.get(slot)
        if info is not None and info.get('landmarks') is not None:
            hand_curr = info['landmarks']
            handedness = info.get('handedness', 'Right')
            
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_curr.landmark])
            
            # 1. Zero-Center to Wrist (makes it invariant to screen position)
            wrist = landmarks[0]
            landmarks = landmarks - wrist
            
            # 2. Depth/Scale Normalization (makes it invariant to camera distance)
            x_max, x_min = np.max(landmarks[:, 0]), np.min(landmarks[:, 0])
            y_max, y_min = np.max(landmarks[:, 1]), np.min(landmarks[:, 1])
            box_size = max(x_max - x_min, y_max - y_min)
            
            if box_size > 0:
                landmarks = landmarks / box_size
                
            flattened = landmarks.flatten()
            
            if handedness == 'Right' and not used_lh: # Swapped per user request
                lh = flattened
                used_lh = True
            elif handedness == 'Left' and np.all(rh == 0):
                rh = flattened
            elif not used_lh:
                lh = flattened
                used_lh = True
            else:
                rh = flattened
                
    return np.concatenate([lh, rh])


# --- MAIN ---

# Load Model
model_path = os.path.join('models', 'unmuted_brain.h5')
# print(f"Loading model from {model_path}...")
try:
    model = load_model(model_path)
    print("Model loaded.")
except:
    print("WARNING: Model not found. Run 2_train_model.py first! (But script will run for testing)")
    model = None

# Variables
sequence = []
sentence = []
predictions = []
last_spoken_word = ""
last_spoken_time = 0
last_avg_v = 0.0

# Convo builder state
convo_tokens = []
convo_velocities = []
last_token_time = 0.0
first_word_ignored = False

# Sign hold state (slightly less than 1 sec to sign)
candidate_sign_idx = None
candidate_start_time = 0.0
candidate_last_seen = 0.0
hands_enter_time = 0.0
last_hand_seen_time = 0.0
hand_tracker = HandLockTracker(max_match_dist=0.25, max_lost_frames=15)

# VidStab Initialization
try:
    from vidstab import VidStab
    stabilizer = VidStab(kp_method='FAST')
except ImportError:
    print("[WARNING] vidstab not installed. Run: pip install vidstab")
    stabilizer = None

stabilization_mode = False  # Toggle via 's' key
tracking_mode = True  # Toggle via 't' key

def open_working_camera(preferred_idx=CAMERA_INDEX):
    """Tries the preferred camera index first, then others if none specified or on failure."""
    indices = [preferred_idx] if preferred_idx is not None else [0, 1, 2]
    for idx in indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"[CAMERA] Using DirectShow camera index {idx}")
                return cap
            cap.release()
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"[CAMERA] Using camera index {idx}")
                return cap
            cap.release()
    return None

cap = open_working_camera()
if cap is None:
    print(f"[ERROR] Could not open camera {CAMERA_INDEX if CAMERA_INDEX is not None else '(auto)'}. Please check your webcam connection.")
    exit(1)

# Hand model (detects up to 4 hands to isolate 3rd-party interference)
with mp_hands.Hands(max_num_hands=4, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        # Apply slight exposure / glare reduction
        if EXPOSURE_ALPHA != 1.0 or EXPOSURE_BETA != 0:
            frame = cv2.convertScaleAbs(frame, alpha=EXPOSURE_ALPHA, beta=EXPOSURE_BETA)

        # detection
        image, results = mediapipe_detection(frame, hands)
        
        # 1. Parse all detected hands in current frame safely
        h_img, w_img, _ = image.shape
        detected_hands = []
        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            num_handedness = len(results.multi_handedness) if results.multi_handedness else 0
            for idx, hand_curr in enumerate(results.multi_hand_landmarks):
                if idx < num_handedness and results.multi_handedness[idx].classification:
                    handedness = results.multi_handedness[idx].classification[0].label
                else:
                    handedness = 'Right' if idx % 2 == 0 else 'Left'
                    
                w_pt = (float(hand_curr.landmark[0].x), float(hand_curr.landmark[0].y))
                xs = [int(lm.x * w_img) for lm in hand_curr.landmark]
                ys = [int(lm.y * h_img) for lm in hand_curr.landmark]
                bbox = (int(max(0, min(xs) - 10)), int(max(0, min(ys) - 10)), int(min(w_img, max(xs) + 10)), int(min(h_img, max(ys) + 10)))
                detected_hands.append({
                    'wrist': w_pt,
                    'landmarks': hand_curr,
                    'handedness': handedness,
                    'bbox': bbox
                })
                
        # 2. Update persistent 2-hand ID-locking tracker
        if tracking_mode:
            locked_hands, rejected_hands = hand_tracker.update(detected_hands)
            # 3. Draw high-tech HUD annotations (Electric Cyan/Bright Green for user, Red for rejected)
            draw_locked_and_rejected_hands(image, locked_hands, rejected_hands)
        else:
            locked_hands = {0: None, 1: None}
            rejected_hands = []
            for i, d in enumerate(detected_hands[:2]):
                locked_hands[i] = {
                    'id': i,
                    'wrist': d['wrist'],
                    'lost_frames': 0,
                    'landmarks': d['landmarks'],
                    'handedness': d['handedness'],
                    'bbox': d['bbox']
                }
            hand_tracker.locked_hands = {0: None, 1: None}
            # Fallback standard landmark renderer (NO bounding box, NO crosshairs, NO lock reticles)
            draw_styled_landmarks(image, results)
        
        # prediction logic
        current_action = "NONE"  # Default to NONE
        
        try:
            # Case A: Always append keypoints (STRICTLY FROM LOCKED USER HANDS ONLY)
            keypoints = extract_keypoints(locked_hands)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            # Active user hands (locked hands detected in current frame)
            active_user_hands = [locked_hands[s] for s in (0, 1) if locked_hands[s] is not None and locked_hands[s]['landmarks'] is not None]
            
            if not active_user_hands:
                current_action = "NONE"
                predictions.clear()
                current_time = time.time()
                if (current_time - last_hand_seen_time) > 0.6:
                    # Hands are genuinely down/absent (> 600ms)
                    candidate_sign_idx = None
                    candidate_start_time = 0.0
                    hands_enter_time = 0.0
            else:
                current_time = time.time()
                last_hand_seen_time = current_time
                if hands_enter_time == 0.0:
                    hands_enter_time = current_time
                
                if len(sequence) == 30 and model:
                    # Metric Hand Tracking updates every frame using primary locked user hand
                    primary_hand = active_user_hands[0]['landmarks']
                    lm0 = primary_hand.landmark[0]
                    lm9 = primary_hand.landmark[9]
                    h, w, c = image.shape
                    u0, v0 = lm0.x * w, lm0.y * h
                    u9, v9 = lm9.x * w, lm9.y * h
                    
                    speed_mps, Z = metric_tracker.update(u0, v0, u9, v9, current_time)
                    last_avg_v = speed_mps

                    # Case B: Resting / Stationary hand guard
                    if last_avg_v < MIN_ACTIVE_VELOCITY:
                        pred_idx = None
                    else:
                        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                        top_idx = int(np.argmax(res))
                        max_confidence = float(res[top_idx])
                        
                        sorted_probs = np.sort(res)[::-1]
                        margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0
                        
                        if max_confidence >= threshold and margin >= CONFIDENCE_MARGIN:
                            if top_idx < len(actions):
                                pred_idx = top_idx
                            else:
                                pred_idx = None
                        else:
                            pred_idx = None

                    # Sign Hold Logic: Fast lock after SIGN_HOLD_DURATION (~0.25s)
                    if pred_idx is not None and pred_idx < len(actions):
                        if candidate_sign_idx == pred_idx:
                            candidate_last_seen = current_time
                            if (current_time - candidate_start_time) >= SIGN_HOLD_DURATION:
                                current_action = actions[pred_idx]
                            else:
                                current_action = "NONE"
                        else:
                            candidate_sign_idx = pred_idx
                            candidate_start_time = current_time
                            candidate_last_seen = current_time
                            current_action = "NONE"
                    else:
                        # Allow 250ms jitter tolerance before dropping candidate
                        if current_time - candidate_last_seen > 0.25:
                            candidate_sign_idx = None
                            candidate_start_time = 0.0
                        current_action = "NONE"

        except Exception as e:
            current_action = "NONE"
        
        # TTS Logic - Only speak if NOT "NONE"
        if current_action != "NONE":
            current_time = time.time()
            
            # Speak if:
            # 1. It's a completely different word than the last one we spoke.
            # 2. OR, it's the SAME word, but enough time (SPEAK_COOLDOWN) has passed since we last spoke it.
            if (current_action != last_spoken_word) or (current_time - last_spoken_time > SPEAK_COOLDOWN):
                display_word = get_display_name(current_action)
                
                if convo_tokens and convo_tokens[-1].lower() == display_word.lower():
                    # Ignore only if the same word is repeated consecutively (e.g. "We We")
                    last_spoken_word = current_action
                    last_spoken_time = current_time
                else:
                    # HARDCODE 'Unmuted': It NEVER comes in between any words; only at the end
                    if display_word.lower() == 'unmuted':
                        if not any(t.lower() == 'unmuted' for t in convo_tokens):
                            convo_tokens.append('Unmuted')
                    else:
                        # If 'Unmuted' was already added earlier, keep it at the end by inserting before it
                        if convo_tokens and convo_tokens[-1].lower() == 'unmuted':
                            convo_tokens.insert(len(convo_tokens) - 1, display_word)
                        else:
                            convo_tokens.append(display_word)

                    convo_velocities.append(last_avg_v)
                    last_spoken_word = current_action
                    last_spoken_time = current_time
                    last_token_time = current_time
                    # Reset candidate after confirming so next word requires fresh deliberate hold
                    candidate_sign_idx = None
                    candidate_start_time = 0.0
            
        # Visualization (Hidden during Stabilization Demo)
        if not stabilization_mode:
            cv2.rectangle(image, (0,0), (640, 60), (50, 50, 50), -1)  # Dark gray background
            
            hands_are_down = (time.time() - last_hand_seen_time) > 0.6
            idle_limit = IDLE_TIMEOUT_HANDS_DOWN if hands_are_down else IDLE_TIMEOUT_HANDS_UP
            
            display_phrase = compile_conversational_sentence(convo_tokens)
            if len(convo_tokens) > 0 and last_token_time > 0:
                time_left = max(0.0, idle_limit - (time.time() - last_token_time))
                phrase_text = f"Building: {display_phrase}  [{time_left:.1f}s]"
            elif convo_tokens:
                phrase_text = f"Building: {display_phrase}"
            else:
                phrase_text = "Building: (ready - show sign)"
            cv2.putText(image, phrase_text, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            if current_action != "NONE":
                display_text = f"Locked: {get_display_name(current_action)}"
                text_color = (0, 255, 0)  # Green
            elif candidate_sign_idx is not None and candidate_start_time > 0 and candidate_sign_idx < len(actions):
                elapsed_hold = time.time() - candidate_start_time
                pct = min(1.0, elapsed_hold / SIGN_HOLD_DURATION)
                display_text = f"Signing: {get_display_name(actions[candidate_sign_idx])} ({int(pct * 100)}%)"
                text_color = (0, 255, 255)  # Yellow progress
            else:
                display_text = "Status: ..."
                text_color = (200, 150, 100)  # Blueish-gray
                
            cv2.putText(image, display_text, (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
                       
            if tracking_mode:
                status_list = []
                for s in (0, 1):
                    if locked_hands[s] is not None and locked_hands[s]['landmarks'] is not None:
                        status_list.append(f"USER_H{s} ({locked_hands[s]['handedness']})")
                status_text = "Locked: " + (", ".join(status_list) if status_list else "SEARCHING...")
                cv2.putText(image, status_text, (10, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if status_list else (0, 255, 255), 1)
                
                if rejected_hands:
                    cv2.putText(image, f"[THREATS BLOCKED: {len(rejected_hands)}]", (10, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                cv2.putText(image, "Tracking: OFF (Standard)", (10, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)
            
            # Small velocity text on screen
            vel_text = f"3D Speed: {last_avg_v:.2f} m/s | Depth: {metric_tracker.current_Z:.2f} m"
            cv2.putText(image, vel_text, (10, 468), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        # VidStab Demo Mode Processing
        if stabilization_mode and stabilizer is not None:
            # We pass 'image' which has landmarks drawn on it, so they stabilize with the hands!
            display_frame = stabilizer.stabilize_frame(input_frame=image, smoothing_window=8, border_type='reflect')
            if display_frame is None:
                display_frame = image
            else:
                cv2.putText(display_frame, "[STABILIZATION: ACTIVE (FAST-LK)]", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            display_frame = image
            cv2.putText(display_frame, "[STABILIZATION: OFF (Press 'S')]", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

        if tracking_mode:
            cv2.putText(display_frame, "[TRACKING: ON (Press 'T')]", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(display_frame, "[TRACKING: OFF (Press 'T')]", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow('UNMUTED Prototype', display_frame)

        # Real-time Terminal Log Window
        log_canvas = np.zeros((350, 800, 3), dtype=np.uint8)
        current_logs = get_terminal_logs_snapshot()
        for i, msg in enumerate(current_logs):
            cv2.putText(log_canvas, msg, (10, 25 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.imshow('Terminal Logs', log_canvas)

        key = cv2.waitKey(10) & 0xFF
        
        current_time = time.time()
        hands_are_down = (current_time - last_hand_seen_time) > 0.6
        idle_limit = IDLE_TIMEOUT_HANDS_DOWN if hands_are_down else IDLE_TIMEOUT_HANDS_UP
        
        # Don't flush if user is actively in the middle of signing a word (> 0.10s)
        is_actively_signing = (candidate_sign_idx is not None and (current_time - candidate_start_time) > 0.10)
        is_idle = (not is_actively_signing) and ((current_time - last_token_time) > idle_limit if last_token_time > 0 else False)
        
        if (key == ord(' ') or is_idle) and len(convo_tokens) > 0:
            avg_vel = sum(convo_velocities) / len(convo_velocities) if convo_velocities else 0.0
            try:
                sentence_text = compile_conversational_sentence(convo_tokens)
                speak(sentence_text, velocity=avg_vel)
            except Exception as e:
                import traceback
                error_msg = f"Crash in Builder: {str(e)}"
                print(error_msg)
                print(traceback.format_exc())
                add_terminal_log(error_msg)
            convo_tokens.clear()
            convo_velocities.clear()
            last_token_time = 0
            candidate_sign_idx = None
            candidate_start_time = 0.0
            last_spoken_word = ""
            
        # Also reset state if the idle timeout cleared an empty sentence
        if is_idle and len(convo_tokens) == 0:
            last_token_time = 0
            last_spoken_word = ""
            candidate_sign_idx = None
            candidate_start_time = 0.0

        if key == ord('q'):
            break
        elif key == ord('s') or key == ord('S'):
            stabilization_mode = not stabilization_mode
            print(f"[STABILIZATION] Toggled: {'ENABLED' if stabilization_mode else 'DISABLED'}")
        elif key == ord('t') or key == ord('T'):
            tracking_mode = not tracking_mode
            print(f"[TRACKING MODE] Toggled: {'ENABLED' if tracking_mode else 'DISABLED'}")
            add_terminal_log(f"[TRACKING] System {'ENABLED' if tracking_mode else 'DISABLED'}")

    cap.release()
    cv2.destroyAllWindows()
