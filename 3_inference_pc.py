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

# --- CONFIGURATION ---
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

    'judges': 'Judges',
    'unmuted': 'Unmuted',
    'we': 'We',
}

def get_display_name(action):
    """Get the display-friendly name for an action."""
    return display_names.get(action, action.replace('_', ' ').title())

sequence_length = 30
threshold = 0.85          # Prediction confidence threshold (filters out low-confidence noise)
CONFIDENCE_MARGIN = 0.35  # Top class must lead runner-up by at least 35% to prevent ambiguous guesses
CONSISTENCY_FRAMES = 6    # Must hold same high-confidence sign for 6 consecutive frames (~200ms)
MIN_ACTIVE_VELOCITY = 0.010 # Active movement gate; stationary hands stay in "Status: ..." and never default
SPEAK_COOLDOWN = 3.0      # Seconds before repeating the same word
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
el_client = ElevenLabs(api_key="sk_ccb3ef641959fec40ddf30a6c2c89b4181ae3fa079c57424")

# --- KINEMATICS & PROSODY TUNING ---
# Lower VELOCITY_MAX means less hand shaking required to reach peak Style and Speed
VELOCITY_MIN = 0.008       # Calm / slow baseline movement
VELOCITY_MAX = 0.065       # Increased from 0.050 (but lower than orig 0.080): balanced effort for max style
THRESHOLD_EXCITED = 0.045  # Velocity threshold to trigger excited exclamation

MIN_STYLE, MAX_STYLE = 0.05, 1.00
MIN_STABILITY, MAX_STABILITY = 0.25, 0.75
MIN_SPEED, MAX_SPEED = 0.85, 1.15

def speak(text, velocity=0.0):
    """Speaks text using ElevenLabs with dynamic emotion/prosody based on velocity."""
    def _speak():
        try:
            # 1. DYNAMIC EMOTION PARAMETER MAPPING
            norm_v = np.clip((velocity - VELOCITY_MIN) / (VELOCITY_MAX - VELOCITY_MIN), 0.0, 1.0)
            
            # Style: calm (0.05) -> excited (1.00)
            dynamic_style = MIN_STYLE + norm_v * (MAX_STYLE - MIN_STYLE)
            dynamic_style = round(float(dynamic_style), 2)
            
            # Stability: map velocity inversely to [0.75, 0.25]
            dynamic_stability = MAX_STABILITY - norm_v * (MAX_STABILITY - MIN_STABILITY)
            dynamic_stability = round(float(dynamic_stability), 2)
            
            # Speed: map velocity [0.85, 1.15]
            dynamic_speed = MIN_SPEED + norm_v * (MAX_SPEED - MIN_SPEED)
            dynamic_speed = round(float(dynamic_speed), 2)
            
            # 2. PROMPT DECORATION (EXCLAMATION INJECTION)
            if velocity >= THRESHOLD_EXCITED:
                log_text = f"[excited] {text.upper()}!"
                spoken_text = f"{text.upper()}!"
            else:
                log_text = f"{text}."
                spoken_text = f"{text}."
                
            print(f"[Voice Agent] Synthesizing: '{log_text}' | Style: {dynamic_style:.2f} | Stab: {dynamic_stability:.2f} | Spd: {dynamic_speed:.2f}")

            # 3. VOICE SETTINGS PAYLOAD
            audio = el_client.text_to_speech.convert(
                text=spoken_text,
                voice_id="JBFqnCBsd6RMkjVDRZzb",
                model_id="eleven_multilingual_v2",  # Best model for expressive nuance and emotion prompts
                voice_settings=VoiceSettings(
                    stability=dynamic_stability,
                    similarity_boost=0.75,
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
        except Exception as e:
            print(f"[TTS ERROR] {e}")
            
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

def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                     mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(250,44,250), thickness=2, circle_radius=2))

def extract_keypoints(results):
    lh = np.zeros(21*3)
    rh = np.zeros(21*3)
    
    if results.multi_hand_landmarks:
        for idx, hand_curr in enumerate(results.multi_hand_landmarks):
            # Checking classification (Left vs Right)
            handedness = results.multi_handedness[idx].classification[0].label
            
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
            
            if handedness == 'Right': # Swapped per user request
                lh = flattened
            else:
                rh = flattened
                
    return np.concatenate([lh, rh])

# --- KINEMATICS DEBUGGER ---
def debug_hand_kinematics(sequence):
    """Extract motion dynamics from a (30, 126) keypoint sequence and log them."""
    seq = np.array(sequence)  # (30, 126)
    
    # Left wrist: indices 0:3 (x, y, z),  Right wrist: indices 63:66 (x, y, z)
    left_wrist  = seq[:, 0:3]    # (30, 3)
    right_wrist = seq[:, 63:66]  # (30, 3)
    
    # Frame-to-frame Euclidean distances for each wrist
    left_deltas  = np.linalg.norm(np.diff(left_wrist,  axis=0), axis=1)  # (29,)
    right_deltas = np.linalg.norm(np.diff(right_wrist, axis=0), axis=1)  # (29,)
    
    # Only count a wrist if it's actually present (non-zero keypoints)
    left_present  = np.any(left_wrist != 0, axis=1).sum() > 1
    right_present = np.any(right_wrist != 0, axis=1).sum() > 1
    
    active_velocities = []
    if left_present:
        active_velocities.extend(left_deltas)
    if right_present:
        active_velocities.extend(right_deltas)
        
    if not active_velocities:
        return 0.0
        
    avg_v = np.mean(active_velocities)
    peak_v = np.max(active_velocities)
    
    # Acceleration / jerk (delta of velocity)
    if left_present and len(left_deltas) > 1:
        left_accel = np.diff(left_deltas)
    else:
        left_accel = [0]
    if right_present and len(right_deltas) > 1:
        right_accel = np.diff(right_deltas)
    else:
        right_accel = [0]
        
    active_accels = []
    if left_present: active_accels.extend(np.abs(left_accel))
    if right_present: active_accels.extend(np.abs(right_accel))
    
    peak_a = np.max(active_accels) if active_accels else 0.0
    
    # Spatial bounding area (max(x)-min(x) * max(y)-min(y) across all active non-zero keypoints)
    all_x = []
    all_y = []
    for frame in seq:
        lh_landmarks = frame[0:63].reshape(21, 3)
        rh_landmarks = frame[63:126].reshape(21, 3)
        for landmarks in [lh_landmarks, rh_landmarks]:
            if np.any(landmarks != 0):
                all_x.extend(landmarks[:, 0].tolist())
                all_y.extend(landmarks[:, 1].tolist())
                
    area = 0.0
    if all_x and all_y:
        span_x = max(all_x) - min(all_x)
        span_y = max(all_y) - min(all_y)
        area = span_x * span_y
        
    # print(f"[KINEMATICS] Avg Vel: {avg_v:.4f} | Peak Vel: {peak_v:.4f} | Peak Accel: {peak_a:.4f} | BBox Area: {area:.4f}")
    return avg_v

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

# VidStab Initialization
try:
    from vidstab import VidStab
    stabilizer = VidStab(kp_method='FAST')
except ImportError:
    print("[WARNING] vidstab not installed. Run: pip install vidstab")
    stabilizer = None

stabilization_mode = False  # Toggle via 's' key

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

# Hand model
with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        # Apply slight exposure / glare reduction
        if EXPOSURE_ALPHA != 1.0 or EXPOSURE_BETA != 0:
            frame = cv2.convertScaleAbs(frame, alpha=EXPOSURE_ALPHA, beta=EXPOSURE_BETA)

        # detection
        image, results = mediapipe_detection(frame, hands)
        
        # draw
        draw_styled_landmarks(image, results)
        
        # prediction logic
        current_action = "NONE"  # Default to NONE
        
        try:
            # Case A: No hands detected
            if not results.multi_hand_landmarks:
                current_action = "NONE"
                sequence.clear()
                predictions.clear()
            else:
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:] # Keep last 30 frames
                
                if len(sequence) == 30 and model:
                    last_avg_v = debug_hand_kinematics(sequence)
                    
                    # Case B: Resting / Stationary hand guard
                    # If hands are resting or motionless on screen, never guess an action
                    if last_avg_v < MIN_ACTIVE_VELOCITY:
                        current_action = "NONE"
                        predictions.append(None)
                    else:
                        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                        top_idx = int(np.argmax(res))
                        max_confidence = float(res[top_idx])
                        
                        # Margin between top-1 and runner-up class
                        sorted_probs = np.sort(res)[::-1]
                        margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0
                        
                        # Case C: Strict Confidence and Margin check
                        # If low confidence or ambiguous, NEVER default to any action
                        if max_confidence < threshold or margin < CONFIDENCE_MARGIN:
                            current_action = "NONE"
                            predictions.append(None) # Low confidence explicitly breaks any streak!
                        else:
                            predictions.append(top_idx)
                            
                            # Case D: Anti-Jitter Consistency Check across CONSISTENCY_FRAMES
                            if len(predictions) >= CONSISTENCY_FRAMES:
                                recent = predictions[-CONSISTENCY_FRAMES:]
                                if all(p == top_idx for p in recent):
                                    current_action = actions[top_idx]
                                else:
                                    current_action = "NONE"
                            else:
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
                speak(display_word, velocity=last_avg_v)
                last_spoken_word = current_action
                last_spoken_time = current_time
                # Notice: We DO NOT clear predictions here. If you hold the sign, it remains the current_action, 
                # but won't be spoken again until SPEAK_COOLDOWN passes!
            
        # Visualization (Hidden during Stabilization Demo)
        if not stabilization_mode:
            cv2.rectangle(image, (0,0), (640, 40), (50, 50, 50), -1)  # Dark gray background
            
            if current_action == "NONE":
                display_text = "Status: ..."
                text_color = (200, 150, 100)  # Blueish-gray
            else:
                display_text = get_display_name(current_action)
                text_color = (0, 255, 0)  # Green
                
            cv2.putText(image, display_text, (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
                       
            status_list = []
            if results.multi_handedness:
                for h in results.multi_handedness:
                    label = h.classification[0].label
                    if label == 'Left': status_list.append('Right')
                    else: status_list.append('Left')
            status_text = "Tracking: " + (", ".join(status_list) if status_list else "NONE")
            cv2.putText(image, status_text, (10, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
            
            # Small velocity text on screen
            vel_text = f"Vel: {last_avg_v:.3f}"
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

        cv2.imshow('UNMUTED Prototype', display_frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') or key == ord('S'):
            stabilization_mode = not stabilization_mode
            print(f"[STABILIZATION] Toggled: {'ENABLED' if stabilization_mode else 'DISABLED'}")

    cap.release()
    cv2.destroyAllWindows()
