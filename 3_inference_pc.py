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
    'judge': 'Judges',
    'hello': 'Hello',
    'help': 'Help',
    'i': 'I',
    'present': 'Present',
    'unmuted': 'Unmuted',
    'we': 'We',
}

def get_display_name(action):
    """Get the display-friendly name for an action."""
    return display_names.get(action, action.replace('_', ' ').title())

sequence_length = 30
threshold = 0.8
SPEAK_COOLDOWN = 3.0  # Seconds before repeating the same word
CAMERA_INDEX = None   # Set to 0, 1, 2 for a specific camera, or None for auto-detect

# Check if camera index passed via command line argument (e.g. `python 3_inference_pc.py 1`)
import sys
if len(sys.argv) > 1 and sys.argv[1].isdigit():
    CAMERA_INDEX = int(sys.argv[1])

# --- TTS Setup (Using Windows Native Speech via PowerShell) ---
import subprocess

def speak(text, sapi_rate=3, sapi_volume=80):
    """Speaks text using Windows native TTS via PowerShell.
    
    Args:
        text: The text to speak.
        sapi_rate: SAPI Rate property (-10 to 10). Default 3.
        sapi_volume: SAPI Volume property (0 to 100). Default 80.
    """
    def _speak():
        try:
            print(f"[TTS] Speaking: {text}  (rate={sapi_rate}, vol={sapi_volume})")  # Debug
            # Use PowerShell to access Windows SAPI directly
            cmd = (
                f'Add-Type -AssemblyName System.Speech; '
                f'$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                f'$synth.Rate = {sapi_rate}; '
                f'$synth.Volume = {sapi_volume}; '
                f'$synth.Speak("{text}")'
            )
            subprocess.run(['powershell', '-Command', cmd], 
                          creationflags=subprocess.CREATE_NO_WINDOW,
                          capture_output=True)
        except Exception as e:
            print(f"[TTS ERROR] {e}")
    threading.Thread(target=_speak, daemon=True).start()

# --- MEDIAPIPE SETUP (HANDS + SELFIE SEGMENTATION) ---
mp_hands = mp.solutions.hands
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    rgb.flags.writeable = False
    results = model.process(rgb)
    return image, results, rgb

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
            
            flattened = np.array([[lm.x, lm.y, lm.z] for lm in hand_curr.landmark]).flatten()
            
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
        return
        
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
        
    print(f"[KINEMATICS] Avg Vel: {avg_v:.4f} | Peak Vel: {peak_v:.4f} | Peak Accel: {peak_a:.4f} | BBox Area: {area:.4f}")

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

# Vision Models (Hands Tracking + Real Hand Semantic Segmentation)
with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
     mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_seg:
    
    # Pre-allocated structuring elements for high-FPS mask operations
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    while cap.isOpened():
        loop_start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # detection
        image, results, rgb_frame = mediapipe_detection(frame, hands)
        
        # draw
        draw_styled_landmarks(image, results)
        
        # prediction logic
        current_action = "NONE"  # Default to NONE
        
        try:
            # Case A: No hands detected
            if not results.multi_hand_landmarks:
                current_action = "NONE"
            else:
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:] # Keep last 30 frames
                
                if len(sequence) == 30 and model:
                    debug_hand_kinematics(sequence)
                    res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                    predictions.append(np.argmax(res))
                    
                    # Case B: Low confidence check
                    max_confidence = res[np.argmax(res)]
                    
                    if max_confidence < threshold:
                        current_action = "NONE"
                    else:
                        # Check consistency (Anti-Jitter)
                        # Check if the last 2 predictions are the same (ULTRA FAST)
                        if len(predictions) >= 2 and np.unique(predictions[-2:])[0] == np.argmax(res): 
                            current_action = actions[np.argmax(res)]
                        else:
                            current_action = "NONE"

        except Exception as e:
            current_action = "NONE"
        
        # TTS Logic - Only speak if NOT "NONE"
        if current_action != "NONE":
            current_time = time.time()
            # Speak if: new word OR same word but cooldown passed
            if (current_action != last_spoken_word) or (current_time - last_spoken_time > SPEAK_COOLDOWN):
                speak(get_display_name(current_action))
                last_spoken_word = current_action
                last_spoken_time = current_time
            
        # Visualization
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
        cv2.putText(image, status_text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)

        # --- SPATIAL VISION HUD (REAL HAND SEMANTIC SEGMENTATION) ---
        visual_hud = np.zeros_like(frame)
        
        if results.multi_hand_landmarks:
            h, w, _ = frame.shape
            hand_roi = np.zeros((h, w), dtype=np.uint8)
            for hand_landmarks in results.multi_hand_landmarks:
                points = []
                for lm in hand_landmarks.landmark:
                    px, py = int(lm.x * w), int(lm.y * h)
                    points.append([px, py])
                
                points = np.array(points, dtype=np.int32)
                hull = cv2.convexHull(points)
                cv2.fillPoly(hand_roi, [hull], 255)
            
            # Dilate ROI mask to encompass the full outer contours of all fingers
            hand_roi = cv2.dilate(hand_roi, dilate_kernel)
            
            # Semantic segmentation on actual camera frame pixels
            seg_results = selfie_seg.process(rgb_frame)
            seg_mask = seg_results.segmentation_mask  # float32 [0.0, 1.0]
            
            # Isolate hand pixels from person segmentation gated by hand ROI
            hand_mask = np.where((seg_mask > 0.35) & (hand_roi > 0), 255, 0).astype(np.uint8)
            
            if np.any(hand_mask > 0):
                # Clean up mask edges
                hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_CLOSE, morph_kernel)
                
                # Glowing Aura (51x51 Gaussian Blur) + Solid Core (~240-255 intensity)
                aura = cv2.GaussianBlur(hand_mask, (51, 51), 0)
                core = cv2.GaussianBlur(hand_mask, (7, 7), 0)
                
                # Smooth glowing transition: solid white core fading into outer halo
                combined = np.clip(core.astype(np.float32) * 0.85 + aura.astype(np.float32) * 0.55, 0, 250).astype(np.uint8)
                visual_hud = cv2.merge([combined, combined, combined])
        
        # Calculate latency and FPS
        latency = (time.time() - loop_start_time) * 1000
        fps = 1000.0 / latency if latency > 0 else 0
        
        # Telemetry HUD
        cv2.putText(visual_hud, "EDGE_AI // SPATIAL_OCCUPANCY_MASK", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(visual_hud, f"LATENCY: {latency:.1f}ms | FPS: {fps:.1f}", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

        cv2.imshow('UNMUTED Prototype', image)
        cv2.imshow('UNMUTED - Spatial Vision HUD', visual_hud)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
