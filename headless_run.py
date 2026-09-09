"""
UNMUTED — Headless Inference Runner
Runs the sign-language inference pipeline without any OpenCV GUI.
Suitable for headless servers, Raspberry Pi, or CI/demo environments.

Usage:
    python headless_run.py
    python headless_run.py --camera 0
    python headless_run.py --duration 60   # Run for 60 seconds then exit
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import sys
import time
import argparse
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

print(f"[HEADLESS] Loaded {len(actions)} actions: {actions.tolist()}")

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

# --- TTS Setup (Using Windows Native Speech via PowerShell) ---

def speak(text, sapi_rate=3, sapi_volume=80):
    """Speaks text using Windows native TTS via PowerShell.

    Args:
        text: The text to speak.
        sapi_rate: SAPI Rate property (-10 to 10). Default 3.
        sapi_volume: SAPI Volume property (0 to 100). Default 80.
    """
    def _speak():
        try:
            print(f"[TTS] Speaking: {text}  (rate={sapi_rate}, vol={sapi_volume})")
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

# --- MEDIAPIPE SETUP (HANDS ONLY) ---
mp_hands = mp.solutions.hands

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return image, results

def extract_keypoints(results):
    lh = np.zeros(21*3)
    rh = np.zeros(21*3)

    if results.multi_hand_landmarks:
        for idx, hand_curr in enumerate(results.multi_hand_landmarks):
            # Checking classification (Left vs Right)
            handedness = results.multi_handedness[idx].classification[0].label

            flattened = np.array([[lm.x, lm.y, lm.z] for lm in hand_curr.landmark]).flatten()

            if handedness == 'Right':  # Swapped per user request
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

def main():
    parser = argparse.ArgumentParser(description='UNMUTED Headless Inference Runner')
    parser.add_argument('--camera', type=int, default=1, help='Camera index (default: 1, falls back to 0)')
    parser.add_argument('--duration', type=int, default=0, help='Max runtime in seconds (0 = run forever)')
    parser.add_argument('--no-tts', action='store_true', help='Disable TTS output (log-only mode)')
    args = parser.parse_args()

    # Load Model
    model_path = os.path.join('models', 'unmuted_brain.h5')
    try:
        model = load_model(model_path)
        print("[HEADLESS] Model loaded.")
    except:
        print("[HEADLESS] WARNING: Model not found. Run 2_train_model.py first!")
        model = None

    # Variables
    sequence = []
    predictions = []
    last_spoken_word = ""
    last_spoken_time = 0
    frame_count = 0
    start_time = time.time()

    # Open camera with fallback
    def open_camera(target_idx):
        for idx in [target_idx, 0, 1]:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    return cap, idx
                cap.release()
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    return cap, idx
                cap.release()
        return None, -1

    cap, used_idx = open_camera(args.camera)
    if cap is None:
        print("[HEADLESS] ERROR: Cannot open any working camera. Exiting.")
        sys.exit(1)

    print(f"[HEADLESS] Camera opened (Index {used_idx}). Running inference (Ctrl+C to stop)...")
    if args.duration > 0:
        print(f"[HEADLESS] Will auto-stop after {args.duration} seconds.")

    # Hand model
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        try:
            while cap.isOpened():
                # Duration check
                if args.duration > 0 and (time.time() - start_time) > args.duration:
                    print(f"\n[HEADLESS] Duration limit ({args.duration}s) reached. Stopping.")
                    break

                ret, frame = cap.read()
                if not ret:
                    print("[HEADLESS] Camera read failed.")
                    break

                frame_count += 1

                # Detection (no drawing in headless mode)
                _, results = mediapipe_detection(frame, hands)

                # Prediction logic
                current_action = "NONE"

                try:
                    if not results.multi_hand_landmarks:
                        current_action = "NONE"
                    else:
                        keypoints = extract_keypoints(results)
                        sequence.append(keypoints)
                        sequence = sequence[-30:]  # Keep last 30 frames

                        if len(sequence) == 30 and model:
                            debug_hand_kinematics(sequence)
                            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                            predictions.append(np.argmax(res))

                            max_confidence = res[np.argmax(res)]

                            if max_confidence < threshold:
                                current_action = "NONE"
                            else:
                                # Anti-jitter: last 2 predictions must agree
                                if len(predictions) >= 2 and np.unique(predictions[-2:])[0] == np.argmax(res):
                                    current_action = actions[np.argmax(res)]
                                else:
                                    current_action = "NONE"

                except Exception as e:
                    current_action = "NONE"

                # TTS Logic - Only speak if NOT "NONE"
                if current_action != "NONE":
                    current_time = time.time()
                    if (current_action != last_spoken_word) or (current_time - last_spoken_time > SPEAK_COOLDOWN):
                        display = get_display_name(current_action)
                        print(f"[DETECT] Frame {frame_count}: {display}")

                        if not args.no_tts:
                            speak(display)

                        last_spoken_word = current_action
                        last_spoken_time = current_time

                # Periodic heartbeat (every 300 frames ≈ 10s at 30fps)
                if frame_count % 300 == 0:
                    elapsed = time.time() - start_time
                    print(f"[HEARTBEAT] Frame {frame_count} | "
                          f"Elapsed: {elapsed:.0f}s | "
                          f"Last detected: {get_display_name(last_spoken_word) if last_spoken_word else 'None'}")

        except KeyboardInterrupt:
            print("\n[HEADLESS] Interrupted by user.")

    cap.release()
    elapsed = time.time() - start_time
    print(f"[HEADLESS] Stopped. Processed {frame_count} frames in {elapsed:.1f}s "
          f"({frame_count/max(elapsed,1):.1f} FPS)")


if __name__ == "__main__":
    main()
