import cv2
import numpy as np
import os
import time
import mediapipe as mp
import tensorflow as tensorflow
from tensorflow.keras.models import load_model
import pyttsx3
import threading

# --- CONFIGURATION ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
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

# --- TTS Setup (Using Windows Native Speech via PowerShell) ---
import subprocess

def speak(text):
    """Speaks text using Windows native TTS via PowerShell (bypasses pyttsx3 issues)"""
    def _speak():
        try:
            print(f"[TTS] Speaking: {text}")  # Debug
            # Use PowerShell to access Windows SAPI directly
            cmd = f'Add-Type -AssemblyName System.Speech; $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; $synth.Rate = 3; $synth.Speak("{text}")'
            subprocess.run(['powershell', '-Command', cmd], 
                          creationflags=subprocess.CREATE_NO_WINDOW,
                          capture_output=True)
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
            
            flattened = np.array([[lm.x, lm.y, lm.z] for lm in hand_curr.landmark]).flatten()
            
            if handedness == 'Right': # Swapped per user request
                lh = flattened
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

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Trying camera 0...")
    cap = cv2.VideoCapture(0)

# Hand model
with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

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
            else:
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:] # Keep last 30 frames
                
                if len(sequence) == 30 and model:
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

        cv2.imshow('UNMUTED Prototype', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
