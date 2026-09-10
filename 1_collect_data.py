import os
# Removing the protobuf python implementation force - this was likely breaking the import
# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import mediapipe as mp
import time

print("Script starting...")
print("Imported modules.")

# --- CONFIGURATION ---
DATA_PATH = os.path.join('data')
no_sequences = 40
sequence_length = 30
start_folder = 0
EXPOSURE_ALPHA = 0.88   # Contrast scale (1.0 = normal, 0.85-0.90 = reduces harsh glare)
EXPOSURE_BETA = -20     # Brightness offset (0 = normal, -15 to -30 = slightly dimmer)

print("Configuration set.")

# --- MEDIAPIPE SETUP (HANDS ONLY) ---
print("Setting up MediaPipe Hands...")
try:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    print("MediaPipe Hands setup complete.")
except Exception as e:
    print(f"FAILED to setup MediaPipe: {e}")
    exit()

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
            # MediaPipe hands label: "Left" means it looks like a left hand.
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
            
            if handedness == 'Right':
                lh = flattened
            else:
                rh = flattened
                
    return np.concatenate([lh, rh])

# --- MAIN DATA COLLECTION LOOP ---
def collect_data_for_action(action, cap, hands):
    """Collect data for a single action/hand sign."""
    print(f"\n{'='*50}")
    print(f"Collecting data for: '{action}'")
    print(f"Will collect {no_sequences} sequences of {sequence_length} frames each.")
    print(f"{'='*50}\n")
    
    # Create directories for this action
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass
    print(f"Directories created for '{action}'.")
    
    quit_requested = False
    
    # Loop through sequences aka videos
    for sequence in range(start_folder, start_folder + no_sequences):
        if quit_requested:
            break
            
        # Loop through video length aka sequence length
        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed.")
                quit_requested = True
                break

            # Apply slight exposure / glare reduction
            if EXPOSURE_ALPHA != 1.0 or EXPOSURE_BETA != 0:
                frame = cv2.convertScaleAbs(frame, alpha=EXPOSURE_ALPHA, beta=EXPOSURE_BETA)

            # Make detections
            image, results = mediapipe_detection(frame, hands)

            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # Logic for display status (Left/Right detected?)
            status_text = "Tracking: "
            if results.multi_handedness:
                detected_hands = []
                for h in results.multi_handedness:
                    label = h.classification[0].label
                    # Visually swap for FPV natural understanding
                    if label == 'Left': detected_hands.append('Right')
                    else: detected_hands.append('Left')
                    
                status_text += ", ".join(detected_hands)
            else:
                status_text += "NONE"
            
            cv2.putText(image, status_text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            
            # Apply wait logic
            if frame_num == 0:
                cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15,12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                # Show to screen
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(2000) # 2 Second pause
            else: 
                cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15,12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                # Show to screen
                cv2.imshow('OpenCV Feed', image)
            
            # Export keypoints
            keypoints = extract_keypoints(results)
            npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
            np.save(npy_path, keypoints)

            # Break gracefully on 'q'
            if cv2.waitKey(10) & 0xFF == ord('q'):
                quit_requested = True
                break
    
    if not quit_requested:
        print(f"\n✓ Finished collecting data for '{action}'!")
    
    return quit_requested


def main():
    """Main function to run interactive data collection."""
    print("\n" + "="*60)
    print("      HAND SIGN DATA COLLECTION - Interactive Mode")
    print("="*60)
    print("\nThis script will collect training data for hand signs.")
    print(f"Each sign will have {no_sequences} sequences of {sequence_length} frames.")
    print("Press 'q' during collection to quit.")
    print("Type 'quit' or 'exit' when prompted for a sign name to stop.\n")
    
    # Check for existing data
    if os.path.exists(DATA_PATH):
        existing_signs = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
        if existing_signs:
            print(f"Existing signs in data folder: {existing_signs}")
    
    print("Attempting to open camera...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open camera 1. Trying camera 0...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera 0 either. Exiting.")
            return
    print("Camera opened successfully.\n")
    
    # Initialize MediaPipe Hands
    print("Initializing MediaPipe Hands (Max 2 hands)...")
    
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        collected_signs = []
        
        while True:
            # Ask for the hand sign name
            print("\n" + "-"*40)
            sign_name = input("Enter the name of the hand sign to collect (or 'quit' to exit): ").strip().lower()
            
            if sign_name in ['quit', 'exit', 'q', '']:
                if sign_name == '':
                    print("Empty name entered. Please enter a valid sign name or 'quit' to exit.")
                    continue
                print("\nExiting data collection...")
                break
            
            # Validate the sign name (remove special characters except underscore)
            sign_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in sign_name)
            
            # Check if data already exists for this sign
            sign_path = os.path.join(DATA_PATH, sign_name)
            if os.path.exists(sign_path):
                existing_sequences = len([d for d in os.listdir(sign_path) if os.path.isdir(os.path.join(sign_path, d))])
                print(f"\n⚠ Warning: Data for '{sign_name}' already exists ({existing_sequences} sequences).")
                overwrite = input("Do you want to overwrite it? (yes/no): ").strip().lower()
                if overwrite not in ['yes', 'y']:
                    print("Skipping this sign...")
                    continue
            
            # Collect data for this sign
            quit_requested = collect_data_for_action(sign_name, cap, hands)
            
            if quit_requested:
                print("\nCollection interrupted by user (pressed 'q').")
                break
            
            collected_signs.append(sign_name)
            print(f"\nSigns collected so far: {collected_signs}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Summary
    print("\n" + "="*60)
    print("                    DATA COLLECTION COMPLETE")
    print("="*60)
    if collected_signs:
        print(f"Collected data for {len(collected_signs)} sign(s): {collected_signs}")
    else:
        print("No new signs were collected.")
    print(f"Data saved to: {os.path.abspath(DATA_PATH)}")
    print("\nNext step: Run 2_train_model.py to train your model.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
