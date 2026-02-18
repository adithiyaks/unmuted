import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("Importing mediapipe...")
import mediapipe as mp
print("Mediapipe imported.")

print("Accessing mp.solutions.holistic...")
try:
    mp_holistic = mp.solutions.holistic
    print("mp.solutions.holistic accessed.")
    
    print("Instantiating Holistic...")
    with mp_holistic.Holistic() as holistic:
        print("Holistic instantiated successfully.")
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()

print("Done.")
