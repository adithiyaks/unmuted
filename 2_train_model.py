import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# --- CONFIGURATION ---
DATA_PATH = os.path.join('data')

# Automatically detect actions from the data folder
if os.path.exists(DATA_PATH):
    actions = np.array([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
else:
    print(f"Error: Data folder '{DATA_PATH}' not found. Please run 1_collect_data.py first.")
    exit()

if len(actions) == 0:
    print("Error: No action data found in the data folder. Please run 1_collect_data.py first.")
    exit()

print(f"Detected {len(actions)} actions: {actions.tolist()}")

no_sequences = 40
sequence_length = 30
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# --- DATA LOADING ---
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []

print("Loading data...")
try:
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
except FileNotFoundError:
    print(f"Error: Data not found in {DATA_PATH}. Please run 1_collect_data.py first.")
    exit()

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}")

# --- MODEL ARCHITECTURE ---
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# --- COMPILATION & TRAINING ---
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print("Starting training...")
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])

print("Training complete.")
model.summary()

# --- SAVING ---
model_path = os.path.join('models', 'unmuted_brain.h5')
model.save(model_path)
print(f"Model saved to {model_path}")
