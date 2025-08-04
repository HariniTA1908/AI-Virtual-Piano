import pygame
import numpy as np
import time
import cv2
import mediapipe as mp
import threading
import pickle
import os
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import speech_recognition as sr

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 800, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI-Enhanced Virtual Piano 🎹 with Voice Assistance")

# Initialize Pygame mixer for sound
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
LIGHT_BLUE = (173, 216, 230)
DARK_BLUE = (70, 130, 180)
GOLD = (255, 215, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Define the notes and their corresponding frequencies
notes = {
    "C4": 261.63, "C#4": 277.18, "D4": 293.66, "D#4": 311.13, "E4": 329.63,
    "F4": 349.23, "F#4": 369.99, "G4": 392.00, "G#4": 415.30, "A4": 440.00,
    "A#4": 466.16, "B4": 493.88, "C5": 523.25, "C#5": 554.37, "D5": 587.33,
    "D#5": 622.25, "E5": 659.25
}

# Key mappings
key_mappings = {
    pygame.K_z: ("Z", "C4", 0),
    pygame.K_s: ("S", "C#4", 1),
    pygame.K_x: ("X", "D4", 2),
    pygame.K_d: ("D", "D#4", 3),
    pygame.K_c: ("C", "E4", 4),
    pygame.K_v: ("V", "F4", 5),
    pygame.K_g: ("G", "F#4", 6),
    pygame.K_b: ("B", "G4", 7),
    pygame.K_h: ("H", "G#4", 8),
    pygame.K_n: ("N", "A4", 9),
    pygame.K_j: ("J", "A#4", 10),
    pygame.K_m: ("M", "B4", 11),
    pygame.K_COMMA: (",", "C5", 12),
    pygame.K_l: ("L", "C#5", 13),
    pygame.K_PERIOD: (".", "D5", 14),
    pygame.K_SEMICOLON: (";", "D#5", 15),
    pygame.K_SLASH: ("/", "E5", 16),
}

# Reverse mapping from note names to keys
note_to_key = {note_name: key for key, (_, note_name, _) in key_mappings.items()}

# Song database - sequences of notes with timing
songs = {
    "Happy Birthday": [
        {"note": "C4", "duration": 0.5},
        {"note": "C4", "duration": 0.5},
        {"note": "D4", "duration": 1.0},
        {"note": "C4", "duration": 1.0},
        {"note": "F4", "duration": 1.0},
        {"note": "E4", "duration": 2.0},
        {"note": "C4", "duration": 0.5},
        {"note": "C4", "duration": 0.5},
        {"note": "D4", "duration": 1.0},
        {"note": "C4", "duration": 1.0},
        {"note": "G4", "duration": 1.0},
        {"note": "F4", "duration": 2.0},
        {"note": "C4", "duration": 0.5},
        {"note": "C4", "duration": 0.5},
        {"note": "C5", "duration": 1.0},
        {"note": "A4", "duration": 1.0},
        {"note": "F4", "duration": 1.0},
        {"note": "E4", "duration": 1.0},
        {"note": "D4", "duration": 1.0},
        {"note": "A#4", "duration": 0.5},
        {"note": "A#4", "duration": 0.5},
        {"note": "A4", "duration": 1.0},
        {"note": "F4", "duration": 1.0},
        {"note": "G4", "duration": 1.0},
        {"note": "F4", "duration": 2.0},
    ],
    "Twinkle Twinkle": [
        {"note": "C4", "duration": 1.0},
        {"note": "C4", "duration": 1.0},
        {"note": "G4", "duration": 1.0},
        {"note": "G4", "duration": 1.0},
        {"note": "A4", "duration": 1.0},
        {"note": "A4", "duration": 1.0},
        {"note": "G4", "duration": 2.0},
        {"note": "F4", "duration": 1.0},
        {"note": "F4", "duration": 1.0},
        {"note": "E4", "duration": 1.0},
        {"note": "E4", "duration": 1.0},
        {"note": "D4", "duration": 1.0},
        {"note": "D4", "duration": 1.0},
        {"note": "C4", "duration": 2.0},
    ]
}

# Active notes (for visualization)
active_notes = {}

# Song playing variables
current_song = None
current_note_index = 0
last_note_time = 0
is_song_playing = False

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,  # Lower threshold for better detection
    min_tracking_confidence=0.5,  # Lower threshold for smoother tracking
    model_complexity=1  # Increased model complexity for better accuracy
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Variables for hand gesture processing
last_played_note = None
last_played_time = 0
cooldown_time = 0.1  # Cooldown time between note plays
sequence_history = deque(maxlen=10)  # Store last 10 notes for prediction

# AI MODEL CONFIGURATIONS
# Path for saving/loading models
MODEL_DIR = "piano_ai_models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# RNN model configuration
SEQUENCE_LENGTH = 5  # Number of notes to consider for prediction
rnn_model = None
is_prediction_active = False
predicted_notes = []
prediction_confidence = 0.0

# Finger mapping for piano key interaction
finger_landmarks = {
    "thumb": 4,  # Thumb tip
    "index": 8,  # Index finger tip
    "middle": 12,  # Middle finger tip
    "ring": 16,  # Ring finger tip
    "pinky": 20  # Pinky finger tip
}

# Initialize speech recognition
recognizer = sr.Recognizer()


# Function to generate a sine wave for a given frequency
def generate_sine_wave(frequency, duration=0.5, volume=0.5, attack=0.05, decay=0.1):
    sample_rate = 44100  # Audio sample rate (44.1 kHz)
    n_samples = int(sample_rate * duration)
    attack_samples = int(sample_rate * attack)
    decay_samples = int(sample_rate * decay)

    # Create time array
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # Create basic sine wave
    wave = np.sin(2 * np.pi * frequency * t)

    # Add harmonics for richer sound
    wave += 0.3 * np.sin(2 * np.pi * (2 * frequency) * t)  # First overtone (octave)
    wave += 0.15 * np.sin(2 * np.pi * (3 * frequency) * t)  # Second overtone

    # Normalize
    wave = wave / np.max(np.abs(wave))

    # Apply envelope (attack and decay)
    envelope = np.ones(n_samples)
    # Attack ramp up
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    # Decay ramp down
    if decay_samples > 0:
        envelope[-decay_samples:] = np.linspace(1, 0.7, decay_samples)

    # Apply envelope and volume
    wave = volume * wave * envelope

    return np.int16(wave * 32767)  # Convert to 16-bit audio


# Function to play a note
def play_note(note_name, duration=0.5):
    global sequence_history

    if note_name in notes:
        frequency = notes[note_name]
        wave = generate_sine_wave(frequency, duration, volume=0.6, attack=0.01, decay=0.1)

        # Convert 1D mono array to 2D stereo array
        stereo_wave = np.column_stack((wave, wave))
        sound = pygame.sndarray.make_sound(stereo_wave)

        # Play the sound
        sound.play()

        # Record the active note with current time
        active_notes[note_name] = time.time()

        # Add to sequence history for ML prediction
        sequence_history.append(note_name)

        # Find corresponding key
        for key, (_, key_note_name, _) in key_mappings.items():
            if key_note_name == note_name:
                active_notes[key] = time.time()
                break


# Function to start playing a song
def start_song(song_name):
    global current_song, current_note_index, last_note_time, is_song_playing
    if song_name in songs:
        current_song = song_name
        current_note_index = 0
        last_note_time = time.time()
        is_song_playing = True
        print(f"Now playing: {song_name}")
    else:
        print(f"Song '{song_name}' not found")


# Function to create and train RNN model for note prediction
def create_rnn_model():
    global rnn_model

    # Define features and labels
    note_to_int = {note: i for i, note in enumerate(notes.keys())}
    int_to_note = {i: note for note, i in note_to_int.items()}
    n_notes = len(notes)

    # Build model
    model = Sequential([
        LSTM(128, input_shape=(SEQUENCE_LENGTH, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(n_notes, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    rnn_model = model

    # Load pretrained model if exists
    model_path = os.path.join(MODEL_DIR, "rnn_note_predictor.h5")
    if os.path.exists(model_path):
        try:
            rnn_model = load_model(model_path)
            print("Loaded existing RNN model")
        except:
            print("Error loading RNN model, creating new one")
    else:
        print("Created new RNN model - will need training data")

    return note_to_int, int_to_note


# Function to predict next note using RNN
def predict_next_note():
    global predicted_notes, prediction_confidence, rnn_model, sequence_history

    if rnn_model is None or len(sequence_history) < SEQUENCE_LENGTH:
        return []

    # Convert notes to integers
    note_to_int = {note: i for i, note in enumerate(notes.keys())}
    int_to_note = {i: note for note, i in note_to_int.items()}

    # Prepare input sequence
    seq = list(sequence_history)[-SEQUENCE_LENGTH:]
    seq_ints = [note_to_int[note] for note in seq]
    x = np.array(seq_ints).reshape(1, SEQUENCE_LENGTH, 1) / len(notes)

    # Predict
    pred = rnn_model.predict(x, verbose=0)[0]
    prediction_confidence = np.max(pred)

    # Get top 3 predictions
    top_indices = pred.argsort()[-3:][::-1]
    predicted_notes = [int_to_note[idx] for idx in top_indices]
    return predicted_notes


# Train RNN model with demo data
def train_rnn_with_demo_data():
    global rnn_model

    print("Training RNN model with example data...")

    # Convert note sequences from demo songs to training data
    note_to_int = {note: i for i, note in enumerate(notes.keys())}
    int_to_note = {i: note for note, i in note_to_int.items()}
    n_notes = len(notes)

    # Collect all sequences from songs
    all_notes = []
    for song_name, note_sequence in songs.items():
        song_notes = [note_data["note"] for note_data in note_sequence]
        all_notes.extend(song_notes)

    # Create training sequences
    sequences = []
    next_notes = []

    for i in range(len(all_notes) - SEQUENCE_LENGTH):
        seq = all_notes[i:i + SEQUENCE_LENGTH]
        next_note = all_notes[i + SEQUENCE_LENGTH]

        # Convert to integers
        seq_ints = [note_to_int[note] for note in seq]
        next_int = note_to_int[next_note]

        sequences.append(seq_ints)
        next_notes.append(next_int)

    # Prepare numpy arrays
    X = np.array(sequences) / n_notes
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = np.eye(n_notes)[next_notes]  # One-hot encode

    # Train model
    if len(sequences) > 0:
        checkpoint = ModelCheckpoint(
            os.path.join(MODEL_DIR, "rnn_note_predictor.h5"),
            monitor='loss',
            save_best_only=True
        )

        rnn_model.fit(
            X, y,
            epochs=50,
            batch_size=32,
            callbacks=[checkpoint],
            verbose=1
        )
        print("RNN model trained")
    else:
        print("Not enough sequence data for training")


# Draw piano keys
def draw_piano():
    # Background
    screen.fill(BLACK)

    # Draw title
    font_title = pygame.font.SysFont('Arial', 28, bold=True)
    title = font_title.render("AI-Enhanced Virtual Piano 🎹 with Voice Assistance", True, GOLD)
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 10))

    # Calculate key dimensions
    white_keys = [k for k, (_, note_name, _) in key_mappings.items() if "#" not in note_name]
    key_width = WIDTH // len(white_keys)
    white_key_height = 200
    black_key_height = 120
    black_key_width = key_width * 0.6

    # Draw white keys first
    white_positions = {}
    current_pos = 0
    for key, (key_char, note_name, _) in sorted(key_mappings.items(), key=lambda x: x[1][2]):
        if "#" not in note_name:  # White keys
            # Check if note is active
            is_active = note_name in active_notes and time.time() - active_notes[note_name] < 0.5
            color = LIGHT_BLUE if is_active else WHITE

            # Handle prediction highlighting
            if is_prediction_active and note_name in predicted_notes:
                # Highlight predicted keys with varying intensity based on position in prediction list
                pred_idx = predicted_notes.index(note_name)
                if pred_idx == 0:
                    color = (200, 255, 200)  # Green tint for top prediction
                elif pred_idx == 1:
                    color = (255, 255, 200)  # Yellow tint for second prediction
                else:
                    color = (255, 230, 200)  # Orange tint for third prediction

            # Draw key
            key_rect = pygame.Rect(current_pos, HEIGHT - white_key_height, key_width, white_key_height)
            pygame.draw.rect(screen, color, key_rect)
            pygame.draw.rect(screen, BLACK, key_rect, 2)  # Border

            # Store position for reference
            white_positions[note_name[0]] = current_pos

            # Draw key label
            font = pygame.font.SysFont('Arial', 18, bold=True)
            key_label = font.render(key_char, True, BLACK)
            screen.blit(key_label, (current_pos + key_width // 2 - key_label.get_width() // 2,
                                    HEIGHT - 40))

            # Draw note name
            note_label = font.render(note_name, True, BLACK)
            screen.blit(note_label, (current_pos + key_width // 2 - note_label.get_width() // 2,
                                     HEIGHT - white_key_height + 20))

            current_pos += key_width

    # Draw black keys on top
    for key, (key_char, note_name, _) in key_mappings.items():
        if "#" in note_name:  # Black keys
            # Find the position based on the note
            base_note = note_name[0]

            # Calculate position
            if base_note in white_positions:
                pos = white_positions[base_note] + key_width - black_key_width // 2

                # Check if note is active
                is_active = note_name in active_notes and time.time() - active_notes[note_name] < 0.5
                color = LIGHT_BLUE if is_active else GRAY

                # Handle prediction highlighting
                if is_prediction_active and note_name in predicted_notes:
                    pred_idx = predicted_notes.index(note_name)
                    if pred_idx == 0:
                        color = (100, 200, 100)  # Green tint for top prediction
                    elif pred_idx == 1:
                        color = (200, 200, 100)  # Yellow tint for second prediction
                    else:
                        color = (200, 150, 100)  # Orange tint for third prediction

                # Draw key
                key_rect = pygame.Rect(pos, HEIGHT - white_key_height, black_key_width, black_key_height)
                pygame.draw.rect(screen, color, key_rect)
                pygame.draw.rect(screen, BLACK, key_rect, 2)  # Border

                # Draw key label
                font = pygame.font.SysFont('Arial', 16, bold=True)
                key_label = font.render(key_char, True, WHITE)
                screen.blit(key_label, (pos + black_key_width // 2 - key_label.get_width() // 2,
                                        HEIGHT - white_key_height + black_key_height - 30))

    # Display AI prediction info if active
    if is_prediction_active and predicted_notes:
        ai_font = pygame.font.SysFont('Arial', 16, bold=True)
        ai_text = ai_font.render(
            f"AI Prediction: {predicted_notes[0]} ({int(prediction_confidence * 100)}% confidence)",
            True, GREEN)
        screen.blit(ai_text, (10, 60))

        # Show secondary predictions
        if len(predicted_notes) > 1:
            secondary = ai_font.render(f"Also consider: {', '.join(predicted_notes[1:])}", True, GREEN)
            screen.blit(secondary, (10, 85))


# Function to handle direct keyboard representation for hand gestures
def detect_hand_gestures():
    global last_played_note, last_played_time

    # Open the camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Get actual camera dimensions
    camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    camera_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate piano keyboard dimensions for overlay
    key_width = camera_width // 10  # Width for white keys in camera view
    keyboard_height = int(camera_height * 0.3)  # Height of keyboard in camera view
    keyboard_y = int(camera_height * 0.6)
    # Y position of keyboard in camera view

    # Create mapping of keyboard positions
    keyboard_mapping = {}
    white_key_indices = [0, 2, 4, 5, 7, 9, 11, 12, 14, 16]  # Indices of white keys in one octave
    black_key_indices = [1, 3, 6, 8, 10, 13, 15]  # Indices of black keys in one octave

    # Define all note names in order
    all_note_names = ["C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4",
                      "C5", "C#5", "D5", "D#5", "E5"]

    # Map white keys
    white_key_width = camera_width // len(white_key_indices)
    for i, idx in enumerate(white_key_indices):
        note_name = all_note_names[idx]
        x1 = i * white_key_width
        x2 = x1 + white_key_width
        keyboard_mapping[note_name] = {
            'x1': x1,
            'x2': x2,
            'y1': keyboard_y,
            'y2': camera_height,
            'is_black': False
        }

    # Map black keys
    black_key_width = white_key_width * 0.6
    for idx in black_key_indices:
        note_name = all_note_names[idx]
        # Find the white key before this black key
        white_key_before = white_key_indices[
            max(0, white_key_indices.index(idx - 1 if idx - 1 in white_key_indices else 0))]
        white_key_idx = white_key_indices.index(white_key_before)
        x_center = (white_key_idx + 1) * white_key_width - white_key_width // 2
        x1 = x_center - black_key_width // 2
        x2 = x_center + black_key_width // 2
        keyboard_mapping[note_name] = {
            'x1': x1,
            'x2': x2,
            'y1': keyboard_y,
            'y2': keyboard_y + keyboard_height * 0.6,  # Black keys are shorter
            'is_black': True
        }

    # For finger tracking
    finger_positions = []
    max_positions = 5  # Positions to average (reduces jitter)

    # For displaying active notes
    active_visual_notes = {}

    # Finger state tracking (for continuous playing)
    finger_states = {finger: {'active': False, 'note': None, 'last_time': 0}
                     for finger in finger_landmarks.keys()}

    # Start camera loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)

        # Convert to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Draw virtual keyboard overlay

        # Draw white keys first
        for note_name, key_data in keyboard_mapping.items():
            if not key_data['is_black']:
                # Determine if note is active
                is_active = note_name in active_visual_notes and time.time() - active_visual_notes[note_name] < 0.5
                color = (200, 230, 255) if is_active else (255, 255, 255)
                cv2.rectangle(
                    frame,
                    (int(key_data['x1']), int(key_data['y1'])),
                    (int(key_data['x2']), int(key_data['y2'])),
                    color,
                    cv2.FILLED
                )
                cv2.rectangle(
                    frame,
                    (int(key_data['x1']), int(key_data['y1'])),
                    (int(key_data['x2']), int(key_data['y2'])),
                    (0, 0, 0),
                    2
                )

                # Add note name
                text_x = int((key_data['x1'] + key_data['x2']) // 2)
                text_y = int(camera_height - 30)
                cv2.putText(
                    frame,
                    note_name,
                    (text_x - 15, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA
                )

        # Draw black keys on top
        for note_name, key_data in keyboard_mapping.items():
            if key_data['is_black']:
                # Determine if note is active
                is_active = note_name in active_visual_notes and time.time() - active_visual_notes[note_name] < 0.5
                color = (100, 150, 200) if is_active else (50, 50, 50)
                cv2.rectangle(
                    frame,
                    (int(key_data['x1']), int(key_data['y1'])),
                    (int(key_data['x2']), int(key_data['y2'])),
                    color,
                    cv2.FILLED
                )
                cv2.rectangle(
                    frame,
                    (int(key_data['x1']), int(key_data['y1'])),
                    (int(key_data['x2']), int(key_data['y2'])),
                    (0, 0, 0),
                    2
                )

                # Add note name
                text_x = int((key_data['x1'] + key_data['x2']) // 2)
                text_y = int(key_data['y1'] + 30)
                cv2.putText(
                    frame,
                    note_name,
                    (text_x - 15, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        # Draw hand landmarks and detect key presses
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Track each finger tip
                current_time = time.time()
                for finger_name, landmark_id in finger_landmarks.items():
                    # Get finger position
                    finger_tip = hand_landmarks.landmark[landmark_id]
                    x, y = int(finger_tip.x * camera_width), int(finger_tip.y * camera_height)

                    # Draw finger marker with finger name
                    cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
                    cv2.putText(
                        frame,
                        finger_name[0].upper(),  # First letter of finger name
                        (x - 5, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA
                    )

                    # Check if finger is on a key and handle key press logic
                    for note_name, key_data in keyboard_mapping.items():
                        if (key_data['x1'] <= x <= key_data['x2'] and
                                key_data['y1'] <= y <= key_data['y2']):

                            # Check if this finger just entered this key or has been inactive
                            if (not finger_states[finger_name]['active'] or
                                    finger_states[finger_name]['note'] != note_name):

                                # Check cooldown time
                                if current_time - finger_states[finger_name]['last_time'] > cooldown_time:
                                    # Play the note
                                    play_note(note_name)

                                    # Update finger state
                                    finger_states[finger_name]['active'] = True
                                    finger_states[finger_name]['note'] = note_name
                                    finger_states[finger_name]['last_time'] = current_time

                                    # Mark for visualization
                                    active_visual_notes[note_name] = current_time

                                    # Display on screen
                                    cv2.putText(
                                        frame,
                                        f"Playing: {note_name}",
                                        (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7,
                                        (0, 255, 0),
                                        2,
                                        cv2.LINE_AA
                                    )
                            break
                    else:
                        # No key found under this finger
                        finger_states[finger_name]['active'] = False

                # Update AI prediction if applicable
                if is_prediction_active and len(sequence_history) >= SEQUENCE_LENGTH:
                    if current_time - last_played_time > 1.0:  # Only predict when there's a pause
                        predicted_notes = predict_next_note()

        # Add AI status information
        if is_prediction_active:
            cv2.putText(
                frame,
                "AI Mode: ON",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            # Display prediction if available
            if predicted_notes:
                prediction_text = f"Predicted: {predicted_notes[0]} ({int(prediction_confidence * 100)}%)"
                cv2.putText(
                    frame,
                    prediction_text,
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 255),
                    2,
                    cv2.LINE_AA
                )

            # Display current sequence
        if sequence_history:
            seq_text = f"Sequence: {', '.join(list(sequence_history)[-5:])}"
            cv2.putText(
                frame,
                seq_text,
                (10, camera_height - keyboard_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 200, 0),
                2,
                cv2.LINE_AA
            )

            # Show frame
        cv2.imshow("AI Piano Hand Tracking", frame)

        # Check for exit key
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

        # Clean up
    cap.release()
    cv2.destroyAllWindows()


# Function to listen for voice commands and play notes
def voice_assistance():
    global is_song_playing, current_song

    while True:
        with sr.Microphone() as source:
            print("Listening for voice commands...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

            try:
                # Recognize speech using Google Speech Recognition
                command = recognizer.recognize_google(audio).lower()
                print(f"Recognized command: {command}")

                # Handle commands
                if "play" in command:
                    # Extract note name from command
                    note_name = None
                    for note in notes.keys():
                        if note.lower() in command:
                            note_name = note
                            break

                    if note_name:
                        play_note(note_name)
                        print(f"Playing note: {note_name}")
                    else:
                        print("Note not recognized")

                elif "song" in command:
                    # Extract song name from command
                    song_name = None
                    for song in songs.keys():
                        if song.lower() in command:
                            song_name = song
                            break

                    if song_name:
                        start_song(song_name)
                        print(f"Playing song: {song_name}")
                    else:
                        print("Song not recognized")

                elif "stop" in command:
                    is_song_playing = False
                    current_song = None
                    print("Stopped playing")

            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")


# Main function to handle game loop
def main():
    print("Starting Virtual Piano...")
    global current_song, current_note_index, last_note_time, is_song_playing
    global is_prediction_active, rnn_model, predicted_notes, prediction_confidence

    # Initialize AI models
    note_to_int, int_to_note = create_rnn_model()

    # Start hand tracking in a separate thread
    hand_thread = threading.Thread(target=detect_hand_gestures)
    hand_thread.daemon = True
    hand_thread.start()

    # Start voice assistance in a separate thread
    voice_thread = threading.Thread(target=voice_assistance)
    voice_thread.daemon = True
    voice_thread.start()

    # Flag for training the model
    should_train_model = True

    running = True
    clock = pygame.time.Clock()

    # Create and train RNN model
    if should_train_model:
        train_thread = threading.Thread(target=train_rnn_with_demo_data)
        train_thread.daemon = True
        train_thread.start()

    # Main game loop
    while running:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_1:
                    start_song("Happy Birthday")
                elif event.key == pygame.K_2:
                    start_song("Twinkle Twinkle")
                elif event.key == pygame.K_p:
                    is_prediction_active = not is_prediction_active
                    print(f"AI Prediction mode: {'ON' if is_prediction_active else 'OFF'}")
                elif event.key in key_mappings:
                    # Play the note
                    _, note_name, _ = key_mappings[event.key]
                    play_note(note_name)
            elif event.type == pygame.KEYUP:
                # Remove key from active notes when released
                if event.key in active_notes:
                    del active_notes[event.key]

        # Handle song playback
        if is_song_playing and current_song:
            now = time.time()
            song_data = songs[current_song]

            if current_note_index < len(song_data):
                note_data = song_data[current_note_index]
                # Check if it's time to play the next note
                if now - last_note_time >= note_data["duration"]:
                    # Play the note
                    play_note(note_data["note"], note_data["duration"])

                    # Move to next note
                    current_note_index += 1
                    last_note_time = now
            else:
                # Song finished
                is_song_playing = False
                print(f"Finished playing: {current_song}")

        # Update active notes (remove old ones)
        now = time.time()
        keys_to_remove = []
        for key, start_time in active_notes.items():
            if now - start_time > 0.5:  # Note active duration
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del active_notes[key]

        # Update AI predictions if active
        if is_prediction_active and rnn_model and len(sequence_history) >= SEQUENCE_LENGTH:
            predicted_notes = predict_next_note()

        # Draw the piano
        draw_piano()

        # Add UI elements
        font = pygame.font.SysFont('Arial', 16)

        # Song controls
        controls_text = font.render("Press: 1 - Happy Birthday | 2 - Twinkle Twinkle | P - Toggle AI Prediction", True,
                                    WHITE)
        screen.blit(controls_text, (10, 40))

        # Current song status
        if is_song_playing:
            status_text = font.render(
                f"Now playing: {current_song} (Note {current_note_index + 1}/{len(songs[current_song])})", True, GREEN)
            screen.blit(status_text, (10, HEIGHT - 220))

        # Display last played sequence
        if sequence_history:
            seq_text = font.render(f"Last notes: {', '.join(list(sequence_history)[-5:])}", True, LIGHT_BLUE)
            screen.blit(seq_text, (10, HEIGHT - 240))

        # Update display
        pygame.display.flip()
        clock.tick(60)  # Limit to 60 FPS

    # Clean up before exit
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    import sys
main()