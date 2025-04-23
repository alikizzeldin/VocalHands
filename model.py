import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import json
import cv2
import mediapipe as mp
from tqdm import tqdm

def collect_gesture_data(gesture_id, num_samples=100):
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Collect data
    gesture_data = []
    sample_count = 0
    
    print(f"\nCollecting data for gesture {gesture_id}")
    print("Press 'q' to stop data collection")
    print(f"Samples needed: {num_samples}")
    
    while sample_count < num_samples:
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract landmarks
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                gesture_data.append(landmarks)
                sample_count += 1
                
                # Display progress
                cv2.putText(frame, f"Samples: {sample_count}/{num_samples}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Data Collection', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    return np.array(gesture_data)

class GestureRecognitionModel:
    def __init__(self, num_classes=28):  # 28 Arabic letters
        self.num_classes = num_classes
        self.model = self.build_model()
        self.gesture_dict = {
            0: "أ", 1: "ب", 2: "ت", 3: "ث", 4: "ج",
            5: "ح", 6: "خ", 7: "د", 8: "ذ", 9: "ر",
            10: "ز", 11: "س", 12: "ش", 13: "ص", 14: "ض",
            15: "ط", 16: "ظ", 17: "ع", 18: "غ", 19: "ف",
            20: "ق", 21: "ك", 22: "ل", 23: "م", 24: "ن",
            25: "ه", 26: "و", 27: "ي"
        }
    
    def build_model(self):
        model = models.Sequential([
            # Input layer - 21 landmarks * 3 coordinates (x, y, z) = 63 features
            layers.Dense(256, activation='relu', input_shape=(63,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_landmarks(self, landmarks):
        # Convert MediaPipe landmarks to numpy array
        # Each hand has 21 landmarks, each with x, y, z coordinates
        hand_data = []
        for landmark in landmarks.landmark:
            hand_data.extend([landmark.x, landmark.y, landmark.z])
        
        return np.array(hand_data)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        # Add data augmentation
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1
        )
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=5,
                    min_lr=0.00001
                )
            ]
        )
        
        return history
    
    def save_model(self, model_path='gesture_model.h5'):
        self.model.save(model_path)
        # Save gesture dictionary
        with open('gesture_dict.json', 'w', encoding='utf-8') as f:
            json.dump(self.gesture_dict, f, ensure_ascii=False)
    
    def load_model(self, model_path='gesture_model.h5'):
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            # Load gesture dictionary
            if os.path.exists('gesture_dict.json'):
                with open('gesture_dict.json', 'r', encoding='utf-8') as f:
                    self.gesture_dict = json.load(f)
            return True
        return False
    
    def predict_gesture(self, landmarks):
        # Preprocess landmarks
        processed_landmarks = self.preprocess_landmarks(landmarks)
        
        # Make prediction
        prediction = self.model.predict(np.expand_dims(processed_landmarks, axis=0))
        gesture_id = np.argmax(prediction)
        confidence = prediction[0][gesture_id]
        
        return gesture_id, confidence, self.gesture_dict.get(str(gesture_id), "Unknown")

class BasicWordsModel:
    def __init__(self, num_classes=12):  # 12 basic words/phrases
        self.num_classes = num_classes
        self.model = self.build_model()
        self.words_dict = {
            0: "Thanks",
            1: "Sorry",
            2: "Salam aleikum",
            3: "Not bad",
            4: "I am sorry",
            5: "I am pleased to meet you",
            6: "I am fine",
            7: "How are you",
            8: "Good morning",
            9: "Good evening",
            10: "Good bye",
            11: "Alhamdulillah"
        }
    
    def build_model(self):
        model = models.Sequential([
            # Input layer - 42 landmarks * 3 coordinates (x, y, z) = 126 features (for two hands)
            layers.Dense(256, activation='relu', input_shape=(126,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_landmarks(self, landmarks):
        # Convert MediaPipe landmarks to numpy array
        # Each hand has 21 landmarks, each with x, y, z coordinates
        hand_data = []
        for landmark in landmarks.landmark:
            hand_data.extend([landmark.x, landmark.y, landmark.z])
        
        # Pad with zeros if only one hand is detected (to match the expected 126 features)
        while len(hand_data) < 126:
            hand_data.extend([0.0, 0.0, 0.0])
        
        return np.array(hand_data)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        # Add data augmentation
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1
        )
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=5,
                    min_lr=0.00001
                )
            ]
        )
        
        return history
    
    def save_model(self, model_path='basic_words_model.h5'):
        self.model.save(model_path)
        # Save words dictionary
        with open('basic_words_dict.json', 'w', encoding='utf-8') as f:
            json.dump(self.words_dict, f, ensure_ascii=False)
    
    def load_model(self, model_path='basic_words_model.h5'):
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            # Load words dictionary
            if os.path.exists('basic_words_dict.json'):
                with open('basic_words_dict.json', 'r', encoding='utf-8') as f:
                    self.words_dict = json.load(f)
            return True
        return False
    
    def predict_gesture(self, landmarks):
        # Preprocess landmarks
        processed_landmarks = self.preprocess_landmarks(landmarks)
        
        # Reshape the input to match the expected shape (batch_size, features)
        processed_landmarks = processed_landmarks.reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(processed_landmarks)
        gesture_id = np.argmax(prediction)
        confidence = prediction[0][gesture_id]
        
        return gesture_id, confidence, self.words_dict.get(str(gesture_id), "Unknown")

class ArabicLettersModel(GestureRecognitionModel):
    def __init__(self):
        super().__init__(num_classes=28)  # 28 Arabic letters
        self.gesture_dict = {
            0: "Alif", 1: "Ba", 2: "Ta", 3: "Tha", 4: "Jeem",
            5: "Ha", 6: "Kha", 7: "Dal", 8: "Dhal", 9: "Ra",
            10: "Zay", 11: "Seen", 12: "Sheen", 13: "Sad", 14: "Dad",
            15: "Ta", 16: "Dha", 17: "Ayn", 18: "Ghayn", 19: "Fa",
            20: "Qaf", 21: "Kaf", 22: "Lam", 23: "Meem", 24: "Noon",
            25: "Ha", 26: "Waw", 27: "Ya"
        }
    
    def load_model(self, model_path='arabic_letters_model.h5'):
        return super().load_model(model_path)
    
    def save_model(self, model_path='arabic_letters_model.h5'):
        super().save_model(model_path) 