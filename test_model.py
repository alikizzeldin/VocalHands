import cv2
import numpy as np
import mediapipe as mp
from model import GestureRecognitionModel
import json
import pyttsx3

def load_letter_mapping():
    with open('processed_data/letter_mapping.json', 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    # Reverse the mapping to get letter from index
    return {v: k for k, v in mapping.items()}

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7
    )
    
    # Initialize text-to-speech
    engine = pyttsx3.init()
    
    # Load the model and letter mapping
    model = GestureRecognitionModel()
    model.load_model('gesture_model.h5')
    letter_mapping = load_letter_mapping()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    print("Starting real-time gesture recognition...")
    print("Press 'q' to quit")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract landmarks
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                # Predict gesture
                prediction = model.predict(np.array([landmarks]))
                predicted_letter = letter_mapping[prediction[0]]
                
                # Display prediction
                cv2.putText(frame, f"Letter: {predicted_letter}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                          1, (0, 255, 0), 2)
                
                # Speak the letter
                engine.say(predicted_letter)
                engine.runAndWait()
        
        # Display frame
        cv2.imshow('Gesture Recognition', frame)
        
        # Break loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 