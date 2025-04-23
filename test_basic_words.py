import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
import traceback
from model import BasicWordsModel

def main():
    try:
        print("Loading class names...")
        # Load class names
        with open('basic_words_classes.json', 'r', encoding='utf-8') as f:
            class_names = json.load(f)
        print(f"Loaded {len(class_names)} classes: {class_names}")
        
        # Arabic translations
        arabic_translations = {
            "Thanks": "شكراً",
            "Sorry": "عذراً",
            "Salam aleikum": "السلام عليكم",
            "Not bad": "ليس سيئاً",
            "I am sorry": "أنا آسف",
            "I am pleased to meet you": "سعيد بلقائك",
            "I am fine": "أنا بخير",
            "How are you": "كيف حالك",
            "Good morning": "صباح الخير",
            "Good evening": "مساء الخير",
            "Good bye": "مع السلامة",
            "Alhamdulillah": "الحمد لله"
        }
        
        print("Initializing MediaPipe Hands...")
        # Initialize MediaPipe Hands
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        print("Loading the trained model...")
        # Load the trained model
        model = BasicWordsModel()
        model.load_model()
        print("Model loaded successfully!")
        
        print("Initializing webcam...")
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not open webcam")
        print("Webcam initialized successfully!")
        
        # Initialize variables for gesture recognition
        gesture_buffer = []
        buffer_size = 5
        current_gesture = None
        confidence_threshold = 0.7
        
        print("\nStarting real-time gesture recognition...")
        print("Press 'q' to quit")
        print("Press 's' to switch to alphabet model")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Add switch button
            cv2.rectangle(frame, (10, 60), (200, 100), (0, 255, 0), -1)
            cv2.putText(frame, "كلمات بسيطة", (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Convert the BGR image to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe
            results = hands.process(frame_rgb)
            
            # Draw hand landmarks and make prediction
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    try:
                        # Make prediction using predict_gesture method
                        gesture_id, confidence, gesture_name = model.predict_gesture(hand_landmarks)
                        
                        # Add to buffer
                        gesture_buffer.append(gesture_id)
                        if len(gesture_buffer) > buffer_size:
                            gesture_buffer.pop(0)
                        
                        # Get most common gesture in buffer
                        if len(gesture_buffer) == buffer_size:
                            counts = np.bincount(gesture_buffer)
                            most_common = np.argmax(counts)
                            if counts[most_common] >= buffer_size * 0.6 and confidence > confidence_threshold:
                                current_gesture = gesture_name
                            else:
                                current_gesture = None
                    except Exception as e:
                        print(f"Error during prediction: {str(e)}")
                        traceback.print_exc()
            
            # Display the current gesture in both English and Arabic
            if current_gesture:
                arabic_text = arabic_translations.get(current_gesture, current_gesture)
                cv2.putText(frame, f"English: {current_gesture}", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Arabic: {arabic_text}", (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Hand Gesture Recognition', frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Switch to alphabet model
                print("Switching to alphabet model...")
                import sign_language_translator
                sign_language_translator.main()
                break
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()
    finally:
        # Release resources
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        print("\nProgram terminated.")

if __name__ == "__main__":
    main() 